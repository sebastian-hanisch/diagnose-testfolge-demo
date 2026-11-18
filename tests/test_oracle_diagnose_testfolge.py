"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2): die exakte Strategie und die Basisratenrechnung gegen andere Rechenwege.

1. Aufzählung aller Strategien als explizite Entscheidungsbäume auf Kleinstinstanzen; jeder Baum wird einzeln über gemeinsame Wahrscheinlichkeiten ausgewertet (ohne
   Posterior, ohne Normierung): das Minimum über alle Bäume ist der Optimalwert der DP.
2. Monte-Carlo-Simulation der DP-Strategie mit festem Zufalls-Seed (SplitMix64): Mittel der realisierten Kosten gegen den DP-Wert innerhalb von vier Standardfehlern.
3. Rauschfreier Grenzfall (Se = Sp = 1, sehr großer Irrtumspreis): Teilmengen-DP über die Diagnosenmengen (andere Zustandsdarstellung) gegen die DP über Ausgänge.
4. Basisratenrechnung gegen Bruchrechnung (fractions) und gegen die Simulation einer Bevölkerung mit Störfaktor."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import dtf_basisrate as BR
import dtf_model as M
import dtf_rules as RU
import dtf_solver as S
from dtf_rng import SplitMix64


# --- 1. Aufzählung aller Strategien -------------------------------------------------------------------------------------------------------

def _trees(pr, used):
    """Alle Strategien ab dem Zustand mit den benutzten Tests `used` als Bäume: ("stop",) oder (Test, {Ausgang: Baum})."""
    out = [("stop",)]
    for t in range(pr.m):
        if t in used:
            continue
        n_out = pr.liks[t].shape[0]
        subs = [_trees(pr, used | {t}) for _ in range(n_out)]
        for combo in itertools.product(*subs):
            out.append((t, dict(enumerate(combo))))
    return out


def _tree_value(pr, tree, joint, cost_so_far=0.0):
    """Erwartete Gesamtkosten des Baums über gemeinsame Wahrscheinlichkeiten joint[d] = P(Diagnose d und bisheriger Pfad): Σ_Pfad P(Pfad) · Testkosten + Verlust an den Blättern
    (L · (P(Blatt) − max_d joint[d])); keine Normierung, kein Posterior."""
    if tree[0] == "stop":
        return pr.loss * (joint.sum() - joint.max()) + cost_so_far * joint.sum()
    t, kids = tree
    total = 0.0
    for o, sub in kids.items():
        total += _tree_value(pr, sub, joint * pr.liks[t][o], cost_so_far + float(pr.costs[t]))
    return total


def _small_problems():
    for seed in range(12):
        inst = M.make_instance(seed, 3, 3, 1.0, 0.85, 0.8, 12.0)
        yield f"binär {seed}", M.Problem.from_instance(inst)
    for seed in range(6):
        inst = M.make_instance(seed, 3, 2, 1.0, 0.9, 0.9, 12.0, u=0.2)
        yield f"unklar {seed}", M.Problem.from_instance(inst)
    for seed in range(6):
        inst = M.make_instance(seed, 3, 2, 1.0, 0.9, 0.9, 12.0, genome=(1.5, 0.5))
        yield f"Genom {seed}", M.Problem.from_instance(inst)


@pytest.mark.parametrize("label,pr", list(_small_problems()), ids=lambda x: x if isinstance(x, str) else "")
def test_dp_equals_the_minimum_over_all_explicit_strategies(label, pr):
    trees = _trees(pr, frozenset())
    values = [_tree_value(pr, tr, pr.prior.copy()) for tr in trees]
    assert min(values) == pytest.approx(S.optimal_value(pr), rel=1e-10, abs=1e-12), label
    assert len(trees) > 1


def test_number_of_enumerated_strategies_is_as_counted_by_hand():
    """Zwei binäre Tests: Stopp + 2 · (Stopp + Test)² · … = 9 Strategien (T(1) = 2, T(2) = 1 + 2 · 2² = 9); drei Tests: 1 + 3 · 9² = 244."""
    p2 = M.Problem.from_instance(M.make_instance(0, 3, 2, 1.0, 0.9, 0.9, 10.0))
    p3 = M.Problem.from_instance(M.make_instance(0, 3, 3, 1.0, 0.9, 0.9, 10.0))
    assert len(_trees(p2, frozenset())) == 9 and len(_trees(p3, frozenset())) == 244


def test_the_enumeration_finds_a_strictly_worse_strategy_than_the_optimum():
    """Die Aufzählung ist nicht leer geprüft: es gibt Strategien, die klar schlechter sind als das Optimum."""
    pr = M.Problem.from_instance(M.make_instance(2, 3, 3, 1.0, 0.85, 0.8, 12.0))
    values = [_tree_value(pr, tr, pr.prior.copy()) for tr in _trees(pr, frozenset())]
    assert max(values) > min(values) + 1.0


# --- 2. Simulation ------------------------------------------------------------------------------------------------------------------------

def _simulate(pr, chooser, runs, seed):
    """Mittel und Standardfehler der realisierten Gesamtkosten (Testkosten + L · [falsche Diagnose genannt]) über `runs` Läufe."""
    rng = SplitMix64(seed)
    costs = []
    for _ in range(runs):
        d = rng.choice(pr.prior)
        state, total = pr.start(), 0.0
        while True:
            t = chooser(state)
            w = pr.posterior(state)
            if t is None:
                total += pr.loss * (int(w.argmax()) != d)
                break
            total += float(pr.costs[t])
            o = rng.choice(pr.liks[t][:, d])
            state = pr.child(state, t, o)
        costs.append(total)
    mean = sum(costs) / runs
    se = math.sqrt(sum((c - mean) ** 2 for c in costs) / (runs - 1)) / math.sqrt(runs)
    return mean, se


@pytest.mark.parametrize("rule", ["opt", "M", "G"])
def test_simulation_agrees_with_the_exact_evaluation(rule):
    pr = M.Problem.from_instance(M.make_instance(7, 4, 4, 1.0, 0.9, 0.9, 20.0))
    if rule == "opt":
        chooser, exact = S.optimal_chooser(pr), S.optimal_value(pr)
    else:
        chooser, exact = RU.chooser_for(pr, rule, 0.9), RU.rule_evaluation(pr, rule, 0.9).total
    mean, se = _simulate(pr, chooser, 30000, seed=11)
    assert abs(mean - exact) < 4 * se, (rule, mean, exact, se)


# --- 3. rauschfreier Grenzfall ------------------------------------------------------------------------------------------------------------

def _identification_cost(prior, rows, costs):
    """Teilmengen-DP: f(S) = min über Tests, die S teilen, von c_t · P(S) + f(S₀) + f(S₁); f(einelementig) = 0. None, wenn S nicht trennbar ist."""
    n = len(prior)
    memo = {}

    def f(S):
        if len(S) <= 1:
            return 0.0
        if S in memo:
            return memo[S]
        p = sum(prior[d] for d in S)
        best = None
        for t, row in enumerate(rows):
            a = tuple(d for d in S if row[d] == 0)
            b = tuple(d for d in S if row[d] == 1)
            if not a or not b:
                continue
            fa, fb = f(a), f(b)
            if fa is None or fb is None:
                continue
            v = costs[t] * p + fa + fb
            if best is None or v < best:
                best = v
        memo[S] = best
        return best

    return f(tuple(range(n)))


def test_noise_free_limit_equals_the_subset_dp():
    checked = 0
    for seed in range(40):
        inst = M.make_instance(seed, 5, 6, 1.0, 1.0, 1.0, 1e6)
        ident = _identification_cost(inst.prior, inst.responds, inst.costs)
        if ident is None:
            continue                                                            # Diagnosen nicht trennbar: kein Vergleichswert
        pr = M.Problem.from_instance(inst)
        assert S.optimal_value(pr) == pytest.approx(ident, rel=1e-9), seed
        checked += 1
    assert checked >= 15


# --- 4. Basisratenrechnung -----------------------------------------------------------------------------------------------------------------

def test_ppv_one_test_by_fractions():
    """Anteil 1/1000, Se 99/100, Falsch-positiv-Rate 1/100: PPV = 99/100000 / (99/100000 + 999/100000) = 99/1098."""
    prev, se, fpr = Fraction(1, 1000), Fraction(99, 100), Fraction(1, 100)
    exact = prev * se / (prev * se + (1 - prev) * fpr)
    assert exact == Fraction(99, 1098)
    assert BR.ppv(1e-3, 0.99, 0.01, 1) == pytest.approx(float(exact), rel=1e-12)


def test_ppv_two_tests_independent_by_fractions():
    prev, se, fpr = Fraction(1, 1000), Fraction(99, 100), Fraction(1, 100)
    exact = prev * se**2 / (prev * se**2 + (1 - prev) * fpr**2)
    assert BR.ppv(1e-3, 0.99, 0.01, 2) == pytest.approx(float(exact), rel=1e-12)


def test_ppv_with_disturbance_by_fractions_and_population():
    """Gemeinsamer Fehler: Bruchrechnung der Mischung, dazu eine Simulation einer Bevölkerung mit festem Zufalls-Seed."""
    a, h = Fraction(2, 100), Fraction(4, 10)
    fpr = Fraction(1, 100)
    e = (fpr - a * h) / (1 - a)
    prev, se = Fraction(5, 100), Fraction(99, 100)
    k = 2
    exact = prev * se**k / (prev * se**k + (1 - prev) * (a * h**k + (1 - a) * e**k))
    assert BR.ppv(0.05, 0.99, 0.01, k, True) == pytest.approx(float(exact), rel=1e-12)
    rng = SplitMix64(5)
    pos_sick = pos_all = 0
    for _ in range(300000):
        sick = rng.random() < 0.05
        if sick:
            both = rng.random() < 0.99 and rng.random() < 0.99
        else:
            disturbed = rng.random() < 0.02
            q = 0.4 if disturbed else float(e)
            both = rng.random() < q and rng.random() < q
        if both:
            pos_all += 1
            pos_sick += sick
    p_hat = pos_sick / pos_all
    se_hat = math.sqrt(p_hat * (1 - p_hat) / pos_all)
    assert abs(p_hat - float(exact)) < 4 * se_hat
    assert np.isfinite(p_hat)
