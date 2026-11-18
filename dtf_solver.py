"""Exakte Strategie: DP über alle Zustände mit optimalem Stoppen, Auswertung beliebiger Strategien und der Baum der besten Folge.

f(s) = min( L · (1 − max_d P(d|s)),  min_t [ c_t + Σ_o P(o|s) · f(s + (t, o)) ] ). Eine Strategie ist eine Funktion Zustand → Test oder None (Stopp);
`evaluate` rechnet ihre erwarteten Testkosten, den erwarteten Irrtumsverlust und die Zahl der Tests exakt (ohne Simulation)."""

from dataclasses import dataclass

from dtf_model import Problem


@dataclass(frozen=True)
class Evaluation:
    total: float        # Testkosten + Irrtumsverlust
    test_cost: float    # erwartete Testkosten
    loss: float         # erwarteter Irrtumsverlust = L · P(Irrtum)
    n_tests: float      # erwartete Zahl durchgeführter Tests
    p_error: float      # Wahrscheinlichkeit, die falsche Diagnose zu nennen


def optimal_value(pr: Problem, state=None):
    """Optimale erwartete Gesamtkosten ab `state` (Standard: Start)."""
    return optimal_first(pr, pr.start() if state is None else state)[0]


def optimal_first(pr: Problem, state):
    """(optimaler Wert, bester erster Test oder None für „Stopp“) im Zustand `state`."""
    key = ("opt", state)
    if key in pr.cache:
        return pr.cache[key]
    w = pr.posterior(state)
    best, best_t = pr.stop_loss(w), None
    for t in pr.free_tests(state):
        v = pr.costs[t] + sum(p * optimal_value(pr, pr.child(state, t, o)) for o, p, _ in pr.outcomes(w, t))
        if v < best - 1e-12:   # bei Gleichstand gilt der frühere Test; Stopp gewinnt Gleichstände nicht gegen einen echten Vorteil
            best, best_t = v, t
    pr.cache[key] = (best, best_t)
    return best, best_t


def forced_first(pr: Problem, t: int, state=None):
    """Wert, wenn zuerst Test `t` genommen und danach optimal weitergemacht wird."""
    state = pr.start() if state is None else state
    w = pr.posterior(state)
    return pr.costs[t] + sum(p * optimal_value(pr, pr.child(state, t, o)) for o, p, _ in pr.outcomes(w, t))


def optimal_chooser(pr: Problem):
    return lambda state: optimal_first(pr, state)[1]


def evaluate(pr: Problem, chooser, state=None) -> Evaluation:
    """Erwartete Kosten der Strategie `chooser` (Zustand → Test oder None), exakt über alle erreichbaren Zustände."""
    state = pr.start() if state is None else state
    tc, loss, nt = _evaluate(pr, chooser, state, {})
    return Evaluation(float(tc + loss), float(tc), float(loss), float(nt), float(loss / pr.loss) if pr.loss else 0.0)


def _evaluate(pr, chooser, state, memo):
    if state in memo:
        return memo[state]
    w = pr.posterior(state)
    t = chooser(state)
    if t is None:
        res = (0.0, pr.stop_loss(w), 0.0)
    else:
        tc, loss, nt = float(pr.costs[t]), 0.0, 1.0
        for o, p, _ in pr.outcomes(w, t):
            a, b, c = _evaluate(pr, chooser, pr.child(state, t, o), memo)
            tc, loss, nt = tc + p * a, loss + p * b, nt + p * c
        res = (tc, loss, nt)
    memo[state] = res
    return res


def policy_tree(pr: Problem, chooser, state=None, prob=1.0, depth=3, min_prob=0.02):
    """Baum einer Strategie als verschachteltes Wörterbuch bis zur Tiefe `depth`; Äste unter `min_prob` (Wahrscheinlichkeit seit dem Start) werden gekürzt.

    Knoten: {"prob", "test" (Index oder None), "label", "diagnosis" (wahrscheinlichste Diagnose), "p_best", "children": [(Ausgang, Wahrscheinlichkeit, Knoten)], "cut" (gekürzte Wahrscheinlichkeit)}."""
    state = pr.start() if state is None else state
    w = pr.posterior(state)
    t = chooser(state)
    node = {"prob": prob, "test": t, "label": "Stopp" if t is None else pr.labels[t], "diagnosis": int(w.argmax()), "p_best": float(w.max() / w.sum()),
            "children": [], "cut": 0.0, "state": state}
    if t is None:
        return node
    for o, p, _ in pr.outcomes(w, t):
        if depth <= 0 or prob * p < min_prob:
            node["cut"] += prob * p
            continue
        node["children"].append((o, p, policy_tree(pr, chooser, pr.child(state, t, o), prob * p, depth - 1, min_prob)))
    return node
