"""Faustregeln: Handrechnungen, Eigenschaft „nie unter dem Optimum“ und Zweig-Tests je Variante (Nullspalten-Falle, DEMO-PLAYBOOK §0)."""

import pytest

import dtf_model as M
import dtf_rules as RU
import dtf_solver as S


def test_all_three_rules_find_the_hand_optimum_with_one_test(one_test):
    for rule in RU.RULES:
        assert RU.rule_evaluation(one_test, rule, 0.95).total == pytest.approx(2.0), rule


def test_threshold_decides_whether_the_second_test_is_made(two_equal_tests):
    """Nach dem ersten Test liegt die beste Diagnose bei 0,9. τ = 0,95: weitertesten (12); τ = 0,8: stoppen (11, das Optimum). Kurzsichtig: stoppen (11)."""
    pr = two_equal_tests
    assert RU.rule_evaluation(pr, "G", 0.95).total == pytest.approx(12.0)
    assert RU.rule_evaluation(pr, "B", 0.95).total == pytest.approx(12.0)
    assert RU.rule_evaluation(pr, "G", 0.8).total == pytest.approx(11.0)
    assert RU.rule_evaluation(pr, "B", 0.8).total == pytest.approx(11.0)
    assert RU.rule_evaluation(pr, "M").total == pytest.approx(11.0)


def test_myopic_stops_when_no_single_test_pays_but_two_together_would():
    """Zwei Tests mit Se = Sp = 0,6 (Kosten je 1), Irrtumspreis 20, Vorab 1/2: ein Test bringt 20·0,5 → 20·0,4 = Gewinn 2 bei Kosten 1 (lohnt); der Wert der Handrechnung wird nur geprüft,
    wenn die Regel überhaupt testet: Gesamtkosten ≤ Stopp sofort (10)."""
    liks = M.binary_liks(((1, 0), (1, 0)), 0.6, 0.6)
    pr = M.Problem([0.5, 0.5], liks, [1.0, 1.0], 20.0)
    assert RU.rule_evaluation(pr, "M").total <= 10.0 + 1e-12


def test_rules_never_beat_the_optimum():
    for seed in range(25):
        pr = M.Problem.from_instance(M.make_instance(seed, 5, 5, 1.0, 0.9, 0.9, 20.0))
        opt = S.optimal_value(pr)
        for rule in RU.RULES:
            for tau in (0.7, 0.9):
                assert RU.rule_evaluation(pr, rule, tau).total >= opt - 1e-9, (seed, rule, tau)


def test_every_rule_differs_from_every_other_on_at_least_one_instance():
    """Zweig-Test: keine Regel ist stillschweigend eine andere (Nullspalte = Variante läuft nie)."""
    seen = {("M", "G"): False, ("M", "B"): False, ("G", "B"): False}
    for seed in range(30):
        pr = M.Problem.from_instance(M.make_instance(seed, 6, 6, 1.0, 0.9, 0.9, 20.0))
        v = {r: RU.rule_evaluation(pr, r, 0.8).total for r in RU.RULES}
        for a, b in seen:
            seen[(a, b)] = seen[(a, b)] or abs(v[a] - v[b]) > 1e-6
    assert all(seen.values()), seen


def test_the_threshold_changes_the_info_and_cheapest_rules():
    changed = {"G": False, "B": False}
    for seed in range(20):
        pr = M.Problem.from_instance(M.make_instance(seed, 6, 6, 1.0, 0.9, 0.9, 20.0))
        for rule in changed:
            changed[rule] = changed[rule] or abs(RU.rule_evaluation(pr, rule, 0.6).total - RU.rule_evaluation(pr, rule, 0.99).total) > 1e-6
    assert all(changed.values())


def test_unclear_outcome_changes_the_optimum_and_the_rules():
    base = unclear = 0.0
    for seed in range(10):
        a = M.Problem.from_instance(M.make_instance(seed, 5, 5, 1.0, 0.9, 0.9, 20.0))
        b = M.Problem.from_instance(M.make_instance(seed, 5, 5, 1.0, 0.9, 0.9, 20.0, u=0.3))
        base += S.optimal_value(a)
        unclear += S.optimal_value(b)
    assert unclear > base + 0.5                                              # unklare Ergebnisse kosten im Mittel etwas


def test_the_genome_test_changes_the_optimum_only_when_it_is_cheap_enough():
    lower = same = 0
    for seed in range(15):
        inst = M.make_instance(seed, 5, 4, 1.0, 0.9, 0.9, 20.0)
        without = S.optimal_value(M.Problem.from_instance(inst))
        cheap = S.optimal_value(M.Problem.from_instance(M.make_instance(seed, 5, 4, 1.0, 0.9, 0.9, 20.0, genome=(0.5, 0.5))))
        dear = S.optimal_value(M.Problem.from_instance(M.make_instance(seed, 5, 4, 1.0, 0.9, 0.9, 20.0, genome=(30.0, 0.5))))
        lower += cheap < without - 1e-9
        same += abs(dear - without) < 1e-9
        assert cheap <= without + 1e-9 and dear <= without + 1e-9              # ein zusätzlicher Test kann nie schaden (Optimum nie höher)
    assert lower >= 10 and same == 15                                          # billig: fast immer hilfreich; sehr teuer: nie gewählt


def test_info_gain_is_zero_for_a_test_that_does_not_split():
    pr = M.Problem([0.5, 0.5], M.binary_liks(((1, 1),), 0.9, 0.9), [1.0], 10.0)    # spricht beide Diagnosen gleich an
    w = pr.posterior(pr.start())
    assert RU.info_gain(pr, w, 0, pr.entropy(w)) == pytest.approx(0.0, abs=1e-12)


def test_cheapest_rule_skips_tests_without_information():
    liks = M.binary_liks(((1, 1), (1, 0)), 0.9, 0.9)                           # Test 1 trennt nicht, Test 2 trennt
    pr = M.Problem([0.5, 0.5], liks, [0.1, 1.0], 10.0)                           # Test 1 ist billiger, aber nutzlos
    assert RU.cheapest_chooser(pr, 0.99)(pr.start()) == 1


def test_info_rule_takes_the_best_gain_per_cost():
    liks = M.binary_liks(((1, 0), (1, 0)), 0.9, 0.9)                           # gleiche Information, Test 2 doppelt so teuer
    pr = M.Problem([0.5, 0.5], liks, [1.0, 2.0], 10.0)
    assert RU.info_chooser(pr, 0.99)(pr.start()) == 0


def test_chooser_for_rejects_unknown_rules():
    pr = M.Problem([0.5, 0.5], M.binary_liks(((1, 0),), 0.9, 0.9), [1.0], 10.0)
    with pytest.raises(ValueError):
        RU.chooser_for(pr, "X")
