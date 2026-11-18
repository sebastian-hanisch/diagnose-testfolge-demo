"""Exakte Strategie an Handrechnungen (DEMO-PLAYBOOK §1b): Optimum, Zerlegung der Kosten, erster Test, Randfälle."""

import math

import numpy as np
import pytest

import dtf_model as M
import dtf_solver as S
from conftest import make_problem


def test_one_test_hand_value(one_test):
    """Ohne Test 5, mit Test 1 + 10 · 0,1 = 2."""
    assert S.optimal_value(one_test) == pytest.approx(2.0, abs=1e-12)
    assert S.optimal_first(one_test, one_test.start())[1] == 0


def test_too_expensive_test_is_not_taken():
    pr = make_problem([0.5, 0.5], [[[0.1, 0.9], [0.9, 0.1]]], [4.0], 10.0)      # 4 + 1 = 5 = Stopp sofort (5)
    value, first = S.optimal_first(pr, pr.start())
    assert value == pytest.approx(5.0, abs=1e-12) and first is None


def test_two_equal_tests_stop_after_the_first(two_equal_tests):
    assert S.optimal_value(two_equal_tests) == pytest.approx(11.0, abs=1e-12)
    ev = S.evaluate(two_equal_tests, S.optimal_chooser(two_equal_tests))
    assert ev.n_tests == pytest.approx(1.0) and ev.test_cost == pytest.approx(1.0) and ev.loss == pytest.approx(10.0)


def test_forcing_the_second_test_costs_twelve(two_equal_tests):
    """Nach einem positiven ersten Test (Wahrscheinlichkeit 0,9 / 0,1): Stopp 10; ein zweiter Test 1 + 0,82 · 100/82 + 0,18 · 50 = 11 (also insgesamt 12)."""
    pr = two_equal_tests
    first_state = pr.child(pr.start(), 0, 1)                                   # erster Test positiv
    assert S.optimal_value(pr, first_state) == pytest.approx(10.0)             # Stopp ist besser als weitertesten
    assert S.forced_first(pr, 1, first_state) == pytest.approx(11.0)
    assert S.forced_first(pr, 0) == pytest.approx(11.0)                        # erster Test, danach optimal: 1 + 10
def test_genome_test_with_certain_yield():
    """Genomtest mit y = 1, Kosten 1: findet die Diagnose sicher, Gesamtkosten 1 (Stopp sofort 5)."""
    pr = make_problem([0.5, 0.5], [M.genome_lik(2, 1.0)], [1.0], 10.0)
    assert S.optimal_value(pr) == pytest.approx(1.0, abs=1e-12)


def test_genome_test_with_half_yield():
    """y = 0,5: mit Wahrscheinlichkeit 1/2 gefunden (Rest 0), sonst Zustand unverändert (Verlust 5): 1 + 0,5 · 5 = 3,5."""
    pr = make_problem([0.5, 0.5], [M.genome_lik(2, 0.5)], [1.0], 10.0)
    assert S.optimal_value(pr) == pytest.approx(3.5, abs=1e-12)


def test_pure_unclear_test_is_worthless():
    liks = M.unclear_liks(((1, 0),), 0.9, 0.9, 1.0)                             # u = 1: der Test sagt immer „unklar“
    pr = M.Problem([0.5, 0.5], liks, [0.1], 10.0)
    value, first = S.optimal_first(pr, pr.start())
    assert value == pytest.approx(5.0) and first is None


def test_evaluation_decomposes_and_matches_the_dp(one_test):
    ev = S.evaluate(one_test, S.optimal_chooser(one_test))
    assert ev.total == pytest.approx(S.optimal_value(one_test))
    assert ev.total == pytest.approx(ev.test_cost + ev.loss)
    assert ev.p_error == pytest.approx(0.1) and ev.loss == pytest.approx(1.0) and ev.test_cost == pytest.approx(1.0) and ev.n_tests == pytest.approx(1.0)


def test_stopping_immediately_evaluates_to_the_stop_loss(one_test):
    ev = S.evaluate(one_test, lambda state: None)
    assert (ev.total, ev.test_cost, ev.n_tests, ev.p_error) == (pytest.approx(5.0), 0.0, 0.0, pytest.approx(0.5))


def test_noise_free_values_are_finite_and_impossible_branches_are_skipped():
    """Se = Sp = 1: ein Ausgang kann Wahrscheinlichkeit 0 haben; kein 0/0, kein NaN."""
    pr = M.Problem([0.5, 0.25, 0.25], M.binary_liks(((1, 0, 0), (0, 1, 0)), 1.0, 1.0), [1.0, 1.0], 20.0)
    value = S.optimal_value(pr)
    assert math.isfinite(value)
    w = pr.posterior(pr.child(pr.start(), 0, 1))                                  # Test 1 positiv: nur Diagnose 1 möglich
    assert [o for o, p, _ in pr.outcomes(w, 1)] == [0]                            # Test 2 kann dann nur negativ ausfallen (Sp = 1)


def test_without_test_removes_exactly_one_test(two_equal_tests):
    smaller = two_equal_tests.without_test(0)
    assert smaller.m == 1 and list(smaller.costs) == [1.0] and smaller.labels == ["T2"]
    assert S.optimal_value(smaller) == pytest.approx(S.optimal_value(two_equal_tests))   # gleiche Tests: Wert unverändert


def test_policy_tree_of_the_hand_instance(one_test):
    tree = S.policy_tree(one_test, S.optimal_chooser(one_test), depth=3, min_prob=0.0)
    assert tree["test"] == 0 and tree["label"] == "T1"
    assert [(o, round(p, 12)) for o, p, _ in tree["children"]] == [(0, 0.5), (1, 0.5)]
    leaf_neg, leaf_pos = (c for _, _, c in tree["children"])
    assert leaf_neg["test"] is None and leaf_neg["diagnosis"] == 1 and leaf_neg["p_best"] == pytest.approx(0.9)
    assert leaf_pos["test"] is None and leaf_pos["diagnosis"] == 0 and leaf_pos["p_best"] == pytest.approx(0.9)


def test_policy_tree_cuts_rare_branches_and_reports_the_cut_mass(one_test):
    tree = S.policy_tree(one_test, S.optimal_chooser(one_test), depth=3, min_prob=0.6)
    assert tree["children"] == [] and tree["cut"] == pytest.approx(1.0)


def test_single_diagnosis_needs_no_test():
    pr = make_problem([1.0], [[[0.4], [0.6]]], [1.0], 10.0)
    assert S.optimal_value(pr) == 0.0


def test_indistinguishable_diagnoses_cannot_be_separated():
    """Zwei Diagnosen mit gleicher Likelihood: kein Test hilft, Optimum = Stopp mit Verlust L / 2."""
    pr = make_problem([0.5, 0.5], [[[0.3, 0.3], [0.7, 0.7]]], [0.1], 10.0)
    assert S.optimal_value(pr) == pytest.approx(5.0)


def test_zero_loss_means_stop_at_once_and_no_division_by_zero():
    pr = make_problem([0.5, 0.5], [[[0.1, 0.9], [0.9, 0.1]]], [1.0], 0.0)
    ev = S.evaluate(pr, S.optimal_chooser(pr))
    assert S.optimal_value(pr) == 0.0 and ev.p_error == 0.0


def test_optimal_value_is_never_above_stopping_or_any_single_test():
    """Optimum ≤ Stopp sofort und ≤ jeder erzwungene erste Test (Eigenschaft über 20 Instanzen)."""
    for seed in range(20):
        inst = M.make_instance(seed, 4, 5, 1.0, 0.9, 0.9, 20.0)
        pr = M.Problem.from_instance(inst)
        v = S.optimal_value(pr)
        assert v <= pr.stop_loss(pr.posterior(pr.start())) + 1e-12
        assert all(v <= S.forced_first(pr, t) + 1e-9 for t in range(pr.m))
        assert any(abs(v - S.forced_first(pr, t)) < 1e-9 for t in range(pr.m)) or S.optimal_first(pr, pr.start())[1] is None


def test_the_dp_value_does_not_depend_on_the_order_of_the_tests():
    """Vertauschte Tests: gleiche Folge von Kosten und Likelihoods in anderer Reihenfolge gibt denselben Optimalwert."""
    inst = M.make_instance(3, 4, 4, 1.0, 0.9, 0.9, 20.0)
    pr = M.Problem.from_instance(inst)
    perm = [2, 0, 3, 1]
    pr2 = M.Problem(pr.prior, [pr.liks[t] for t in perm], pr.costs[perm], pr.loss)
    assert S.optimal_value(pr) == pytest.approx(S.optimal_value(pr2), rel=1e-12)


def test_a_cached_result_is_the_same_object_value():
    pr = M.Problem.from_instance(M.make_instance(1, 4, 4, 1.0, 0.9, 0.9, 20.0))
    assert S.optimal_value(pr) == S.optimal_value(pr)
    assert isinstance(S.optimal_value(pr), (float, np.floating))
