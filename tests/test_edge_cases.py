"""Randfälle der Kernroutinen: Entartungen der Instanz (eine Diagnose, ununterscheidbare Diagnosen, gleiche Kosten, Irrtumspreis 0, fehlerfreie Tests)."""

import math

import pytest

import dtf_model as M
import dtf_rules as RU
import dtf_solver as S
from conftest import make_problem


def test_exactly_two_diagnoses_and_two_tests_is_the_smallest_real_case():
    inst = M.make_instance(0, 2, 2, 1.0, 0.9, 0.9, 20.0)
    pr = M.Problem.from_instance(inst)
    assert math.isfinite(S.optimal_value(pr)) and S.optimal_value(pr) <= pr.stop_loss(pr.posterior(pr.start())) + 1e-12


def test_all_costs_equal_all_rules_are_defined():
    pr = M.Problem([0.4, 0.3, 0.3], M.binary_liks(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 0.9, 0.9), [1.0, 1.0, 1.0], 20.0)
    for rule in RU.RULES:
        assert math.isfinite(RU.rule_evaluation(pr, rule, 0.9).total)


def test_free_tests_with_zero_cost_are_all_used_when_they_help():
    pr = M.Problem([0.5, 0.5], M.binary_liks(((1, 0),), 0.9, 0.9), [0.0], 10.0)
    assert S.optimal_value(pr) == pytest.approx(1.0)                                    # 0 + 10 · 0,1


def test_single_diagnosis_all_rules_stop_at_once():
    pr = make_problem([1.0], [[[0.4], [0.6]]], [1.0], 10.0)
    for rule in RU.RULES:
        assert RU.rule_evaluation(pr, rule).total == 0.0
    ev = S.evaluate(pr, S.optimal_chooser(pr))
    assert ev.n_tests == 0.0 and ev.p_error == 0.0


def test_zero_loss_the_myopic_rule_stops_and_the_threshold_rules_still_test():
    """Irrtumspreis 0: Stopp kostet nichts (Optimum 0, kurzsichtig stoppt); Info/Kosten und billigster Test testen weiter, solange die beste Diagnose unter τ = 0,95 liegt
    (hier ein Test für 1): Gesamtkosten 1. Die Stoppschwelle kennt den Irrtumspreis nicht."""
    pr = make_problem([0.5, 0.5], [[[0.1, 0.9], [0.9, 0.1]]], [1.0], 0.0)
    assert S.optimal_value(pr) == 0.0
    assert RU.rule_evaluation(pr, "M").total == 0.0
    assert RU.rule_evaluation(pr, "G", 0.95).total == pytest.approx(1.0) and RU.rule_evaluation(pr, "B", 0.95).total == pytest.approx(1.0)


def test_noise_free_perfect_tests_identify_exactly_when_the_diagnoses_are_separable():
    for seed in range(10):
        inst = M.make_instance(seed, 4, 6, 1.0, 1.0, 1.0, 1e6)
        distinct = len({tuple(inst.responds[t][d] for t in range(6)) for d in range(4)}) == 4
        pr = M.Problem.from_instance(inst)
        ev = S.evaluate(pr, S.optimal_chooser(pr))
        assert (ev.p_error < 1e-9) == distinct, seed


def test_very_large_loss_reaches_the_bayes_error_of_using_all_tests():
    """Irrtumspreis 10^6: Testkosten sind dagegen vernachlässigbar, der Irrtum des Optimums ist der Bayes-Fehler bei Einsatz ALLER Tests
    (Aufzählung aller 2^5 Ergebnisvektoren: Fehler = Σ_o (P(o) − max_d P(d, o)))."""
    import itertools
    inst = M.make_instance(2, 3, 5, 1.0, 0.9, 0.9, 1e6)
    pr = M.Problem.from_instance(inst)
    bayes = 0.0
    for outcome in itertools.product((0, 1), repeat=5):
        joint = pr.prior.copy()
        for t, o in enumerate(outcome):
            joint = joint * pr.liks[t][o]
        bayes += joint.sum() - joint.max()
    ev = S.evaluate(pr, S.optimal_chooser(pr))
    assert ev.p_error == pytest.approx(bayes, abs=1e-4) and ev.n_tests >= 3.0


def test_very_cheap_irrtum_means_no_tests():
    pr = M.Problem.from_instance(M.make_instance(2, 4, 5, 1.0, 0.9, 0.9, 0.5))
    ev = S.evaluate(pr, S.optimal_chooser(pr))
    assert ev.n_tests == 0.0 and ev.total == pytest.approx(0.5 * (1 - max(pr.prior)))


def test_genome_with_zero_yield_never_helps_and_is_never_chosen():
    inst = M.make_instance(4, 4, 4, 1.0, 0.9, 0.9, 20.0, genome=(0.1, 0.0))
    pr = M.Problem.from_instance(inst)
    without = M.Problem.from_instance(inst, with_genome=False)
    assert S.optimal_value(pr) == pytest.approx(S.optimal_value(without), rel=1e-12)
    assert S.optimal_first(pr, pr.start())[1] != pr.m - 1
