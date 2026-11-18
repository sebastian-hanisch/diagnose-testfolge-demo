"""Instanz-Erzeugung, Zufallsquelle und Likelihood-Tabellen."""

import math
from pathlib import Path

import numpy as np
import pytest

import dtf_constants as C
import dtf_model as M
import dtf_rng as RNG

ROOT = Path(__file__).resolve().parent.parent


def test_splitmix64_first_value_is_the_published_one():
    """SplitMix64 mit Seed 0 beginnt mit 16294208416658607535 (Referenzwert des Verfahrens)."""
    assert RNG.SplitMix64(0).next_u64() == 16294208416658607535


def test_rng_copy_matches_the_portfolio_template_if_present():
    template = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not template.exists():
        pytest.skip("Vorlage nicht vorhanden")
    assert (ROOT / "dtf_rng.py").read_text(encoding="utf-8").replace("\r\n", "\n") == template.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_exact_exp_agrees_with_math_exp_to_rounding():
    for x in (-3.0, -0.5, 0.0, 0.7, 2.5):
        assert M.exact_exp(x) == pytest.approx(math.exp(x), rel=1e-15)


def test_zipf_weights_by_hand():
    assert M.zipf_weights(3, 0.0) == [1.0, 1.0, 1.0]
    assert M.zipf_weights(3, 1.0) == [1.0, 0.5, 1.0 / 3.0]
    w = M.zipf_weights(4, 1.5)
    assert w == pytest.approx([r ** -1.5 for r in range(1, 5)], rel=1e-14)
    with pytest.raises(ValueError):
        M.zipf_weights(3, 0.7)


def test_instance_is_reproducible_and_seed_dependent():
    a = M.make_instance(5, 6, 8, 1.0, 0.9, 0.9, 20)
    assert a == M.make_instance(5, 6, 8, 1.0, 0.9, 0.9, 20)
    assert a != M.make_instance(6, 6, 8, 1.0, 0.9, 0.9, 20)


def test_instance_basic_properties():
    for seed in range(20):
        inst = M.make_instance(seed, 6, 8, 1.0, 0.9, 0.9, 20)
        assert sum(inst.prior) == pytest.approx(1.0) and all(p > 0 for p in inst.prior)
        assert len(inst.responds) == 8 and all(len(r) == 6 and set(r) <= {0, 1} for r in inst.responds)
        assert sum(inst.costs) / 8 == pytest.approx(1.0) and all(c > 0 for c in inst.costs)


def test_the_prior_is_shuffled_over_the_diagnoses():
    firsts = {M.make_instance(s, 6, 4, 1.0, 0.9, 0.9, 20).prior.index(max(M.make_instance(s, 6, 4, 1.0, 0.9, 0.9, 20).prior)) for s in range(30)}
    assert len(firsts) >= 4                                                     # der Spitzenreiter steht nicht immer auf Platz 1


def test_prior_follows_the_zipf_ratio():
    inst = M.make_instance(1, 4, 3, 1.0, 0.9, 0.9, 20)
    srt = sorted(inst.prior, reverse=True)
    assert srt[0] / srt[1] == pytest.approx(2.0) and srt[0] / srt[3] == pytest.approx(4.0)


def test_likelihood_tables_are_probability_distributions():
    inst = M.make_instance(3, 5, 4, 1.0, 0.8, 0.9, 20, u=0.2, genome=(2.0, 0.5))
    for tab in M.binary_liks(inst.responds, inst.se, inst.sp) + M.unclear_liks(inst.responds, inst.se, inst.sp, inst.u) + [M.genome_lik(5, 0.5)]:
        assert np.allclose(tab.sum(axis=0), 1.0)                                # je Diagnose Summe über die Ausgänge = 1
        assert (tab >= 0).all()


def test_binary_likelihood_by_hand():
    tab = M.binary_liks(((1, 0),), 0.8, 0.9)[0]
    assert tab[1].tolist() == pytest.approx([0.8, 0.1]) and tab[0].tolist() == pytest.approx([0.2, 0.9])


def test_unclear_likelihood_by_hand():
    tab = M.unclear_liks(((1, 0),), 0.8, 0.9, 0.25)[0]
    assert tab[2].tolist() == pytest.approx([0.75 * 0.8, 0.75 * 0.1]) and tab[1].tolist() == pytest.approx([0.25, 0.25])


def test_effective_m_caps():
    assert M.effective_m(8, 0.0, False) == 8
    assert M.effective_m(8, 0.1, False) == C.M_CAP_UNCLEAR and M.effective_m(8, 0.0, True) == C.M_CAP_GENOME
    assert M.effective_m(5, 0.3, True) == 5 and M.effective_m(8, 0.3, True) == C.M_CAP_GENOME


def test_instance_from_settings_uses_the_quality_pair_and_genome():
    s = M.Settings(m=8, n=4, quality="0,8 / 0,9", genome=True, g_cost=4.0, g_yield=0.25, seed=3)
    inst = M.instance_from_settings(s)
    assert (inst.se, inst.sp) == (0.8, 0.9) and inst.genome == (4.0, 0.25) and len(inst.responds) == C.M_CAP_GENOME and len(inst.prior) == 4


def test_problem_from_instance_appends_the_genome_test_last():
    inst = M.make_instance(3, 5, 4, 1.0, 0.9, 0.9, 20, genome=(2.0, 0.5))
    pr = M.Problem.from_instance(inst)
    assert pr.m == 5 and pr.labels[-1] == "Genom" and pr.costs[-1] == 2.0 and pr.liks[-1].shape == (6, 5)
    assert M.Problem.from_instance(inst, with_genome=False).m == 4


def test_display_seeds_never_coincide_with_sweep_seeds():
    assert C.SEED_MAX < C.SWEEP_SEED_BASE


def test_settings_are_hashable_for_the_cache():
    assert hash(M.Settings()) == hash(M.Settings()) and M.Settings() != M.Settings(seed=1)
