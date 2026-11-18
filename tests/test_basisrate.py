"""Basisratenfalle: Eigenschaften der Rechnung (die unabhängige Prüfung gegen Bruchrechnung und Simulation steht in test_oracle_diagnose_testfolge.py)."""

import pytest

import dtf_basisrate as BR


def test_one_positive_test_is_the_same_with_and_without_shared_error():
    for prev in BR.PREVALENCE_OPTIONS:
        for se in BR.SE_OPTIONS:
            for fpr in BR.FPR_OPTIONS:
                assert BR.ppv(prev, se, fpr, 1, False) == pytest.approx(BR.ppv(prev, se, fpr, 1, True), rel=1e-12)


def test_shared_error_lowers_the_ppv_from_two_tests_on():
    for prev in BR.PREVALENCE_OPTIONS:
        for fpr in BR.FPR_OPTIONS:
            for k in range(2, 7):
                assert BR.ppv(prev, 0.99, fpr, k, True) < BR.ppv(prev, 0.99, fpr, k, False), (prev, fpr, k)


def test_ppv_rises_with_each_further_positive_test():
    for shared in (False, True):
        values = [BR.ppv(1e-3, 0.99, 0.01, k, shared) for k in range(1, 8)]
        assert values == sorted(values) and values[-1] > values[0]


def test_ppv_rises_with_the_share_of_sick_people():
    values = [BR.ppv(p, 0.99, 0.01, 1) for p in BR.PREVALENCE_OPTIONS]
    assert values == sorted(values)


def test_residual_fpr_keeps_the_marginal_rate():
    for fpr in BR.FPR_OPTIONS:
        e = BR.residual_fpr(fpr)
        assert BR.DISTURB_SHARE * BR.DISTURB_PROB + (1 - BR.DISTURB_SHARE) * e == pytest.approx(fpr)


def test_residual_fpr_rejects_a_rate_below_the_disturbance_share():
    with pytest.raises(ValueError):
        BR.residual_fpr(0.005)


def test_tests_until_by_hand_threshold():
    """1:1.000, Se 0,99, Falsch-positiv-Rate 0,01: unabhängig reichen 2 positive Tests für 0,9 (0,9075), mit gemeinsamem Fehler sind es 6 (0,9200; fünf geben 0,8230)."""
    assert BR.tests_until(1e-3, 0.99, 0.01, 0.9, False) == 2
    assert BR.tests_until(1e-3, 0.99, 0.01, 0.9, True) == 6
    assert BR.ppv(1e-3, 0.99, 0.01, 5, True) < 0.9 <= BR.ppv(1e-3, 0.99, 0.01, 6, True)


def test_tests_until_returns_none_when_never_reached():
    assert BR.tests_until(1e-4, 0.9, 0.05, 0.999999, True, k_max=3) is None


def test_every_fpr_option_admits_the_disturbance_model():
    assert all(f >= BR.DISTURB_SHARE * BR.DISTURB_PROB for f in BR.FPR_OPTIONS)


def test_referral_filter_raises_the_ppv():
    values = [BR.ppv(p, 0.99, 0.01, 1) for _, p in BR.REFERRALS]
    assert values == sorted(values) and values[0] < 0.02 and values[-1] > 0.9
