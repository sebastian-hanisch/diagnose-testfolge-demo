"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) in dtf_evaluation.py und bekommt Werte, nicht Seeds."""

import dtf_evaluation as ev


def test_rules_verdict_reicht_is_a_success_with_rule_gap_and_sample_size():
    kind, text = ev.verdict_rules("reicht", "G", 1.84, 0.4, 60)
    assert kind == "success" and "Hier reicht eine einfache Regel" in text
    assert "Info/Kosten: im Mittel 1,8 % über dem Optimum (± 0,4, 60 Instanzen" in text


def test_rules_verdict_merklich_is_info():
    kind, text = ev.verdict_rules("merklich", "M", 6.66, 0.6, 60)
    assert kind == "info" and "kostet merklich" in text and "kurzsichtig (ein Test voraus): im Mittel 6,7 %" in text


def test_rules_verdict_verfehlt_is_a_warning():
    kind, text = ev.verdict_rules("verfehlt", "B", 21.0, 1.0, 60)
    assert kind == "warning" and "verfehlt das Optimum deutlich" in text and "billigster nützlicher Test: im Mittel 21,0 %" in text


def test_rules_verdict_names_an_inexact_cell_only_when_the_settings_differ():
    assert "weicht von Ihren Einstellungen ab" not in ev.verdict_rules("merklich", "M", 6.7, 0.6, 60)[1]
    for state in ("reicht", "merklich", "verfehlt"):
        assert "weicht von Ihren Einstellungen ab" in ev.verdict_rules(state, "G", 6.7, 0.6, 60, exact=False)[1]


def test_genome_verdict_thresholds_at_ninety_and_ten_percent():
    assert ev.verdict_genome(0.9, 0.0, 46.0, 0.5, 1.0, True)[0] == "success"
    assert ev.verdict_genome(0.89, 0.0, 46.0, 0.5, 1.0, True)[0] == "warning"
    assert ev.verdict_genome(0.11, 3.0, 18.0, 0.5, 2.0, True)[0] == "warning"
    assert ev.verdict_genome(0.1, 25.0, 1.7, 0.5, 4.0, True)[0] == "info"


def test_genome_verdict_names_the_numbers_and_the_inexact_cell():
    kind, text = ev.verdict_genome(0.46, 2.4, 18.3, 0.5, 2.0, False)
    assert "erster Test in 46 % der Instanzen" in text and "„Genom immer zuerst“ kostet 2,4 %" in text and "„Genom nie“ 18,3 % mehr" in text
    assert "Ausbeute 0,50, Kosten 2,0" in text and "nächstliegende gemessene Zelle" in text
    assert "nächstliegende gemessene Zelle" not in ev.verdict_genome(0.46, 2.4, 18.3, 0.5, 2.0, True)[1]


def test_ppv_verdict_after_one_test_is_neutral():
    kind, text = ev.verdict_ppv(1e-3, 1, 0.09, 0.09)
    assert kind == "info" and "Nach einem positiven Test sind beide Rechnungen gleich" in text


def test_ppv_verdict_threshold_at_five_points():
    assert ev.verdict_ppv(1e-3, 2, 0.90, 0.851)[0] == "success"            # Unterschied 0,049
    kind, text = ev.verdict_ppv(1e-3, 2, 0.9075, 0.2344)                    # Unterschied 0,67
    assert kind == "warning" and "0,91" in text and "0,23" in text and "(0,67)" in text


def test_number_formatting_uses_decimal_commas():
    assert ev.fmt1(3.14159) == "3,1" and ev.fmt2(0.5) == "0,50"
