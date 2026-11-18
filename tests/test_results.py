"""Messreihe (data/dtf_results.json): Struktur, innere Stimmigkeit, Neu-Rechnen einzelner Instanzen (Band), Zell-Zuordnung und Urteilslogik."""

import json

import pytest

import dtf_constants as C
import dtf_measure as MS
import dtf_results as R
from dtf_model import Settings

RES = R.load()


def _cell(cid):
    return next(c for c in C.CELLS if c["id"] == cid)


def test_all_cells_are_present_with_the_documented_sample_size():
    assert set(RES["cells"]) == {c["id"] for c in C.CELLS} and len(C.CELLS) == 24
    assert RES["meta"]["n_instances"] == C.N_INSTANCES == 120 and RES["meta"]["n_tune"] == C.N_TUNE == 60
    for c in RES["cells"].values():
        assert c["n_instances"] == 120 and len(c["per_instance"]["opt"]) == 120


def test_rule_cells_have_consistent_statistics():
    for cid, c in RES["cells"].items():
        if "tau" not in c:
            continue
        assert c["tau"]["G"] in C.TAUS and c["tau"]["B"] in C.TAUS
        for rule in ("M", "G", "B"):
            test = c["per_instance"][rule][C.N_TUNE:]
            assert c[rule]["mean"] == pytest.approx(sum(test) / len(test), abs=1e-9), (cid, rule)
            assert c[rule]["share_over"] == pytest.approx(sum(1 for g in test if g > C.GAP_MARK) / len(test)), (cid, rule)
            assert all(g >= -1e-9 for g in c["per_instance"][rule]), (cid, rule)            # eine Regel liegt nie unter dem Optimum
            assert c[rule]["se"] >= 0
        diff = [m - g for m, g in zip(c["per_instance"]["M"][C.N_TUNE:], c["per_instance"]["G"][C.N_TUNE:])]
        assert c["diff_MG"]["mean"] == pytest.approx(sum(diff) / len(diff), abs=1e-9)
        assert c["opt_mean"] == pytest.approx(sum(c["per_instance"]["opt"][C.N_TUNE:]) / 60, abs=1e-9)
        assert c["opt_mean_all"] == pytest.approx(sum(c["per_instance"]["opt"]) / 120, abs=1e-9)


def test_the_threshold_curve_contains_the_chosen_threshold_at_the_measured_mean():
    for cid, c in RES["cells"].items():
        if "tau" not in c:
            continue
        for rule in ("G", "B"):
            curve = c["tau_curve"][rule]
            assert set(curve) == {str(t) for t in C.TAUS}
            assert curve[str(c["tau"][rule])] == pytest.approx(c[rule]["mean"], abs=1e-9), (cid, rule)


def test_a_fixed_threshold_of_095_is_worse_on_average_for_both_threshold_rules():
    cells = [c for c in RES["cells"].values() if "tau" in c and c["params"]["kind"] == "bin"]
    for rule in ("G", "B"):
        diffs = [c["tau_curve"][rule]["0.95"] - c[rule]["mean"] for c in cells]
        assert len(diffs) == 11 and sum(diffs) / len(diffs) > 5.0, (rule, diffs)


def test_genome_cells_have_consistent_statistics():
    for cid, c in RES["cells"].items():
        if "share_first" not in c:
            continue
        first = c["per_instance"]["first_genome"]
        assert c["share_first"] == pytest.approx(sum(1 for f in first if f) / len(first))
        assert 0.0 <= c["share_first"] <= 1.0
        assert c["opt_mean"] <= c["opt_without_mean"] + 1e-9                               # ein zusätzlicher Test kann das Optimum nie verschlechtern
        assert c["forced"]["mean"] >= -1e-9 and c["never"]["mean"] >= -1e-9


def test_sweep_seeds_are_disjoint_between_cells():
    seeds = {MS.instance_seed(c, i) for c in C.CELLS for i in range(C.N_INSTANCES)}
    assert len(seeds) == len(C.CELLS) * C.N_INSTANCES and min(seeds) >= C.SWEEP_SEED_BASE


@pytest.mark.parametrize("cid", ["q2", "u1"])
def test_recomputing_the_first_instances_reproduces_the_stored_values(cid):
    """Neu-Rechnen als Band (relative Toleranz 1e-9), nicht bitgleich: die letzte Binärstelle hängt an der Plattform (DEMO-PLAYBOOK §5)."""
    cell = _cell(cid)
    stored = RES["cells"][cid]
    for i in range(4):
        row = MS.measure_rules_instance(cell, i)
        assert row["opt"] == pytest.approx(stored["per_instance"]["opt"][i], rel=1e-9)
        assert row["M"] == pytest.approx(stored["per_instance"]["M"][i], rel=1e-9, abs=1e-9)
        assert row["G"][stored["tau"]["G"]] == pytest.approx(stored["per_instance"]["G"][i], rel=1e-9, abs=1e-9)
        assert row["B"][stored["tau"]["B"]] == pytest.approx(stored["per_instance"]["B"][i], rel=1e-9, abs=1e-9)


def test_recomputing_genome_instances_reproduces_the_stored_values():
    cell = _cell("y50c2")
    stored = RES["cells"]["y50c2"]
    for i in range(6):
        row = MS.measure_genome_instance(cell, i)
        assert row["opt"] == pytest.approx(stored["per_instance"]["opt"][i], rel=1e-9)
        assert row["first_genome"] == stored["per_instance"]["first_genome"][i]


def test_aggregation_picks_the_threshold_on_the_tuning_half_only():
    """Künstliche Zeilen: τ = 0,9 ist auf den ersten 60 Instanzen am besten, auf den letzten 60 aber schlecht; gemessen wird auf den letzten 60 mit dem gewählten τ."""
    rows = []
    for i in range(120):
        tune = i < 60
        good, bad = (1.0, 9.0) if tune else (9.0, 1.0)
        rows.append({"opt": 1.0, "opt_tests": 1.0, "opt_perr": 0.1, "M": 2.0, "G": {t: (good if t == 0.9 else bad) for t in C.TAUS},
                     "B": {t: (good if t == 0.8 else bad) for t in C.TAUS}})
    agg = MS.aggregate_rules(rows)
    assert agg["tau"] == {"G": 0.9, "B": 0.8}
    assert agg["G"]["mean"] == pytest.approx(9.0) and agg["B"]["mean"] == pytest.approx(9.0) and agg["M"]["mean"] == pytest.approx(2.0)
    assert agg["diff_MG"]["mean"] == pytest.approx(2.0 - 9.0)


def test_gap_function_and_standard_error():
    assert MS.gap(12.0, 10.0) == pytest.approx(20.0) and MS.gap(1.0, 0.0) == 0.0
    assert MS.se([1.0, 3.0]) == pytest.approx(1.0) and MS.mean([1, 2, 3]) == 2


def test_every_preset_lies_exactly_on_a_measured_cell():
    from dtf_stories import STORIES
    for name, preset in C.PRESETS.items():
        cell_id, diff = R.nearest_cell(Settings(**preset))
        assert diff == [] and cell_id == STORIES[name][2], (name, cell_id, diff)


def test_off_grid_settings_name_the_differing_parameters():
    cid, diff = R.nearest_cell(Settings(m=5))
    assert "m" in diff and cid.startswith(("q", "g"))
    cid, diff = R.nearest_cell(Settings(u=0.3))
    assert cid.startswith("u") and "m" in diff                                              # „unklar“ nur mit 6 Tests gemessen, Standard hat 8 (gerechnet 7)
    cid, diff = R.nearest_cell(Settings(genome=True, g_cost=4.0, g_yield=0.25))
    assert cid == "y25c4" and diff == []


def test_an_unclear_setting_never_maps_to_a_binary_cell_and_vice_versa():
    assert R.nearest_cell(Settings(u=0.1))[0].startswith("u")
    assert not R.nearest_cell(Settings())[0].startswith(("u", "y"))
    assert R.nearest_cell(Settings(genome=True))[0].startswith("y")


def test_verdict_state_thresholds_are_inclusive_as_documented():
    assert R.verdict_state(2.999) == "reicht" and R.verdict_state(3.0) == "merklich"
    assert R.verdict_state(10.0) == "merklich" and R.verdict_state(10.001) == "verfehlt"


def test_best_rule_picks_the_smallest_mean_gap():
    fake = {"M": {"mean": 5.0}, "G": {"mean": 3.0}, "B": {"mean": 9.0}}
    assert R.best_rule(fake) == ("G", 3.0)


def test_results_file_is_valid_json_with_the_documented_meta():
    data = json.loads(R.DATA.read_text(encoding="utf-8"))
    assert data["meta"]["taus"] == list(C.TAUS) and data["meta"]["seed_base"] == C.SWEEP_SEED_BASE
