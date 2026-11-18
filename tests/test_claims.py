"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4, §11a): die README wird von tools/make_readme.py aus data/dtf_results.json erzeugt, der Test prüft
Gleichheit mit dem Erzeugnis (Tabellen und Prosa) und stützt die Urteilssätze mit Zählungen über ALLE Zellen, nicht über eine einzelne Zelle."""

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import dtf_constants as C  # noqa: E402
import dtf_results as R  # noqa: E402
import make_readme as MR  # noqa: E402

RES = R.load()
README = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")


@lru_cache(maxsize=1)
def n_tests() -> int:
    """Testzahl der Suite per `pytest --collect-only`. NICHT beim Import berechnen: das Sammeln importiert diese Datei wieder und würde sich selbst endlos aufrufen."""
    return MR.count_tests()


def test_readme_equals_the_generated_text():
    """Jede Tabelle und jede Zahl der Prosa stammt aus der Messreihe: die README ist genau das Erzeugnis von tools/make_readme.py (inklusive Testzahl)."""
    assert README == MR.render(RES, n_tests()),"README veraltet: python tools/make_readme.py ausführen (Text ändern nur in der Vorlage)"


def test_test_count_in_the_readme_is_the_suite_size():
    assert f"## Verifikation ({n_tests()} Tests)" in README


def test_the_results_file_is_the_one_the_sweep_would_write_in_structure():
    data = json.loads((ROOT / "data" / "dtf_results.json").read_text(encoding="utf-8"))
    assert set(data["cells"]) == {c["id"] for c in C.CELLS}


# --- Urteilssätze als Zählungen über das ganze Raster -----------------------------------------------------------------------------------------

BIN = MR.BIN_IDS


def test_best_rule_below_five_percent_count_and_cells():
    under = [c for c in BIN if R.best_rule(RES["cells"][c])[1] < 5.0]
    assert under == ["q0", "q4", "q6"] and f"In {len(under)} von {len(BIN)} binären Zellen liegt die beste einfache Regel unter 5 %" in README


def test_all_three_rules_below_five_percent_only_without_noise():
    assert [c for c in BIN if all(RES["cells"][c][r]["mean"] < 5.0 for r in "MGB")] == ["q0"]


def test_noise_opens_the_gap_of_the_information_rule():
    assert RES["cells"]["q0"]["G"]["mean"] < 3.0 < 10.0 < min(RES["cells"][c]["G"]["mean"] for c in BIN if c != "q0")


def test_winner_counts_with_two_standard_errors():
    d = MR.derived(RES)
    assert (d["n_M"], d["n_G"], d["n_open"]) == (7, 3, 1)
    assert [c for c in BIN if d["sign"][c] == "G"] == ["q0", "q5", "g1"] and [c for c in BIN if d["sign"][c] == "offen"] == ["g2"]
    assert "Info/Kosten in 3 (Se/Sp 1,0/1,0; Irrtumspreis 100; Vorab gleichverteilt)" in README


def test_the_cheapest_useful_test_is_above_the_information_rule_in_every_cell():
    assert all(RES["cells"][c]["B"]["mean"] > RES["cells"][c]["G"]["mean"] for c in BIN)
    assert [c for c in BIN if RES["cells"][c]["B"]["mean"] < RES["cells"][c]["M"]["mean"]] == ["q5", "g1"]


def test_verdict_states_over_the_grid():
    d = MR.derived(RES)
    assert d["states"] == {"reicht": 2, "merklich": 6, "verfehlt": 3}
    assert [c for c in BIN if d["state"][c] == "verfehlt"] == ["q5", "g1", "g2"]


def test_the_threshold_of_the_third_state_was_set_after_seeing_the_largest_gap():
    assert max(g for _, g in MR.derived(RES)["best"].values()) < 15.0 and C.VERDICT_VERFEHLT == 10.0
    assert "steht die Schwelle darum bei 10 %" in README.replace("\n", " ") or "die Schwelle steht darum bei 10 %" in README


def test_the_thresholds_sit_inside_gaps_of_the_cell_values():
    """Die Schwellen 3 und 10 % sind Setzungen; sie dürfen nicht an einem Messwert kleben. Eigene Rechnung, ohne make_readme: jede Schwelle zwischen dem größten Wert
    darunter und dem kleinsten Wert darüber gibt in den 14 Zellen mit Regelvergleich dieselbe Einteilung, und die README nennt diese Spanne."""
    ids = [f"q{i}" for i in range(7)] + [f"g{i}" for i in range(4)] + ["u0", "u1", "u2"]
    vals = [min(RES["cells"][c][r]["mean"] for r in ("M", "G", "B")) for c in ids]
    assert len(vals) == 14

    def parts(t_reicht, t_verfehlt):
        return tuple(("reicht" if v < t_reicht else "verfehlt" if v >= t_verfehlt else "merklich") for v in vals)

    base = parts(C.VERDICT_REICHT, C.VERDICT_VERFEHLT)
    for thr, which in ((C.VERDICT_REICHT, 0), (C.VERDICT_VERFEHLT, 1)):
        below = max(v for v in vals if v < thr)
        above = min(v for v in vals if v >= thr)
        assert below < thr <= above
        assert above - below > 2.0, "die Lücke der Messwerte um die Schwelle ist zu eng, um sie eine Lücke zu nennen"
        for t in (below + 0.01, (below + above) / 2, above):
            pair = (t, C.VERDICT_VERFEHLT) if which == 0 else (C.VERDICT_REICHT, t)
            assert parts(*pair) == base
        # knapp außerhalb der Lücke ändert sich die Einteilung wirklich (die Spanne ist nicht zu großzügig angegeben)
        pair = (below, C.VERDICT_VERFEHLT) if which == 0 else (C.VERDICT_REICHT, below)
        assert parts(*pair) != base
    assert "dieselbe Einteilung" in README


def _paired_sign(cell, a, b):
    """Eigene Rechnung ohne make_readme: Lücke je Instanz nach der Abstimmung, Unterschied a − b gepaart, Vorzeichen mit zwei Standardfehlern."""
    import math
    n_tune = RES["meta"]["n_tune"]
    per = cell["per_instance"]
    d = [per[a][i] - per[b][i] for i in range(n_tune, len(per["opt"]))]  # je Instanz ist die Lücke der Regel in % gespeichert
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return "over" if m > 2 * se else "under" if m < -2 * se else "open"


def test_the_paired_counts_of_the_cheapest_rule_are_recomputed_independently():
    bin_ids = [f"q{i}" for i in range(7)] + [f"g{i}" for i in range(4)]
    for other in ("G", "M"):
        got = {"over": 0, "under": 0, "open": 0}
        for c in bin_ids:
            got[_paired_sign(RES["cells"][c], "B", other)] += 1
        assert got == MR.derived(RES)["pair_B" + other], other
        assert sum(got.values()) == 11
    # der Vergleich mit kurzsichtig ist nicht überall klar: „11 von 11“ gilt nur gegen Info/Kosten
    assert MR.derived(RES)["pair_BG"]["over"] == 11 and MR.derived(RES)["pair_BM"]["over"] < 11


def test_the_stored_mean_of_the_paired_difference_agrees_with_the_per_instance_data():
    c = RES["cells"]["q2"]
    m, s = MR.paired(c, "M", "G", RES["meta"]["n_tune"])
    assert abs(m - c["diff_MG"]["mean"]) < 1e-9 and abs(s - c["diff_MG"]["se"]) < 1e-9


def test_the_unclear_cells_keep_the_order_of_the_rules():
    for cid in ("u0", "u1", "u2"):
        c = RES["cells"][cid]
        assert c["M"]["mean"] < c["G"]["mean"] < c["B"]["mean"]
    d = MR.derived(RES)
    assert d["pair_MG_unclear"]["under"] == 3 and "Kurzsichtig liegt in 3 der drei Zellen mehr als zwei Standardfehler unter Info/Kosten" in README


def test_unclear_raises_the_optimum_by_the_stated_percentages():
    u = MR.derived(RES)["unclear"]
    assert u[0] == 0.0 and 12.0 < u[1] < 12.5 and 41.0 < u[2] < 41.5


def test_genome_step_between_the_measured_costs():
    c = RES["cells"]
    first = lambda y, k: c[f"y{y}c{k}"]["share_first"]  # noqa: E731
    assert first(50, 1) > 0.9 > 0.1 > first(50, 4) and 0.2 <= first(50, 2) <= 0.6        # Ausbeute 0,5: zwischen den Kosten 2 und 4 kippt es, bei 1 ist es fast immer der erste Test
    assert first(25, 1) > 0.2 and first(25, 2) == 0.0 and all(first(y, k) == 0.0 for y in (25, 50) for k in (8, 16))
    assert "zwischen den Kosten 1 und 2 beziehungsweise 2 und 4 mittleren Testkosten" in README


def test_genome_simple_policies_cost_more_at_the_wrong_end():
    c = RES["cells"]
    assert c["y50c1"]["never"]["mean"] > 40.0 and c["y50c4"]["forced"]["mean"] > 25.0 and c["y50c8"]["forced"]["mean"] > 100.0
    assert c["y50c1"]["forced"]["mean"] < 1.0 and c["y50c8"]["never"]["mean"] == 0.0


def test_the_optimum_with_genome_is_never_above_the_optimum_without():
    assert all(c["opt_mean"] <= c["opt_without_mean"] + 1e-9 for cid, c in RES["cells"].items() if cid.startswith("y"))


def test_base_rate_numbers_in_the_readme():
    import dtf_basisrate as BR
    assert BR.tests_until(1e-3, 0.99, 0.01, 0.9, False) == 2 and BR.tests_until(1e-3, 0.99, 0.01, 0.9, True) == 6
    assert round(BR.ppv(1e-3, 0.99, 0.01, 2), 2) == 0.91 and round(BR.ppv(1e-3, 0.99, 0.01, 2, True), 2) == 0.23
    assert "sind es 6 (nach zwei Tests 0,23)" in README and "(0,91 nach zwei)" in README


def test_standard_cell_numbers_by_formatting():
    q2 = RES["cells"]["q2"]
    assert f"{q2['M']['mean']:.1f}".replace(".", ",") in README and f"{q2['opt_mean']:.2f}".replace(".", ",") in README


def test_old_values_in_the_comparison_tables_come_from_the_report_script():
    from report import VORMESSUNG_REGELN
    for cid in ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "g1", "g2"):
        old = VORMESSUNG_REGELN[cid]
        assert f"{old[1]:.1f}".replace(".", ",") in README and f"{old[0]:.2f}".replace(".", ",") in README


def test_make_readme_check_mode_agrees():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "make_readme.py"), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr


def test_no_forbidden_wording_in_readme_and_app_texts():
    """Playbook §7/§11: „trägt/tragen“ nicht bildlich, „Saatwert“, „Seed“ ohne „Zufalls-“ als Eigenname der Einstellung; kein „geplant“ für Gebautes."""
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (README, app):
        assert "Saatwert" not in text and "trägt" not in text.replace("Kosten tragen", "") and "geplant" not in text
