"""Tabellen, Figuren und PDF: reine Funktionen und Rauchtests an Standardinstanz, jedem Preset und den Rändern."""

import pytest

import dtf_constants as C
import dtf_evaluation as ev
import dtf_pdf_export as PDF
import dtf_results as R
import dtf_ui_panel as UI
import dtf_visualization as V
from dtf_model import Settings

RES = R.load()


@pytest.fixture(scope="module")
def default_analysis():
    return ev.analyse(Settings())


def test_analysis_numbers_are_consistent(default_analysis):
    a = default_analysis
    assert a.opt.total == pytest.approx(a.opt.test_cost + a.opt.loss)
    assert all(a.gap(r) >= -1e-9 for r in ("M", "G", "B"))
    assert a.best_rule() in ("M", "G", "B") and a.gap(a.best_rule()) == min(a.gap(r) for r in ("M", "G", "B"))
    assert a.m_total == 8 and len(a.tests) == 8 and abs(sum(a.prior) - 1.0) < 1e-12
    assert a.genome is None and a.rule_cell == "q2" and a.rule_diff == ()


def test_analysis_uses_the_threshold_of_the_nearest_cell(default_analysis):
    assert default_analysis.tau == RES["cells"]["q2"]["tau"]


def test_analysis_with_genome_reports_forced_and_never_gaps():
    a = ev.analyse(Settings(**C.PRESETS["Genom zuerst?"]))
    assert a.genome is not None and a.genome["forced_gap"] >= -1e-9 and a.genome["never_gap"] >= -1e-9 and a.m_total == 7
    assert a.tests[-1][0] == "Genom"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_gives_tables_figures_and_a_pdf(name):
    s = Settings(**C.PRESETS[name])
    a = ev.analyse(s)
    assert UI.rules_rows(a)[0]["Strategie"] == "Optimum (exakt)" and len(UI.rules_rows(a)) == 4 and len(UI.tests_rows(a)) == a.m_total
    fig = V.build_tree(a.tree, s.u > 0, len(a.prior))
    assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange
    pdf = PDF.build_pdf(a, UI.settings_text(s))
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500


def test_extreme_settings_give_figures_and_a_pdf():
    for s in (Settings(m=5, n=4, quality="1,0 / 1,0", loss=5), Settings(m=8, n=10, zipf=1.5, quality="0,7 / 0,8", loss=100, u=0.3, genome=True, g_cost=8.0, g_yield=0.25)):
        a = ev.analyse(s)
        assert V.build_tree(a.tree, s.u > 0, len(a.prior)) and V.build_prior(a.prior) and PDF.build_pdf(a, UI.settings_text(s))[:4] == b"%PDF"


def test_all_figures_have_fixed_axes():
    figs = [V.build_gap_bars(RES["cells"]["q2"], "x"), V.build_regime(RES, ["q0", "q1"], ["a", "b"]), V.build_genome_curve(RES, 0.5), V.build_genome_excess(RES, 0.25),
            V.build_ppv([1, 2], [0.1, 0.9], [0.1, 0.2]), V.build_prior([0.5, 0.5])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and fig.layout.dragmode is False


def test_the_tree_shows_every_outcome_label_kind():
    assert V.outcome_label("T1", 0, False, 4) == "negativ" and V.outcome_label("T1", 1, False, 4) == "positiv"
    assert [V.outcome_label("T1", o, True, 4) for o in range(3)] == ["negativ", "unklar", "positiv"]
    assert V.outcome_label("Genom", 2, False, 4) == "D3 gefunden" and V.outcome_label("Genom", 4, False, 4) == "kein Befund"


def test_the_tree_uses_short_edge_labels():
    assert [V.outcome_short("T1", o, False, 4) for o in range(2)] == ["–", "+"] and [V.outcome_short("T1", o, True, 4) for o in range(3)] == ["–", "?", "+"]
    assert V.outcome_short("Genom", 2, False, 4) == "D3" and V.outcome_short("Genom", 4, False, 4) == "kein"


def test_pdf_text_replaces_special_characters_and_never_raises():
    assert PDF.pdf_text("τ ≥ 0,9 – „ja“ → 3 × 2 −1") == 'tau >= 0,9 - "ja" -> 3 x 2 -1'
    assert PDF.pdf_text("ä ö ü ß") == "ä ö ü ß"
    PDF.pdf_text("☃ unbekannt")                                                          # Rest wird zu „?“, kein Fehler


def test_cell_label_and_diff_note():
    label = UI.cell_label(RES["cells"]["q2"]["params"])
    assert "6 Diagnosen, 8 Tests, Vorab schief, Testgüte 0,9 / 0,9, Irrtumspreis 20" == label
    assert UI.diff_note(()) == "" and UI.diff_note(("m", "loss")).endswith("Zahl der Tests, Irrtumspreis.")
    assert "Genomtest Kosten 2,0, Ausbeute 0,50" in UI.cell_label(RES["cells"]["y50c2"]["params"])


def test_table_builders_cover_every_measured_group():
    assert len(UI.messreihe_rows(RES, [f"q{i}" for i in range(7)] + [f"g{i}" for i in range(4)])) == 11
    assert [r["Anteil unklar"] for r in UI.unclear_rows(RES)] == ["0,00", "0,10", "0,30"]
    assert [r["Genomkosten"] for r in UI.genome_rows(RES, 0.5)] == ["1,0", "2,0", "4,0", "8,0", "16,0"]
    assert [r["Anteil Kranke"] for r in UI.referral_rows()] == ["1 von 10.000", "1 von 1.000", "1 von 50", "1 von 5"]
    assert [r["Annahme"] for r in UI.tests_until_rows(1e-3, 0.99, 0.01)] == ["unabhängig gerechnet", "mit gemeinsamem Fehler"]


def test_readme_tables_are_markdown_tables_with_the_right_row_counts():
    t = UI.readme_tables(RES)
    assert set(t) == {"regeln", "unklar", "genom", "ppv"}
    assert len(t["regeln"].splitlines()) == 2 + 11 and len(t["unklar"].splitlines()) == 2 + 3 and len(t["genom"].splitlines()) == 2 + 10 and len(t["ppv"].splitlines()) == 2 + 4
    assert all("|" in line for table in t.values() for line in table.splitlines())
