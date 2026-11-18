"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen und prüft, dass
kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei Charts ohne key= und StreamlitAPIException bei Slidern mit min == max).
Dazu die Eckfall-Matrix (DEMO-PLAYBOOK §1d): jeder Regler einzeln an Minimum und Maximum, alle Minima und alle Maxima zusammen, der Permalink."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import dtf_basisrate as BR
import dtf_constants as C
import dtf_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen
REGLER_ARTEN = ("slider", "select_slider", "checkbox", "number_input")


def _fresh_app(**query) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    for k, v in query.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


NUR_MIT_GENOM = ("gcost_select", "gyield_select")  # ohne Genomtest ausgegraut (AppTest darf sie dann nicht setzen)


def _widget(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert len(at.metric) == 4 and at.title and "Diagnose-Testfolge" in at.title[0].value


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
@pytest.mark.parametrize("edge", ["lo", "hi"])
def test_every_control_at_its_minimum_and_maximum_does_not_raise(state_key, edge):
    at = _fresh_app()
    spec = P.SETTING_SPECS[state_key]
    if state_key in NUR_MIT_GENOM:
        at.sidebar.checkbox(key="genome_check").set_value(True).run(timeout=TIMEOUT)
    _widget(at, state_key).set_value(spec.lo if edge == "lo" else spec.hi).run(timeout=TIMEOUT)
    assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("edge", ["lo", "hi"])
def test_all_controls_at_their_minima_or_maxima_together_do_not_raise(edge):
    at = _fresh_app()
    at.sidebar.checkbox(key="genome_check").set_value(True).run(timeout=TIMEOUT)          # schaltet die Genom-Regler frei; „lo“ schaltet ihn danach wieder aus
    for state_key, spec in P.SETTING_SPECS.items():
        if state_key == "genome_check":
            continue
        _widget(at, state_key).set_value(spec.lo if edge == "lo" else spec.hi)
    _widget(at, "genome_check").set_value(edge == "hi")
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_calculator_widgets_in_the_main_area_do_not_raise():
    steps = {"ppv_prev": BR.PREVALENCE_OPTIONS, "ppv_se": BR.SE_OPTIONS, "ppv_fpr": BR.FPR_OPTIONS}
    at = _fresh_app()
    for key, opts in steps.items():
        for value in (opts[0], opts[-1]):
            at.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])
    for k in (1, 6):
        at.slider(key="ppv_k").set_value(k).run(timeout=TIMEOUT)
        assert not at.exception


def test_permalink_sets_the_controls_and_snaps_invalid_values():
    at = _fresh_app(l="100", z="1.5", q="0,95 / 0,95", g="True", gc="4.0", gy="0.25", m="6", n="10", seed="7")
    assert at.sidebar.select_slider(key="loss_select").value == 100
    assert at.sidebar.select_slider(key="zipf_select").value == 1.5
    assert at.sidebar.select_slider(key="quality_select").value == "0,95 / 0,95"
    assert at.sidebar.checkbox(key="genome_check").value is True
    assert at.sidebar.select_slider(key="gcost_select").value == 4.0 and at.sidebar.select_slider(key="gyield_select").value == 0.25
    assert at.sidebar.slider(key="m_slider").value == 6 and at.sidebar.select_slider(key="n_select").value == 10 and at.sidebar.number_input(key="seed_input").value == 7
    at2 = _fresh_app(l="55", q="kaputt", m="99")                                          # ungültig: Stufe ignoriert, Zahl begrenzt
    assert at2.sidebar.select_slider(key="loss_select").value == C.DEFAULT_LOSS
    assert at2.sidebar.select_slider(key="quality_select").value == C.DEFAULT_QUALITY and at2.sidebar.slider(key="m_slider").value == C.M_MAX


def test_the_address_bar_mirrors_the_configuration():
    at = _fresh_app()
    at.sidebar.select_slider(key="loss_select").set_value(100).run(timeout=TIMEOUT)
    assert at.query_params["l"] == ["100"] or at.query_params["l"] == "100"


def test_the_genome_controls_are_disabled_without_the_genome_test():
    at = _fresh_app()
    assert at.sidebar.select_slider(key="gcost_select").disabled and at.sidebar.select_slider(key="gyield_select").disabled
    at.sidebar.checkbox(key="genome_check").set_value(True).run(timeout=TIMEOUT)
    assert not at.sidebar.select_slider(key="gcost_select").disabled


def test_the_hint_about_capped_tests_appears_exactly_when_tests_are_capped():
    at = _fresh_app()
    assert not [i for i in at.info if "einfache Tests gerechnet" in i.value]
    at.sidebar.select_slider(key="unclear_select").set_value(0.1).run(timeout=TIMEOUT)
    assert [i for i in at.info if "7 einfache Tests gerechnet, nicht 8" in i.value]
    at.sidebar.checkbox(key="genome_check").set_value(True).run(timeout=TIMEOUT)
    assert [i for i in at.info if "6 einfache Tests gerechnet, nicht 8" in i.value]


def test_the_three_kernfragen_give_a_verdict_each():
    at = _fresh_app()
    headings = [m.value for m in at.markdown if m.value.startswith("## ❓ Kernfrage")]
    assert len(headings) == 3
    verdicts = [v.value for group in (at.success, at.info, at.warning) for v in group]
    assert any("einfache Regel" in v for v in verdicts) and any("Genomtest" in v for v in verdicts) and any("positiven Tests" in v for v in verdicts)
