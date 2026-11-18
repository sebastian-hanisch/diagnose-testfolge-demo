"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink (Einrasten und Begrenzen)."""

import pytest

import dtf_constants as C
import dtf_model as M
import dtf_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_for_their_controls():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.options is not None:
                assert value in spec.options, (name, key, value)
            else:
                assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_equals_the_default_settings():
    assert M.Settings(**C.PRESETS["Standard (Voreinstellung)"]) == M.Settings()
    assert next(iter(C.PRESETS)) == "Standard (Voreinstellung)"


def test_setting_specs_cover_every_preset_key_and_every_settings_field():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == set(M.Settings.__dataclass_fields__)


def test_defaults_in_the_specs_match_the_settings_defaults():
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == getattr(M.Settings(), key), key


def test_presets_differ_from_the_standard_in_exactly_the_described_controls():
    base = C.PRESETS["Standard (Voreinstellung)"]
    changed = {name: {k for k in p if p[k] != base[k]} for name, p in C.PRESETS.items()}
    assert changed["Ohne Rauschen"] == {"quality"} and changed["Gute Tests"] == {"quality"}
    assert changed["Teurer Irrtum"] == {"loss"} and changed["Alle gleich wahrscheinlich"] == {"zipf"}
    assert changed["Genom zuerst?"] == {"m", "genome"}


def test_url_params_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


@pytest.mark.parametrize("state_key,raw,expected", [
    ("m_slider", "99", C.M_MAX), ("m_slider", "0", C.M_MIN), ("m_slider", "6", 6), ("m_slider", "abc", None),
    ("seed_input", "-5", 0), ("seed_input", "100000", C.SEED_MAX),
    ("n_select", "6", 6), ("n_select", "5", None), ("zipf_select", "1.5", 1.5), ("zipf_select", "0.7", None),
    ("quality_select", "0,95 / 0,95", "0,95 / 0,95"), ("quality_select", "0,5 / 0,5", None),
    ("loss_select", "100", 100), ("loss_select", "50", None), ("genome_check", "True", True), ("genome_check", "false", False),
])
def test_permalink_values_snap_to_the_steps_or_are_clamped(state_key, raw, expected):
    assert P.parse_value(P.SETTING_SPECS[state_key], raw) == expected
