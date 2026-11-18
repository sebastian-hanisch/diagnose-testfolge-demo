"""Permalink (Begrenzen/Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Auswahlregler (select_slider) rasten auf ihre Stufen ein: ein Permalink mit einem Wert außerhalb der Stufen wird ignoriert, Zahlen werden auf die Reglergrenzen begrenzt."""

import random
from dataclasses import dataclass

import streamlit as st

import dtf_constants as C
from dtf_model import Settings


def _bool(text):
    return str(text).lower() in ("1", "true", "wahr")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: object = None
    hi: object = None
    options: tuple | None = None   # Stufen eines Auswahlreglers; lo und hi sind dann die erste und die letzte Stufe


SETTING_SPECS = {
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "n_select": SettingSpec("n", int, C.DEFAULT_N, C.N_OPTIONS[0], C.N_OPTIONS[-1], C.N_OPTIONS),
    "zipf_select": SettingSpec("z", float, C.DEFAULT_ZIPF, C.ZIPF_OPTIONS[0], C.ZIPF_OPTIONS[-1], C.ZIPF_OPTIONS),
    "quality_select": SettingSpec("q", str, C.DEFAULT_QUALITY, C.QUALITY_OPTIONS[0], C.QUALITY_OPTIONS[-1], C.QUALITY_OPTIONS),
    "loss_select": SettingSpec("l", int, C.DEFAULT_LOSS, C.LOSS_OPTIONS[0], C.LOSS_OPTIONS[-1], C.LOSS_OPTIONS),
    "unclear_select": SettingSpec("u", float, C.DEFAULT_UNCLEAR, C.UNCLEAR_OPTIONS[0], C.UNCLEAR_OPTIONS[-1], C.UNCLEAR_OPTIONS),
    "genome_check": SettingSpec("g", _bool, False, False, True, (False, True)),
    "gcost_select": SettingSpec("gc", float, C.DEFAULT_GENOME_COST, C.GENOME_COST_OPTIONS[0], C.GENOME_COST_OPTIONS[-1], C.GENOME_COST_OPTIONS),
    "gyield_select": SettingSpec("gy", float, C.DEFAULT_GENOME_YIELD, C.GENOME_YIELD_OPTIONS[0], C.GENOME_YIELD_OPTIONS[-1], C.GENOME_YIELD_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
# Preset-Schlüssel (Felder von Settings) -> Reglerschlüssel
PRESET_KEYS = {"m": "m_slider", "n": "n_select", "zipf": "zipf_select", "quality": "quality_select", "loss": "loss_select", "u": "unclear_select",
               "genome": "genome_check", "g_cost": "gcost_select", "g_yield": "gyield_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key):
    return SETTING_SPECS[state_key].options


def parse_value(spec: SettingSpec, raw):
    """Wert aus der Adresszeile: auf Stufen einrasten (Auswahl) bzw. auf die Grenzen begrenzen (Zahl); None, wenn unbrauchbar."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.options is not None:
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = parse_value(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem
    Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf
    (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def settings_from_state() -> Settings:
    """Settings aus dem Session State (jeder Regler, fehlende Werte = Standard)."""
    g = {key: st.session_state.get(state_key, SETTING_SPECS[state_key].default) for key, state_key in PRESET_KEYS.items()}
    return Settings(**g)
