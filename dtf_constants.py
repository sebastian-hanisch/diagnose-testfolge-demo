"""Konstanten: Regler-Stufen, feste Annahmen, Farben, Presets und die gemessenen Zellen (Gerüst aus tools/new_demo.py, DEMO-PLAYBOOK §1)."""

# --- Regler (Stufen sind die gemessenen Zellen, siehe CELLS) -------------------------------------------------------------------------------
M_MIN, M_MAX, DEFAULT_M = 5, 8, 8                  # höchstens 8 Tests: die exakte Rechnung hat 3^m Zustände (rund 0,2 s bei m = 8, maschinenabhängig)
N_OPTIONS = (4, 6, 10)                             # Zahl der Diagnosen
DEFAULT_N = 6
ZIPF_OPTIONS = (0.0, 1.0, 1.5)                     # Schiefe des Vorab (Zipf-Exponent): gleichverteilt / schief / sehr schief
ZIPF_LABELS = {0.0: "gleich", 1.0: "schief", 1.5: "sehr schief"}
DEFAULT_ZIPF = 1.0
QUALITY_OPTIONS = ("1,0 / 1,0", "0,95 / 0,95", "0,9 / 0,9", "0,8 / 0,9", "0,7 / 0,8")  # Se / Sp
QUALITY_VALUES = {"1,0 / 1,0": (1.0, 1.0), "0,95 / 0,95": (0.95, 0.95), "0,9 / 0,9": (0.9, 0.9), "0,8 / 0,9": (0.8, 0.9), "0,7 / 0,8": (0.7, 0.8)}
DEFAULT_QUALITY = "0,9 / 0,9"
LOSS_OPTIONS = (5, 20, 100)                        # Irrtumspreis L in mittleren Testkosten
DEFAULT_LOSS = 20
UNCLEAR_OPTIONS = (0.0, 0.1, 0.3)                  # Anteil „unklar“
DEFAULT_UNCLEAR = 0.0
GENOME_COST_OPTIONS = (1.0, 2.0, 4.0, 8.0)         # Kosten des Genomtests in mittleren Testkosten
DEFAULT_GENOME_COST = 2.0
GENOME_YIELD_OPTIONS = (0.25, 0.5)                 # Ausbeute: Wahrscheinlichkeit, die Diagnose eindeutig zu finden
DEFAULT_GENOME_YIELD = 0.5
SEED_MAX, DEFAULT_SEED = 299, 35                   # Anzeige-Seeds 0 bis 299; die Messreihe nutzt Seeds ab SWEEP_SEED_BASE (nie dieselben)

# --- feste Annahmen des Modells --------------------------------------------------------------------------------------------------------
DENSITY = 0.5                                      # Anteil der Diagnosen, die ein Test anspricht
COST_SIGMA = 1.0                                   # Streuung der logarithmierten Testkosten (Mittel danach auf 1 normiert)
M_CAP_UNCLEAR, M_CAP_GENOME = 7, 6                 # bei „unklar“ höchstens 7, mit Genomtest höchstens 6 binäre Tests (Rechenzeit, Plan Abschnitt 5)

# --- Messreihe ---------------------------------------------------------------------------------------------------------------------------
TAUS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)           # Stoppschwellen der Regeln „Info/Kosten“ und „billigster“
N_INSTANCES, N_TUNE = 120, 60                      # je Zelle; τ auf den ersten N_TUNE abgestimmt, Lücke auf den übrigen gemessen
SWEEP_SEED_BASE = 100000                           # Seed der Instanz i in Zelle c: BASE + 1000 * c + i
GAP_MARK = 5.0                                     # „Lücke über 5 %“ für den Anteil der Instanzen
VERDICT_REICHT, VERDICT_VERFEHLT = 3.0, 10.0       # Schwellen der Meldung in drei Zuständen (Lücke der besten einfachen Regel in %)

# --- Farben --------------------------------------------------------------------------------------------------------------------------------
COLOR_OPT = "#1f2733"
COLOR_M, COLOR_G, COLOR_B = "#2a7de1", "#c77700", "#8a96a8"
COLOR_GOOD = "#2F6B65"

# --- Presets (Name -> Einstellungen; alle Regler, damit die Hervorhebung zustandslos stimmt) ------------------------------------------------
_BASIS = {"m": DEFAULT_M, "n": DEFAULT_N, "zipf": DEFAULT_ZIPF, "quality": DEFAULT_QUALITY, "loss": DEFAULT_LOSS, "u": DEFAULT_UNCLEAR,
          "genome": False, "g_cost": DEFAULT_GENOME_COST, "g_yield": DEFAULT_GENOME_YIELD, "seed": DEFAULT_SEED}
PRESETS = {
    "Standard (Voreinstellung)": dict(_BASIS),
    "Ohne Rauschen": {**_BASIS, "quality": "1,0 / 1,0"},
    "Gute Tests": {**_BASIS, "quality": "0,95 / 0,95"},
    "Teurer Irrtum": {**_BASIS, "loss": 100},
    "Alle gleich wahrscheinlich": {**_BASIS, "zipf": 0.0},
    "Genom zuerst?": {**_BASIS, "m": M_CAP_GENOME, "genome": True},
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "8 Tests, 6 Diagnosen mit schiefem Vorab, Testgüte 0,9 / 0,9, Irrtumspreis 20 mittlere Testkosten.",
    "Ohne Rauschen": "Alle Tests fehlerfrei (Sensitivität und Spezifität 1,0), sonst wie der Standard. Frage: Reicht hier eine einfache Regel?",
    "Gute Tests": "Testgüte 0,95 / 0,95 statt 0,9 / 0,9, sonst wie der Standard. Frage: Was ändert schon kleines Rauschen?",
    "Teurer Irrtum": "Irrtumspreis 100 statt 20 mittlere Testkosten, sonst wie der Standard. Frage: Welche Regel liegt jetzt vorn?",
    "Alle gleich wahrscheinlich": "Vorab gleichverteilt statt schief, sonst wie der Standard. Frage: Welche Regel liegt ohne Schiefe vorn?",
    "Genom zuerst?": "Ein Genomtest (Kosten 2 mittlere Testkosten, Ausbeute 0,5) neben 6 einfachen Tests. Frage: Wird er als erster Test genommen?",
}

# --- gemessene Zellen -----------------------------------------------------------------------------------------------------------------------
# Jede Zelle: id, Art ("bin" = binäre Tests, "unklar" = dritter Ausgang, "genom" = Genomtest), Parameter. Die Reihenfolge legt die Seed-Blöcke fest.
_BIN = {"kind": "bin", "n": 6, "m": 8, "zipf": 1.0, "se": 0.9, "sp": 0.9, "loss": 20, "u": 0.0}
CELLS = []
for _i, (_se, _sp, _l) in enumerate([(1.0, 1.0, 20), (0.95, 0.95, 20), (0.9, 0.9, 20), (0.8, 0.9, 20), (0.9, 0.9, 5), (0.9, 0.9, 100), (0.7, 0.8, 20)]):
    CELLS.append({**_BIN, "id": f"q{_i}", "se": _se, "sp": _sp, "loss": _l})
for _i, (_n, _z) in enumerate([(4, 1.0), (6, 0.0), (6, 1.5), (10, 1.0)]):
    CELLS.append({**_BIN, "id": f"g{_i}", "n": _n, "zipf": _z})
for _i, _u in enumerate(UNCLEAR_OPTIONS):
    CELLS.append({**_BIN, "id": f"u{_i}", "kind": "unklar", "m": 6, "u": _u})
for _y in GENOME_YIELD_OPTIONS:
    for _c in (1.0, 2.0, 4.0, 8.0, 16.0):
        CELLS.append({**_BIN, "id": f"y{int(_y * 100)}c{int(_c)}", "kind": "genom", "m": 6, "g_yield": _y, "g_cost": _c})
GENOME_COSTS_MEASURED = (1.0, 2.0, 4.0, 8.0, 16.0)
