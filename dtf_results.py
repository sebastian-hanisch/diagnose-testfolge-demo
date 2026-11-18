"""Vorgerechnete Messreihe (data/dtf_results.json, erzeugt von tools/sweep.py): Laden, Zuordnung einer Einstellung zur nächstliegenden gemessenen Zelle,
Urteil in drei Zuständen. Die App rechnet die Messreihe nie im Browser."""

import json
from functools import lru_cache
from pathlib import Path

import dtf_constants as C
from dtf_model import Settings, effective_m

DATA = Path(__file__).resolve().parent / "data" / "dtf_results.json"

# Namen der Parameter, nach denen eine Einstellung mit einer Zelle verglichen wird (für den Hinweis „weicht ab in: …“)
PARAM_NAMES = {"n": "Zahl der Diagnosen", "m": "Zahl der Tests", "zipf": "Vorab", "se_sp": "Testgüte", "loss": "Irrtumspreis", "u": "Anteil unklar",
               "g_cost": "Genomkosten", "g_yield": "Genomausbeute", "genome": "Genomtest"}


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def cell(cell_id: str) -> dict:
    return load()["cells"][cell_id]


def _cell_kind(s: Settings) -> str:
    return "genom" if s.genome else ("unklar" if s.u > 0 else "bin")


def _params(s: Settings) -> dict:
    se, sp = C.QUALITY_VALUES[s.quality]
    return {"n": s.n, "m": effective_m(s.m, s.u, s.genome), "zipf": s.zipf, "se_sp": (se, sp), "loss": s.loss, "u": s.u,
            "g_cost": s.g_cost if s.genome else None, "g_yield": s.g_yield if s.genome else None}


def _cell_params(c: dict) -> dict:
    return {"n": c["n"], "m": c["m"], "zipf": c["zipf"], "se_sp": (c["se"], c["sp"]), "loss": c["loss"], "u": c["u"],
            "g_cost": c.get("g_cost"), "g_yield": c.get("g_yield")}


def nearest_cell(s: Settings):
    """(Zell-Id, Liste der Parameter, in denen die Einstellung von der Zelle abweicht). Eine Zelle derselben Art (binär, „unklar“, Genom) wird immer vorgezogen;
    unter diesen gewinnt die mit den wenigsten abweichenden Parametern, bei Gleichstand die erste in der Reihenfolge von C.CELLS."""
    kind = _cell_kind(s)
    want = _params(s)
    best = None
    for c in C.CELLS:
        if c["kind"] != kind:
            continue
        have = _cell_params(c)
        diff = [k for k in want if want[k] != have[k]]
        # Genomkosten außerhalb der gemessenen Stufen gibt es bei den Reglern nicht; bei binären Zellen sind g_* None und gleich
        if best is None or len(diff) < len(best[1]):
            best = (c["id"], diff)
    return best


def best_rule(cell_result: dict):
    """(Regel, Lücke in %) der besten einfachen Regel einer binären oder „unklar“-Zelle."""
    return min(((r, cell_result[r]["mean"]) for r in ("M", "G", "B")), key=lambda x: x[1])


def verdict_state(best_gap: float) -> str:
    """Meldung in drei Zuständen aus der Lücke der besten einfachen Regel: 'reicht' (unter 3 %), 'merklich' (3 bis 10 %), 'verfehlt' (über 10 %)."""
    if best_gap < C.VERDICT_REICHT:
        return "reicht"
    if best_gap <= C.VERDICT_VERFEHLT:
        return "merklich"
    return "verfehlt"
