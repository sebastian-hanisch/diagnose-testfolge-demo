"""Abnahmekriterien der Presets: Was jedes Preset zeigen soll, als Prüfung gegen die vorgerechnete Messreihe (tests/test_stories.py).

Vorzeichen-Kriterien haben immer eine Standardfehler-Bedingung (Unterschied mehr als zwei Standardfehler), Schwellen stehen in dtf_constants."""

import dtf_results as R


def _cell(res, cid):
    return res["cells"][cid]


def _clearly_positive(d: dict) -> bool:
    """Mittel größer als zwei Standardfehler (Vorzeichen gesichert)."""
    return d["mean"] > 2.0 * d["se"]


def story_standard(res):
    c = _cell(res, "q2")
    return c["M"]["mean"] < c["G"]["mean"] < c["B"]["mean"]


def story_ohne_rauschen(res):
    c = _cell(res, "q0")
    return R.verdict_state(R.best_rule(c)[1]) == "reicht"


def story_gute_tests(res):
    c = _cell(res, "q1")
    return c["G"]["mean"] >= 10.0 and c["B"]["mean"] >= 25.0


def story_teurer_irrtum(res):
    return _clearly_positive(_cell(res, "q5")["diff_MG"])


def story_alle_gleich(res):
    return _clearly_positive(_cell(res, "g1")["diff_MG"])


def story_genom(res):
    return 0.2 <= _cell(res, "y50c2")["share_first"] <= 0.6


STORIES = {
    "Standard (Voreinstellung)": ("kurzsichtig liegt unter Info/Kosten und diese unter dem billigsten Test (Lücken im Mittel)", story_standard, "q2"),
    "Ohne Rauschen": ("die beste einfache Regel liegt unter 3 % über dem Optimum (Meldung „reicht“)", story_ohne_rauschen, "q0"),
    "Gute Tests": ("Info/Kosten mindestens 10 %, billigster nützlicher Test mindestens 25 % über dem Optimum", story_gute_tests, "q1"),
    "Teurer Irrtum": ("kurzsichtig minus Info/Kosten positiv mit mehr als zwei Standardfehlern (Info/Kosten liegt vorn)", story_teurer_irrtum, "q5"),
    "Alle gleich wahrscheinlich": ("kurzsichtig minus Info/Kosten positiv mit mehr als zwei Standardfehlern (Info/Kosten liegt vorn)", story_alle_gleich, "g1"),
    "Genom zuerst?": ("der Genomtest ist in 20 bis 60 % der Instanzen der erste Test", story_genom, "y50c2"),
}


def check_all(res) -> dict:
    """Name -> bool für jedes Preset."""
    return {name: fn(res) for name, (_, fn, _) in STORIES.items()}
