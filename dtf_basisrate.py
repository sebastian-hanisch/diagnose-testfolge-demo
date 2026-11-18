"""Basisratenfalle: Wahrscheinlichkeit, wirklich krank zu sein (PPV) nach k positiven Tests, mit und ohne gemeinsamen Fehler der Tests.

Kranke: jeder Test positiv mit Wahrscheinlichkeit se, unabhängig. Gesunde, unabhängiger Fall: jeder Test mit Wahrscheinlichkeit fpr falsch positiv.
Gesunde, gemeinsamer Fehler: ein Anteil a der Gesunden trägt einen Störfaktor, der jeden Test mit Wahrscheinlichkeit h positiv macht; die übrigen sind
mit Wahrscheinlichkeit e falsch positiv, e so, dass die Rand-Falsch-positiv-Rate je Test unverändert fpr bleibt: a·h + (1 − a)·e = fpr.
Beide Modelle geben nach EINEM positiven Test denselben PPV; sie unterscheiden sich erst bei mehreren."""

DISTURB_SHARE, DISTURB_PROB = 0.02, 0.4            # a und h (angenommen, nicht gemessen)
PREVALENCE_OPTIONS = (1e-4, 1e-3, 1e-2, 5e-2)
PREVALENCE_LABELS = {1e-4: "1 von 10.000", 1e-3: "1 von 1.000", 1e-2: "1 von 100", 5e-2: "1 von 20"}
SE_OPTIONS = (0.9, 0.95, 0.99)
FPR_OPTIONS = (0.01, 0.02, 0.05)                   # Falsch-positiv-Rate je Test (mindestens a·h = 0,008, sonst gäbe es kein e ≥ 0)
K_OPTIONS = (1, 2, 3, 4, 5, 6)
REFERRALS = (("Bevölkerung", 1e-4), ("Hausarzt", 1e-3), ("Überweisung", 0.02), ("Spezialambulanz", 0.2))
REFERRAL_LABELS = {1e-4: "1 von 10.000", 1e-3: "1 von 1.000", 0.02: "1 von 50", 0.2: "1 von 5"}


def residual_fpr(fpr: float, a: float = DISTURB_SHARE, h: float = DISTURB_PROB) -> float:
    """e = (fpr − a·h) / (1 − a): Falsch-positiv-Rate der Gesunden ohne Störfaktor."""
    if fpr < a * h:
        raise ValueError("Falsch-positiv-Rate kleiner als a·h: ein Störfaktor mit diesen Werten ist nicht möglich")
    return (fpr - a * h) / (1.0 - a)


def ppv(prevalence: float, se: float, fpr: float, k: int, shared_error: bool = False) -> float:
    """Wahrscheinlichkeit krank zu sein, wenn k Tests positiv sind."""
    sick = prevalence * se ** k
    if shared_error:
        e = residual_fpr(fpr)
        healthy = (1.0 - prevalence) * (DISTURB_SHARE * DISTURB_PROB ** k + (1.0 - DISTURB_SHARE) * e ** k)
    else:
        healthy = (1.0 - prevalence) * fpr ** k
    return sick / (sick + healthy)


def tests_until(prevalence: float, se: float, fpr: float, target: float, shared_error: bool, k_max: int = 12):
    """Kleinste Zahl positiver Tests mit PPV ≥ target, sonst None."""
    for k in range(1, k_max + 1):
        if ppv(prevalence, se, fpr, k, shared_error) >= target:
            return k
    return None
