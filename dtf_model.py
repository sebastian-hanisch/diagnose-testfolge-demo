"""Modell: Einstellungen, Instanz-Erzeugung (nur SplitMix64), Likelihood-Tabellen und das Problem mit Posterior und Zuständen.

Eine Instanz: n Diagnosen mit Vorab p_d (Zipf), m binäre Tests (Test t spricht eine Teilmenge der Diagnosen an), Kosten je Test (lognormal, Mittel 1),
Irrtumspreis L. Dazu optional ein dritter Ausgang „unklar“ (Wahrscheinlichkeit u, unabhängig von der Diagnose) und ein Genomtest mit n + 1 Ausgängen
(Diagnose d mit Wahrscheinlichkeit y eindeutig gefunden, sonst „kein Befund“). Zustand = Tupel je Test: 0 ungenutzt, o + 1 = Ausgang o beobachtet."""

import decimal
import math
from dataclasses import dataclass

import numpy as np

import dtf_constants as C
from dtf_rng import SplitMix64

_CTX = decimal.Context(prec=45)


def exact_exp(x: float) -> float:
    """e^x auf jeder Plattform bitgleich (decimal, 45 Stellen, einmal auf double gerundet); math.exp kommt aus der C-Bibliothek."""
    return float(_CTX.exp(decimal.Decimal(x)))


@dataclass(frozen=True)
class Settings:
    """Alle Regler der Seitenleiste (jeder Regler steht als Feld hier und in dtf_presets.SETTING_SPECS, nie positional übergeben)."""
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    zipf: float = C.DEFAULT_ZIPF
    quality: str = C.DEFAULT_QUALITY
    loss: int = C.DEFAULT_LOSS
    u: float = C.DEFAULT_UNCLEAR
    genome: bool = False
    g_cost: float = C.DEFAULT_GENOME_COST
    g_yield: float = C.DEFAULT_GENOME_YIELD
    seed: int = C.DEFAULT_SEED


def effective_m(m: int, u: float, genome: bool) -> int:
    """Zahl der binären Tests, die wirklich gerechnet werden (Rechenzeit: 4^m Zustände bei „unklar“, ein Test mehr mit Genom)."""
    cap = C.M_CAP_GENOME if genome else (C.M_CAP_UNCLEAR if u > 0 else C.M_MAX)
    return max(1, min(m, cap))


@dataclass(frozen=True)
class Instance:
    prior: tuple          # Vorab je Diagnose, Summe 1
    responds: tuple       # je binärem Test: 0/1 je Diagnose (1 = der Test spricht die Diagnose an)
    costs: tuple          # je binärem Test, im Mittel 1
    se: float
    sp: float
    loss: float
    u: float = 0.0
    genome: tuple | None = None   # (Kosten, Ausbeute) oder None


def zipf_weights(n: int, s: float) -> list:
    """1 / Rang^s für s in {0, 1, 1,5} ohne pow der C-Bibliothek (1,5: r · √r, die Wurzel ist exakt gerundet)."""
    if s == 0.0:
        return [1.0] * n
    if s == 1.0:
        return [1.0 / r for r in range(1, n + 1)]
    if s == 1.5:
        return [1.0 / (r * math.sqrt(r)) for r in range(1, n + 1)]
    raise ValueError(f"Zipf-Exponent {s} nicht vorgesehen")


def make_instance(seed: int, n: int, m: int, zipf: float, se: float, sp: float, loss: float, u: float = 0.0, genome: tuple | None = None) -> Instance:
    """Eine Instanz aus einem Zufalls-Seed. Reihenfolge der Ziehungen: Vorab mischen, Testmatrix, Kosten (feste Folge, damit Messreihe und App dasselbe sehen)."""
    rng = SplitMix64(seed)
    w = zipf_weights(n, zipf)
    total = sum(w)
    prior = [x / total for x in w]
    rng.shuffle(prior)
    responds = tuple(tuple(1 if rng.random() < C.DENSITY else 0 for _ in range(n)) for _ in range(m))
    raw = [exact_exp(C.COST_SIGMA * rng.normal()) for _ in range(m)]
    mean = sum(raw) / m
    costs = tuple(x / mean for x in raw)
    return Instance(tuple(prior), responds, costs, se, sp, float(loss), u, genome)


def instance_from_settings(s: Settings) -> Instance:
    se, sp = C.QUALITY_VALUES[s.quality]
    m = effective_m(s.m, s.u, s.genome)
    return make_instance(s.seed, s.n, m, s.zipf, se, sp, s.loss, s.u, (s.g_cost, s.g_yield) if s.genome else None)


def binary_liks(responds, se, sp):
    """Je Test eine Tabelle (2, n): Ausgang 0 negativ, 1 positiv; positiv mit Wahrscheinlichkeit se, wenn die Diagnose den Test anspricht, sonst 1 - sp."""
    out = []
    for row in responds:
        pos = np.array([se if r else 1.0 - sp for r in row])
        out.append(np.array([1.0 - pos, pos]))
    return out


def unclear_liks(responds, se, sp, u):
    """Ausgänge 0 negativ, 1 unklar (Wahrscheinlichkeit u, unabhängig von der Diagnose), 2 positiv."""
    out = []
    for row in responds:
        pos = np.array([se if r else 1.0 - sp for r in row])
        out.append(np.array([(1.0 - u) * (1.0 - pos), np.full(len(row), u), (1.0 - u) * pos]))
    return out


def genome_lik(n: int, y: float):
    """n + 1 Ausgänge: Ausgang d = Diagnose d gefunden (Wahrscheinlichkeit y, wenn sie vorliegt); Ausgang n = kein Befund."""
    a = np.zeros((n + 1, n))
    for d in range(n):
        a[d, d] = y
        a[n, d] = 1.0 - y
    return a


class Problem:
    """Posterior, Ausgangswahrscheinlichkeiten und Verlust bei Stopp für eine Instanz. Zustände sind Tupel (siehe Modulkopf); `cache` gehört dem Aufrufer-Modul."""

    def __init__(self, prior, liks, costs, loss, labels=None):
        self.prior = np.asarray(prior, float)
        self.liks = list(liks)
        self.costs = np.asarray(costs, float)
        self.loss = float(loss)
        self.m = len(self.costs)
        self.labels = list(labels) if labels is not None else [f"T{t + 1}" for t in range(self.m)]
        self.cache = {}

    @classmethod
    def from_instance(cls, inst: Instance, with_genome: bool = True):
        n = len(inst.prior)
        liks = unclear_liks(inst.responds, inst.se, inst.sp, inst.u) if inst.u > 0 else binary_liks(inst.responds, inst.se, inst.sp)
        costs = list(inst.costs)
        labels = [f"T{t + 1}" for t in range(len(costs))]
        if inst.genome is not None and with_genome:
            liks.append(genome_lik(n, inst.genome[1]))
            costs.append(inst.genome[0])
            labels.append("Genom")
        return cls(inst.prior, liks, costs, inst.loss, labels)

    def start(self):
        return tuple([0] * self.m)

    def posterior(self, state):
        w = self.prior.copy()
        for t, o in enumerate(state):
            if o > 0:
                w = w * self.liks[t][o - 1]
        return w

    def outcomes(self, w, t):
        """Mögliche Ausgänge von Test t im Zustand mit Gewichten w: Liste (Ausgang, Wahrscheinlichkeit, neue Gewichte); unmögliche Äste fehlen."""
        parts = [w * self.liks[t][o] for o in range(self.liks[t].shape[0])]
        sums = [float(x.sum()) for x in parts]
        tot = sum(sums)
        return [(o, sm / tot, x) for o, (x, sm) in enumerate(zip(parts, sums)) if sm > 0.0]

    def stop_loss(self, w):
        """Erwarteter Verlust, wenn jetzt die wahrscheinlichste Diagnose genannt wird: L · (1 − größte Posterior-Wahrscheinlichkeit)."""
        return self.loss * (1.0 - w.max() / w.sum())

    @staticmethod
    def entropy(w):
        q = w / w.sum()
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    @staticmethod
    def child(state, t, o):
        s = list(state)
        s[t] = o + 1
        return tuple(s)

    def free_tests(self, state):
        return [t for t in range(self.m) if state[t] == 0]

    def without_test(self, index):
        """Dasselbe Problem ohne Test `index` (für „Genom nie“)."""
        keep = [t for t in range(self.m) if t != index]
        return Problem(self.prior, [self.liks[t] for t in keep], self.costs[keep], self.loss, [self.labels[t] for t in keep])
