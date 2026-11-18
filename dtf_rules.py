"""Faustregeln für die Testfolge als Strategien (Zustand → Test oder None = Stopp); ausgewertet wird exakt mit dtf_solver.evaluate.

kurzsichtig (M): noch ein Test, wenn der erwartete Verlust nach EINEM Test plus dessen Kosten unter dem jetzigen liegt (wählt den besten solchen Test).
Info/Kosten (G): der Test mit dem größten Entropiegewinn je Kosten; Stopp, sobald die beste Diagnose mindestens τ Wahrscheinlichkeit hat oder kein Test Information bringt.
billigster nützlicher Test (B): der billigste Test mit Informationsgewinn; derselbe Stopp. τ ist keine Kostengröße und wird je Szenario abgestimmt (dtf_results)."""

from dtf_model import Problem
from dtf_solver import Evaluation, evaluate

RULES = ("M", "G", "B")
RULE_NAMES = {"M": "kurzsichtig (ein Test voraus)", "G": "Info/Kosten", "B": "billigster nützlicher Test"}
EPS_GAIN = 1e-9   # Informationsgewinn darunter gilt als „keine Information“


def info_gain(pr: Problem, w, t, h0):
    """Entropie der Diagnosen jetzt minus erwartete Entropie nach Test t."""
    return h0 - sum(p * pr.entropy(x) for _, p, x in pr.outcomes(w, t))


def myopic_chooser(pr: Problem):
    cache = {}

    def choose(state):
        if state in cache:
            return cache[state]
        w = pr.posterior(state)
        best_t, best_v = None, pr.stop_loss(w)
        for t in pr.free_tests(state):
            v = pr.costs[t] + sum(p * pr.stop_loss(x) for _, p, x in pr.outcomes(w, t))
            if v < best_v:
                best_t, best_v = t, v
        cache[state] = best_t
        return best_t
    return choose


def _useful(pr: Problem, state, w):
    h0 = pr.entropy(w)
    gains = {t: info_gain(pr, w, t, h0) for t in pr.free_tests(state)}
    return {t: g for t, g in gains.items() if g > EPS_GAIN}


def info_chooser(pr: Problem, tau: float):
    cache = {}

    def choose(state):
        if state in cache:
            return cache[state]
        w = pr.posterior(state)
        t = None
        if w.max() / w.sum() < tau and pr.free_tests(state):
            useful = _useful(pr, state, w)
            if useful:
                t = max(useful, key=lambda x: useful[x] / pr.costs[x])
        cache[state] = t
        return t
    return choose


def cheapest_chooser(pr: Problem, tau: float):
    cache = {}

    def choose(state):
        if state in cache:
            return cache[state]
        w = pr.posterior(state)
        t = None
        if w.max() / w.sum() < tau and pr.free_tests(state):
            useful = _useful(pr, state, w)
            if useful:
                t = min(useful, key=lambda x: pr.costs[x])
        cache[state] = t
        return t
    return choose


def chooser_for(pr: Problem, rule: str, tau: float = 0.95):
    if rule == "M":
        return myopic_chooser(pr)
    if rule == "G":
        return info_chooser(pr, tau)
    if rule == "B":
        return cheapest_chooser(pr, tau)
    raise ValueError(rule)


def rule_evaluation(pr: Problem, rule: str, tau: float = 0.95) -> Evaluation:
    return evaluate(pr, chooser_for(pr, rule, tau))
