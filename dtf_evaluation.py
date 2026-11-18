"""Auswertung einer Instanz (live) und die Verdikte der App als reine Funktionen (Texte und Schwellen, damit tests/test_verdicts.py sie an gesetzten Werten prüft)."""

from dataclasses import dataclass, replace

import dtf_results as R
from dtf_model import Problem, Settings, instance_from_settings
from dtf_rules import RULE_NAMES, RULES, rule_evaluation
from dtf_solver import Evaluation, evaluate, forced_first, optimal_chooser, optimal_first, optimal_value, policy_tree

__all__ = ["Settings", "Analysis", "analyse", "verdict_rules", "verdict_genome", "verdict_ppv", "fmt1", "fmt2"]


def fmt1(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def fmt2(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    m_total: int                   # Zahl der Tests einschließlich Genomtest
    prior: tuple
    tests: tuple                   # je Test: (Bezeichnung, Kosten, Diagnosen, die er anspricht oder „alle“)
    opt: Evaluation
    first_test: str | None         # bester erster Test (Bezeichnung) oder None, wenn Stopp sofort optimal ist
    rules: dict                    # Regel -> Evaluation
    tau: dict                      # Regel -> verwendete Stoppschwelle (G, B)
    rule_cell: str                 # nächstliegende gemessene binäre oder „unklar“-Zelle (für τ und die Meldung zu den Faustregeln)
    rule_diff: tuple               # Parameter, in denen die Einstellung von dieser Zelle abweicht
    tree: dict
    genome: dict | None            # {"first": bool, "forced_gap": %, "never_gap": %} oder None (Genomtest aus)

    def gap(self, rule: str) -> float:
        return 100.0 * (self.rules[rule].total / self.opt.total - 1.0) if self.opt.total > 1e-12 else 0.0

    def best_rule(self):
        return min(RULES, key=self.gap)


def analyse(s: Settings) -> Analysis:
    """Eine Instanz zu den Einstellungen: exaktes Optimum, die drei Faustregeln (τ der nächstliegenden gemessenen Zelle), Baum, Test-Tabelle, Genom-Kennzahlen."""
    inst = instance_from_settings(s)
    pr = Problem.from_instance(inst)
    n = len(inst.prior)
    value, first = optimal_first(pr, pr.start())
    opt = evaluate(pr, optimal_chooser(pr))
    base_cell_id, base_diff = R.nearest_cell(replace(s, genome=False))      # τ stammt aus einer binären oder „unklar“-Zelle
    tau = {rule: R.cell(base_cell_id)["tau"][rule] for rule in ("G", "B")}
    rules = {"M": rule_evaluation(pr, "M"), "G": rule_evaluation(pr, "G", tau["G"]), "B": rule_evaluation(pr, "B", tau["B"])}
    names = [f"D{d + 1}" for d in range(n)]
    tests = []
    for t in range(pr.m):
        if t < len(inst.responds):
            hit = [names[d] for d in range(n) if inst.responds[t][d]]
            tests.append((pr.labels[t], float(pr.costs[t]), ", ".join(hit) if hit else "keine"))
        else:
            tests.append((pr.labels[t], float(pr.costs[t]), "findet die Diagnose (Ausbeute y)"))
    genome = None
    if s.genome:
        g = pr.m - 1
        without = optimal_value(pr.without_test(g))
        genome = {"first": first == g, "forced_gap": 100.0 * (forced_first(pr, g) / value - 1.0), "never_gap": 100.0 * (without / value - 1.0)}
    tree = policy_tree(pr, optimal_chooser(pr), depth=3, min_prob=0.02)
    return Analysis(s, pr.m, inst.prior, tuple(tests), opt, None if first is None else pr.labels[first], rules, tau, base_cell_id, tuple(base_diff), tree, genome)


# --- Verdikte (reine Funktionen) -----------------------------------------------------------------------------------------------------------

def verdict_rules(state: str, rule: str, gap: float, se: float, n_eval: int, exact: bool = True) -> tuple:
    """Meldung zu Kernfrage 1: Zustand 'reicht' | 'merklich' | 'verfehlt', die beste einfache Regel und ihre gemessene Lücke in der Zelle.
    Liegen die Einstellungen nicht genau auf einer gemessenen Zelle (exact False), sagt der Text es."""
    zahl = f"{RULE_NAMES[rule]}: im Mittel {fmt1(gap)} % über dem Optimum (± {fmt1(se)}, {n_eval} Instanzen der gemessenen Zelle)"
    zusatz = "" if exact else " Die gemessene Zelle weicht von Ihren Einstellungen ab (Hinweis unter der Grafik)."
    if state == "reicht":
        return "success", f"**Hier reicht eine einfache Regel.** Die beste, {zahl}.{zusatz}"
    if state == "merklich":
        return "info", f"**Die einfache Regel kostet merklich.** Die beste, {zahl}.{zusatz}"
    return "warning", f"**Die einfache Regel verfehlt das Optimum deutlich.** Die beste, {zahl}.{zusatz}"


def verdict_genome(share_first: float, forced: float, never: float, y: float, c: float, exact: bool) -> tuple:
    """Meldung zu Kernfrage 2 aus der gemessenen Genom-Zelle: Anteil der Instanzen, in denen der Genomtest der erste Test ist, und die Mehrkosten der beiden einfachen Politiken."""
    zusatz = "" if exact else " (nächstliegende gemessene Zelle, siehe Hinweis)"
    kopf = f"Genomtest (Ausbeute {fmt2(y)}, Kosten {fmt1(c)} mittlere Testkosten){zusatz}: erster Test in {share_first * 100:.0f} % der Instanzen; „Genom immer zuerst“ kostet {fmt1(forced)} %, „Genom nie“ {fmt1(never)} % mehr als das Optimum."
    if share_first >= 0.9:
        return "success", "**Hier lohnt der Genomtest als erster.** " + kopf
    if share_first <= 0.1:
        return "info", "**Hier lohnt der Genomtest nicht als erster.** " + kopf
    return "warning", "**Hier kippt die Entscheidung von Instanz zu Instanz.** " + kopf


def verdict_ppv(prevalence: float, k: int, ppv_independent: float, ppv_shared: float) -> tuple:
    """Meldung zu Kernfrage 3: PPV nach k positiven Tests, unabhängig gerechnet gegen mit gemeinsamem Fehler."""
    diff = ppv_independent - ppv_shared
    text = f"Nach {k} positiven Tests: Wahrscheinlichkeit krank zu sein {fmt2(ppv_independent)} bei unabhängigen Tests, {fmt2(ppv_shared)} bei gemeinsamem Fehler."
    if k == 1:
        return "info", "Nach einem positiven Test sind beide Rechnungen gleich. " + text
    if diff < 0.05:
        return "success", f"**Hier macht der gemeinsame Fehler wenig aus** (Unterschied {fmt2(diff)}). " + text
    return "warning", f"**Der gemeinsame Fehler macht einen großen Unterschied** ({fmt2(diff)}). " + text
