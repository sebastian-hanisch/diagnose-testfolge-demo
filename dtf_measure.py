"""Messreihe: Messung einer Instanz je Zelle und Aggregation je Zelle (von tools/sweep.py getrieben, von den Tests teilweise neu gerechnet).

Binäre Zellen und „unklar“: Optimum, drei Faustregeln (G und B für jede Stoppschwelle τ aus C.TAUS); τ wird auf den ersten C.N_TUNE Instanzen abgestimmt und die
Lücke auf den übrigen gemessen (getrennte Instanzen, sonst optimistisch). Genom-Zellen: Optimum mit Genom, ohne Genom, Genom als erster Test, „immer zuerst“.
Lücke = 100 · (Wert der Regel / Optimum − 1), Zähleinheit: Instanz (je Instanz ein Zufalls-Seed)."""

import math

import dtf_constants as C
from dtf_model import Problem, make_instance
from dtf_rules import RULES, rule_evaluation
from dtf_solver import evaluate, forced_first, optimal_chooser, optimal_first, optimal_value


def cell_index(cell_id: str) -> int:
    return next(i for i, c in enumerate(C.CELLS) if c["id"] == cell_id)


def instance_seed(cell: dict, i: int) -> int:
    return C.SWEEP_SEED_BASE + 1000 * cell_index(cell["id"]) + i


def cell_problem(cell: dict, i: int) -> Problem:
    genome = (cell["g_cost"], cell["g_yield"]) if cell["kind"] == "genom" else None
    inst = make_instance(instance_seed(cell, i), cell["n"], cell["m"], cell["zipf"], cell["se"], cell["sp"], cell["loss"], cell["u"], genome)
    return Problem.from_instance(inst)


def gap(value: float, opt: float) -> float:
    return 100.0 * (value / opt - 1.0) if opt > 1e-12 else 0.0


def measure_rules_instance(cell: dict, i: int) -> dict:
    """Eine Instanz einer binären oder „unklar“-Zelle: Optimum (mit Zerlegung) und die Lücken aller Regeln, G und B für jedes τ."""
    pr = cell_problem(cell, i)
    opt = optimal_value(pr)
    ev_opt = evaluate(pr, optimal_chooser(pr))
    out = {"opt": opt, "opt_tests": ev_opt.n_tests, "opt_perr": ev_opt.p_error, "M": gap(rule_evaluation(pr, "M").total, opt), "G": {}, "B": {}}
    for tau in C.TAUS:
        out["G"][tau] = gap(rule_evaluation(pr, "G", tau).total, opt)
        out["B"][tau] = gap(rule_evaluation(pr, "B", tau).total, opt)
    return out


def measure_genome_instance(cell: dict, i: int) -> dict:
    """Eine Instanz einer Genom-Zelle: Optimum mit und ohne Genomtest, ist Genom der erste Test, Mehrkosten von „Genom immer zuerst“ und „Genom nie“."""
    pr = cell_problem(cell, i)
    g = pr.m - 1
    v, first = optimal_first(pr, pr.start())
    without = optimal_value(pr.without_test(g))
    return {"opt": v, "opt_without": without, "first_genome": first == g, "forced": gap(forced_first(pr, g), v), "never": gap(without, v)}


def mean(x):
    return sum(x) / len(x)


def se(x):
    n = len(x)
    if n < 2:
        return float("nan")
    mu = mean(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / (n - 1)) / math.sqrt(n)


def aggregate_rules(rows: list) -> dict:
    """rows: Ergebnis von measure_rules_instance je Instanz (Reihenfolge = Instanzen 0 bis N-1). Schwelle auf den ersten N_TUNE abstimmen, Lücke auf den übrigen messen."""
    tune, test = rows[: C.N_TUNE], rows[C.N_TUNE:]
    res = {"n_instances": len(rows), "n_tune": C.N_TUNE, "opt_mean": mean([r["opt"] for r in test]), "opt_mean_all": mean([r["opt"] for r in rows]),
           "opt_tests_mean": mean([r["opt_tests"] for r in test]), "opt_perr_mean": mean([r["opt_perr"] for r in test])}
    tau = {}
    for rule in ("G", "B"):
        tau[rule] = min(C.TAUS, key=lambda t: mean([r[rule][t] for r in tune]))
    series = {"M": [r["M"] for r in test], "G": [r["G"][tau["G"]] for r in test], "B": [r["B"][tau["B"]] for r in test]}
    for rule in RULES:
        res[rule] = {"mean": mean(series[rule]), "se": se(series[rule]), "share_over": sum(1 for g in series[rule] if g > C.GAP_MARK) / len(series[rule])}
    diff = [a - b for a, b in zip(series["M"], series["G"])]
    res["tau"] = tau
    # mittlere Lücke auf der Messhälfte für JEDE Schwelle (zeigt, was eine feste Standardschwelle gekostet hätte)
    res["tau_curve"] = {rule: {str(t): mean([r[rule][t] for r in test]) for t in C.TAUS} for rule in ("G", "B")}
    res["diff_MG"] = {"mean": mean(diff), "se": se(diff)}
    res["per_instance"] = {"opt": [r["opt"] for r in rows], "M": [r["M"] for r in rows], "G": [r["G"][tau["G"]] for r in rows], "B": [r["B"][tau["B"]] for r in rows]}
    return res


def aggregate_genome(rows: list) -> dict:
    test = rows
    return {"n_instances": len(rows), "opt_mean": mean([r["opt"] for r in test]), "opt_without_mean": mean([r["opt_without"] for r in test]),
            "share_first": sum(1 for r in test if r["first_genome"]) / len(test),
            "forced": {"mean": mean([r["forced"] for r in test]), "se": se([r["forced"] for r in test])},
            "never": {"mean": mean([r["never"] for r in test]), "se": se([r["never"] for r in test])},
            "per_instance": {"opt": [r["opt"] for r in rows], "first_genome": [r["first_genome"] for r in rows]}}


def measure_cell(cell: dict, n_instances: int | None = None) -> dict:
    """Ganze Zelle: Messung aller Instanzen und Aggregation. (n_instances nur für Tests mit kleiner Menge.)"""
    n = C.N_INSTANCES if n_instances is None else n_instances
    if cell["kind"] == "genom":
        return aggregate_genome([measure_genome_instance(cell, i) for i in range(n)])
    return aggregate_rules([measure_rules_instance(cell, i) for i in range(n)])
