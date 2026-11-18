"""Tabellen und Textbausteine der Oberfläche als reine Funktionen (Zeilen als Wörterbücher, Zahlen deutsch formatiert): testbar ohne Streamlit."""

from dataclasses import replace

import dtf_basisrate as BR
import dtf_constants as C
import dtf_results as R
from dtf_evaluation import fmt1, fmt2
from dtf_model import Settings
from dtf_rules import RULE_NAMES, RULES


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f} %".replace(".", ",")


def settings_text(s: Settings) -> str:
    """Einstellungen in einem Satz (PDF und Hinweise)."""
    teile = [f"{s.m} Tests (gerechnet: siehe Hinweis in der App), {s.n} Diagnosen, Vorab {C.ZIPF_LABELS[s.zipf]}", f"Testgüte (Sensitivität / Spezifität) {s.quality}",
             f"Irrtumspreis {s.loss} mittlere Testkosten", f"Anteil unklar {fmt2(s.u)}"]
    if s.genome:
        teile.append(f"Genomtest: Kosten {fmt1(s.g_cost)}, Ausbeute {fmt2(s.g_yield)}")
    teile.append(f"Zufalls-Seed {s.seed}")
    return "; ".join(teile)


def tests_rows(a) -> list:
    return [{"Test": label, "Kosten": fmt2(cost), "spricht an": hits} for label, cost, hits in a.tests]


def rules_rows(a) -> list:
    rows = [{"Strategie": "Optimum (exakt)", "Gesamtkosten": fmt2(a.opt.total), "Testkosten": fmt2(a.opt.test_cost), "Irrtumsverlust": fmt2(a.opt.loss),
             "Tests im Mittel": fmt2(a.opt.n_tests), "Lücke": "–", "Stoppschwelle": "–"}]
    for rule in RULES:
        ev = a.rules[rule]
        rows.append({"Strategie": RULE_NAMES[rule], "Gesamtkosten": fmt2(ev.total), "Testkosten": fmt2(ev.test_cost), "Irrtumsverlust": fmt2(ev.loss),
                     "Tests im Mittel": fmt2(ev.n_tests), "Lücke": pct(a.gap(rule)), "Stoppschwelle": fmt2(a.tau[rule]) if rule in a.tau else "–"})
    return rows


def cell_label(cell: dict) -> str:
    """Kurzbeschreibung einer gemessenen Zelle für Überschriften und Hinweise."""
    p = cell
    text = f"{p['n']} Diagnosen, {p['m']} Tests, Vorab {C.ZIPF_LABELS[p['zipf']]}, Testgüte {str(p['se']).replace('.', ',')} / {str(p['sp']).replace('.', ',')}, Irrtumspreis {p['loss']}"
    if p.get("u", 0):
        text += f", unklar {str(p['u']).replace('.', ',')}"
    if p.get("g_cost") is not None:
        text += f", Genomtest Kosten {fmt1(p['g_cost'])}, Ausbeute {fmt2(p['g_yield'])}"
    return text


def diff_note(diff) -> str:
    """Hinweis, wenn die Einstellung von der gemessenen Zelle abweicht; leer, wenn sie genau auf einer Zelle liegt."""
    if not diff:
        return ""
    return "Die gemessene Zelle weicht von Ihren Einstellungen ab in: " + ", ".join(R.PARAM_NAMES[k] for k in diff) + "."


def messreihe_rows(results: dict, cell_ids) -> list:
    rows = []
    for cid in cell_ids:
        c = results["cells"][cid]
        p = c["params"]
        row = {"Zelle": cell_label(p), "Optimum": fmt2(c["opt_mean"])}
        for rule in RULES:
            r = c[rule]
            row[RULE_NAMES[rule]] = f"{fmt1(r['mean'])} ± {fmt1(r['se'])} %"
        row["Sieger"] = RULE_NAMES[R.best_rule(c)[0]]
        rows.append(row)
    return rows


def unclear_rows(results: dict) -> list:
    rows = []
    base = results["cells"]["u0"]["opt_mean_all"]
    for cid in ("u0", "u1", "u2"):
        c = results["cells"][cid]
        rows.append({"Anteil unklar": fmt2(c["params"]["u"]), "Optimum (alle Instanzen)": fmt2(c["opt_mean_all"]), "Zuwachs": pct(100 * (c["opt_mean_all"] / base - 1.0)),
                     "kurzsichtig": f"{fmt1(c['M']['mean'])} %", "Info/Kosten": f"{fmt1(c['G']['mean'])} %", "billigster nützlicher Test": f"{fmt1(c['B']['mean'])} %"})
    return rows


def genome_rows(results: dict, y: float) -> list:
    rows = []
    for cost in C.GENOME_COSTS_MEASURED:
        c = results["cells"][f"y{int(y * 100)}c{int(cost)}"]
        rows.append({"Genomkosten": fmt1(cost), "Genom ist erster Test": f"{c['share_first'] * 100:.0f} %", "„Genom immer zuerst“ kostet mehr": f"{fmt1(c['forced']['mean'])} %",
                     "„Genom nie“ kostet mehr": f"{fmt1(c['never']['mean'])} %", "Optimum ohne Genom": fmt2(c["opt_without_mean"]), "Optimum mit Genom": fmt2(c["opt_mean"])})
    return rows


def referral_rows(se: float = 0.99, fpr: float = 0.01) -> list:
    rows = []
    for name, prev in BR.REFERRALS:
        rows.append({"Wo getestet wird": name, "Anteil Kranke": BR.REFERRAL_LABELS[prev], "Wahrscheinlichkeit krank bei einem positiven Test": fmt2(BR.ppv(prev, se, fpr, 1))})
    return rows


def tests_until_rows(prevalence: float, se: float, fpr: float, target: float = 0.9) -> list:
    out = []
    for label, shared in (("unabhängig gerechnet", False), ("mit gemeinsamem Fehler", True)):
        k = BR.tests_until(prevalence, se, fpr, target, shared)
        out.append({"Annahme": label, f"positive Tests bis Wahrscheinlichkeit ≥ {fmt2(target)}": "mehr als 12" if k is None else str(k)})
    return out


def genome_cell_for(s: Settings):
    """(Zell-Id, abweichende Parameter) der gemessenen Genom-Zelle zu den Genom-Reglern (unabhängig davon, ob der Genomtest eingeschaltet ist)."""
    return R.nearest_cell(replace(s, genome=True))


# --- Tabellen der README (aus der Messreihe erzeugt; tests/test_claims.py prüft, dass die README sie wörtlich enthält) ----------------------------

def md_table(head, rows) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _short(p: dict) -> str:
    return f"{p['n']} Diagnosen, {p['m']} Tests, Vorab {C.ZIPF_LABELS[p['zipf']]}, Se/Sp {str(p['se']).replace('.', ',')}/{str(p['sp']).replace('.', ',')}, L {p['loss']}"


def readme_tables(res: dict) -> dict:
    """Name -> Markdown-Tabelle für die README, ausschließlich aus data/dtf_results.json."""
    out = {}
    rows = []
    for cid in [f"q{i}" for i in range(7)] + [f"g{i}" for i in range(4)]:
        c = res["cells"][cid]
        rows.append([_short(c["params"]), fmt2(c["opt_mean"])] + [f"{fmt1(c[r]['mean'])} ± {fmt1(c[r]['se'])}" for r in RULES]
                    + [f"{c['diff_MG']['mean']:+.1f} ± {fmt1(c['diff_MG']['se'])}".replace(".", ","), f"{fmt2(c['tau']['G'])} / {fmt2(c['tau']['B'])}"])
    out["regeln"] = md_table(["Zelle", "Optimum", "kurzsichtig (%)", "Info/Kosten (%)", "billigster nützlicher Test (%)", "kurzsichtig minus Info/Kosten", "τ (Info/Kosten / billigster)"], rows)
    rows = []
    base = res["cells"]["u0"]["opt_mean_all"]
    for cid in ("u0", "u1", "u2"):
        c = res["cells"][cid]
        rows.append([fmt2(c["params"]["u"]), fmt2(c["opt_mean_all"]), pct(100 * (c["opt_mean_all"] / base - 1.0))] + [f"{fmt1(c[r]['mean'])} ± {fmt1(c[r]['se'])}" for r in RULES])
    out["unklar"] = md_table(["Anteil unklar", "Optimum (alle Instanzen)", "Zuwachs", "kurzsichtig (%)", "Info/Kosten (%)", "billigster nützlicher Test (%)"], rows)
    rows = []
    for y in C.GENOME_YIELD_OPTIONS:
        for cost in C.GENOME_COSTS_MEASURED:
            c = res["cells"][f"y{int(y * 100)}c{int(cost)}"]
            rows.append([fmt2(y), fmt1(cost), f"{c['share_first'] * 100:.0f} %", f"{fmt1(c['forced']['mean'])} ± {fmt1(c['forced']['se'])}", f"{fmt1(c['never']['mean'])} ± {fmt1(c['never']['se'])}",
                         fmt2(c["opt_without_mean"]), fmt2(c["opt_mean"])])
    out["genom"] = md_table(["Ausbeute y", "Genomkosten", "Genom ist erster Test", "„immer zuerst“ (% mehr)", "„nie“ (% mehr)", "Optimum ohne Genom", "Optimum mit Genom"], rows)
    rows = []
    for prev in BR.PREVALENCE_OPTIONS:
        rows.append([BR.PREVALENCE_LABELS[prev]] + [f"{BR.ppv(prev, 0.99, 0.01, 1):.3f}".replace(".", ",")]
                    + [f"{BR.ppv(prev, 0.99, 0.01, k):.3f} / {BR.ppv(prev, 0.99, 0.01, k, True):.3f}".replace(".", ",") for k in (2, 3)])
    out["ppv"] = md_table(["Anteil Kranke", "1 positiver Test", "2 positive Tests (unabhängig / gemeinsamer Fehler)", "3 positive Tests (unabhängig / gemeinsamer Fehler)"], rows)
    return out
