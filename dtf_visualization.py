"""Plotly-Figuren: Baum der besten Folge, Lücken der Faustregeln, Regime-Balken, Genom-Kurve, PPV-Kurve, Vorab-Balken.
Alle Achsen fest (fixedrange), Beschriftungen als Annotationen mit Hintergrund, kein Text-Trace (DEMO-PLAYBOOK §3)."""

import plotly.graph_objects as go

import dtf_constants as C
from dtf_rules import RULE_NAMES, RULES

_RULE_COLORS = {"M": C.COLOR_M, "G": C.COLOR_G, "B": C.COLOR_B}
_BG = "rgba(255,255,255,0.88)"


def _fixed(fig, height, xtitle=None, ytitle=None, legend_bottom=False):
    fig.update_xaxes(fixedrange=True, title_text=xtitle)
    fig.update_yaxes(fixedrange=True, title_text=ytitle)
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=30, b=70 if legend_bottom else 40), dragmode=False)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    return fig


def outcome_label(test_label: str, o: int, unclear: bool, n_diag: int) -> str:
    if test_label == "Genom":
        return f"D{o + 1} gefunden" if o < n_diag else "kein Befund"
    names = ("negativ", "unklar", "positiv") if unclear else ("negativ", "positiv")
    return names[o]


def outcome_short(test_label: str, o: int, unclear: bool, n_diag: int) -> str:
    """Kurzform für die Kanten des Baums: – negativ, + positiv, ? unklar; beim Genomtest die gefundene Diagnose oder „kein“."""
    if test_label == "Genom":
        return f"D{o + 1}" if o < n_diag else "kein"
    return ("–", "?", "+")[o] if unclear else ("–", "+")[o]


def build_prior(prior):
    """Vorab-Wahrscheinlichkeit je Diagnose."""
    fig = go.Figure(go.Bar(x=[f"D{d + 1}" for d in range(len(prior))], y=[100 * p for p in prior], marker_color=C.COLOR_GOOD))
    return _fixed(fig, 220, None, "Vorab in %")


def _layout_tree(node, depth, counter, out):
    """Setzt für jeden Knoten x = Tiefe und y = Mitte seiner Blätter; sammelt (Knoten, x, y, Eltern-Index, Kantenbeschriftung)."""
    idx = len(out)
    out.append({"node": node, "x": depth, "y": None, "parent": None, "edge": ""})
    ys = []
    for o, p, child in node["children"]:
        cidx = len(out)
        _layout_tree(child, depth + 1, counter, out)
        out[cidx]["parent"] = idx
        out[cidx]["edge"] = (o, p)
        ys.append(out[cidx]["y"])
    if node["cut"] > 0:
        cut_idx = len(out)
        out.append({"node": {"cut_leaf": node["cut"]}, "x": depth + 1, "y": float(counter[0]), "parent": idx, "edge": None})
        counter[0] += 1
        ys.append(out[cut_idx]["y"])
    if not ys:
        out[idx]["y"] = float(counter[0])
        counter[0] += 1
    else:
        out[idx]["y"] = sum(ys) / len(ys)
    return idx


def build_tree(tree, unclear: bool, n_diag: int):
    """Baum der besten Folge: Knoten = Test oder Stopp (mit der genannten Diagnose), Kanten = Ausgang mit Wahrscheinlichkeit; gekürzte Äste als „…“."""
    out, counter = [], [0]
    _layout_tree(tree, 0, counter, out)
    max_x = max(e["x"] for e in out)
    fig = go.Figure()
    for e in out:
        if e["parent"] is None:
            continue
        par = out[e["parent"]]
        fig.add_trace(go.Scatter(x=[par["x"], e["x"]], y=[par["y"], e["y"]], mode="lines", line=dict(color="rgba(120,120,120,0.7)", width=1.5), hoverinfo="skip", showlegend=False))
        if e["edge"] is not None:
            o, p = e["edge"]
            lab = outcome_short(par["node"]["label"], o, unclear, n_diag)
            fig.add_annotation(x=par["x"] + 0.33 * (e["x"] - par["x"]), y=par["y"] + 0.33 * (e["y"] - par["y"]), text=f"{lab} {100 * p:.0f} %", showarrow=False, bgcolor=_BG, font=dict(size=10, color="#1c2430"))
    for e in out:
        nd = e["node"]
        if "cut_leaf" in nd:
            fig.add_annotation(x=e["x"], y=e["y"], text=f"… ({100 * nd['cut_leaf']:.0f} % der Fälle gekürzt)", showarrow=False, bgcolor=_BG, font=dict(size=10, color="#555"))
            continue
        if nd["test"] is None:
            text = f"Stopp → D{nd['diagnosis'] + 1}<br>({100 * nd['p_best']:.0f} % sicher)"
            color = "#d9ead3"
        else:
            text = f"{nd['label']}"
            color = "#dbe7fb"
        fig.add_annotation(x=e["x"], y=e["y"], text=text, showarrow=False, bgcolor=color, bordercolor="#667", borderwidth=1, borderpad=3, font=dict(size=11, color="#1c2430"))
    fig.update_xaxes(range=[-0.5, max_x + 1.2], visible=False)
    fig.update_yaxes(range=[counter[0] - 0.3, -0.7], visible=False)
    return _fixed(fig, max(260, 60 * counter[0] + 60))


def build_gap_bars(cell_result: dict, title_cell: str):
    """Lücke der drei Faustregeln in einer gemessenen Zelle (Mittel, Fehlerbalken = Standardfehler) und der Anteil der Instanzen mit Lücke über 5 %."""
    fig = go.Figure()
    for rule in RULES:
        r = cell_result[rule]
        fig.add_trace(go.Bar(name=RULE_NAMES[rule], x=[RULE_NAMES[rule]], y=[r["mean"]], error_y=dict(type="data", array=[r["se"]], visible=True), marker_color=_RULE_COLORS[rule],
                             text=[f"{r['mean']:.1f} %".replace(".", ",")], textposition="inside", insidetextanchor="middle", showlegend=False))
    fig.update_layout(title=dict(text=title_cell, font=dict(size=13)))
    return _fixed(fig, 320, None, "Lücke über dem Optimum in %")


def build_regime(results: dict, cell_ids, labels):
    """Gruppierte Balken: Lücke der drei Regeln über mehrere gemessene Zellen."""
    fig = go.Figure()
    for rule in RULES:
        fig.add_trace(go.Bar(name=RULE_NAMES[rule], x=list(labels), y=[results["cells"][c][rule]["mean"] for c in cell_ids],
                             error_y=dict(type="data", array=[results["cells"][c][rule]["se"] for c in cell_ids], visible=True), marker_color=_RULE_COLORS[rule]))
    fig.update_layout(barmode="group")
    fig = _fixed(fig, 420, None, "Lücke über dem Optimum in %", legend_bottom=True)
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(margin=dict(l=50, r=20, t=30, b=140), legend=dict(orientation="h", yanchor="top", y=-0.38, x=0))
    return fig


def build_genome_curve(results: dict, y: float):
    """Anteil der Instanzen, in denen der Genomtest der erste Test ist, über die Genomkosten (gemessene Zellen) für eine Ausbeute."""
    costs = C.GENOME_COSTS_MEASURED
    ids = [f"y{int(y * 100)}c{int(c)}" for c in costs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(costs), y=[100 * results["cells"][i]["share_first"] for i in ids], mode="lines+markers", name="Genom ist erster Test (% der Instanzen)",
                             line=dict(color=C.COLOR_GOOD, width=3)))
    fig.update_xaxes(type="log", tickvals=list(costs), ticktext=[str(int(c)) for c in costs])
    return _fixed(fig, 300, "Kosten des Genomtests in mittleren Testkosten", "Anteil in %", legend_bottom=True)


def build_genome_excess(results: dict, y: float):
    """Mehrkosten der beiden einfachen Politiken „Genom immer zuerst“ und „Genom nie“ über die Genomkosten."""
    costs = C.GENOME_COSTS_MEASURED
    ids = [f"y{int(y * 100)}c{int(c)}" for c in costs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(costs), y=[results["cells"][i]["forced"]["mean"] for i in ids], mode="lines+markers", name="Genom immer zuerst", line=dict(color=C.COLOR_G, width=3)))
    fig.add_trace(go.Scatter(x=list(costs), y=[results["cells"][i]["never"]["mean"] for i in ids], mode="lines+markers", name="Genom nie", line=dict(color=C.COLOR_M, width=3)))
    fig.update_xaxes(type="log", tickvals=list(costs), ticktext=[str(int(c)) for c in costs])
    return _fixed(fig, 300, "Kosten des Genomtests in mittleren Testkosten", "Mehrkosten über dem Optimum in %", legend_bottom=True)


def build_ppv(ks, independent, shared):
    """Wahrscheinlichkeit, krank zu sein, nach k positiven Tests: unabhängige Tests gegen gemeinsamen Fehler."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ks), y=[100 * v for v in independent], mode="lines+markers", name="unabhängig gerechnet", line=dict(color=C.COLOR_G, width=3)))
    fig.add_trace(go.Scatter(x=list(ks), y=[100 * v for v in shared], mode="lines+markers", name="mit gemeinsamem Fehler", line=dict(color=C.COLOR_M, width=3)))
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 105])
    return _fixed(fig, 320, "Zahl positiver Tests", "Wahrscheinlichkeit krank zu sein in %", legend_bottom=True)
