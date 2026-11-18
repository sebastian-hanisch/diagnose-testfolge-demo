"""PDF-Auswertung einer Instanz (fpdf2, Core-Schriften: nur Latin-1, deshalb werden Sonderzeichen ersetzt; DEMO-PLAYBOOK §7)."""

from fpdf import FPDF

from dtf_rules import RULE_NAMES, RULES

_REPLACE = {"–": "-", "—": "-", "−": "-", "→": "->", "≥": ">=", "≤": "<=", "τ": "tau", "„": '"', "“": '"', "”": '"', "’": "'", "·": "*", "…": "...", "×": "x",
            "Σ": "Summe", "γ": "gamma", "€": "EUR", "✓": "ja", "≈": "ca.", "σ": "sigma"}


def pdf_text(text: str) -> str:
    """Text auf Latin-1 bringen: bekannte Sonderzeichen ersetzen, alles Übrige als „?“."""
    for k, v in _REPLACE.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _tree_lines(node, unclear, n_diag, outcome_label, indent=0):
    pad = "    " * indent
    if node["test"] is None:
        return [f"{pad}Stopp -> D{node['diagnosis'] + 1} ({100 * node['p_best']:.0f} % sicher)"]
    lines = [f"{pad}{node['label']}"]
    for o, p, child in node["children"]:
        lines.append(f"{pad}  {outcome_label(node['label'], o, unclear, n_diag)} ({100 * p:.0f} %):")
        lines += _tree_lines(child, unclear, n_diag, outcome_label, indent + 1)
    if node["cut"] > 0:
        lines.append(f"{pad}  ... ({100 * node['cut']:.0f} % der Fälle gekürzt)")
    return lines


def build_pdf(a, settings_text: str) -> bytes:
    """Auswertung der Analyse `a` (dtf_evaluation.Analysis): Einstellungen, Kennzahlen, Faustregeln, Test-Tabelle und der Baum der besten Folge."""
    from dtf_visualization import outcome_label

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def line(text, size=10, bold=False, h=5.5):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.multi_cell(0, h, pdf_text(text), new_x="LMARGIN", new_y="NEXT")

    line("Diagnose-Testfolge: Auswertung einer Instanz", 15, True, 8)
    line("Lehrmodell mit erfundenen Daten, keine medizinische Beratung.", 9)
    pdf.ln(2)
    line("Einstellungen", 11, True)
    line(settings_text)
    pdf.ln(2)
    line("Kennzahlen (eine Instanz, kein Beleg)", 11, True)
    line(f"Erwartete Gesamtkosten des Optimums: {a.opt.total:.2f} (Testkosten {a.opt.test_cost:.2f}, Irrtumsverlust {a.opt.loss:.2f}; Wahrscheinlichkeit des Irrtums {100 * a.opt.p_error:.1f} %)")
    line(f"Erwartete Zahl durchgeführter Tests: {a.opt.n_tests:.2f}; bester erster Test: {a.first_test or 'keiner (sofort Stopp)'}")
    pdf.ln(2)
    line("Faustregeln gegen das Optimum", 11, True)
    for rule in RULES:
        tau = f", Stoppschwelle {a.tau[rule]}" if rule in a.tau else ""
        line(f"{RULE_NAMES[rule]}: Gesamtkosten {a.rules[rule].total:.2f}, Lücke {a.gap(rule):.1f} % über dem Optimum{tau}")
    pdf.ln(2)
    line("Tests", 11, True)
    for label, cost, hits in a.tests:
        line(f"{label}: Kosten {cost:.2f}; spricht an: {hits}")
    pdf.ln(2)
    line("Baum der besten Folge (nach Wahrscheinlichkeit gekürzt)", 11, True)
    pdf.set_font("Courier", "", 8)
    for t in _tree_lines(a.tree, a.settings.u > 0, len(a.prior), outcome_label):
        pdf.multi_cell(0, 4, pdf_text(t), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
