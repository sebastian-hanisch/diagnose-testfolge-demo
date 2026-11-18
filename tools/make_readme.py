"""Erzeugt README.md aus der Vorlage unten und data/dtf_results.json: jede Zahl der README kommt aus der Messreihe, nicht aus dem Gedächtnis.

    python tools/make_readme.py            # schreibt README.md
    python tools/make_readme.py --check    # prüft, dass README.md dem Ergebnis entspricht (tests/test_claims.py ruft render() ebenfalls auf)

Die Testzahl wird per `pytest --collect-only` gezählt und der Vorlage übergeben. Eine Änderung des Textes gehört in diese Vorlage, nicht in die README."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dtf_basisrate as BR  # noqa: E402
import dtf_constants as C  # noqa: E402
import dtf_results as R  # noqa: E402
import dtf_ui_panel as UI  # noqa: E402
from dtf_evaluation import fmt1, fmt2  # noqa: E402
from report import VORMESSUNG_GENOM, VORMESSUNG_REGELN  # noqa: E402

BIN_IDS = [f"q{i}" for i in range(7)] + [f"g{i}" for i in range(4)]
ALL_IDS = BIN_IDS + ["u0", "u1", "u2"]  # alle Zellen mit Regelvergleich (die Genom-Zellen haben keinen)


def paired(cell, a, b, n_tune):
    """Gepaarter Unterschied der Lücken (Regel a minus Regel b, in Punkten) auf den Instanzen nach der Abstimmung: (Mittel, Standardfehler)."""
    per = cell["per_instance"]  # je Instanz: Kosten des Optimums und die Lücke jeder Regel in % (nicht ihre Kosten)
    diff = np.array(per[a], float)[n_tune:] - np.array(per[b], float)[n_tune:]
    return float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff)))


def sign_counts(cells, ids, a, b, n_tune):
    """Zählung über die Zellen: a liegt mehr als zwei Standardfehler über b / darunter / offen (gepaart)."""
    out = {"over": 0, "under": 0, "open": 0}
    for cid in ids:
        m, s = paired(cells[cid], a, b, n_tune)
        out["over" if m > 2 * s else "under" if m < -2 * s else "open"] += 1
    return out


def stable_interval(values, thr):
    """Die Schwelle `thr` teilt die Werte in „unter thr“ und „ab thr“. Jede Schwelle t mit lo <= t <= hi (eine Nachkommastelle) gibt dieselbe Einteilung:
    t muss über dem größten Wert unter thr und höchstens beim kleinsten Wert ab thr liegen."""
    below = max(v for v in values if v < thr)
    above = min(v for v in values if v >= thr)
    return (math.floor(below * 10 + 1e-9) + 1) / 10, math.floor(above * 10 + 1e-9) / 10
FOOTER = ("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning "
          "([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).")


def count_tests() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out) or re.search(r"^(\d+) tests", out, re.M)
    return int(m.group(1))


def _pm(c, rule):
    return f"{fmt1(c[rule]['mean'])} ± {fmt1(c[rule]['se'])}"


def _diff(c):
    d = c["diff_MG"]
    return f"{d['mean']:+.1f} ± {fmt1(d['se'])}".replace(".", ",").replace("-", "−")  # typografisches Minus wie im übrigen Text


def unclear_increase_se(cells):
    """Standardfehler des Zuwachses m_c/m_0 − 1 (in Prozentpunkten) für ungepaarte Mittel, Delta-Methode:
    SE = (m_c/m_0) · sqrt((se_c/m_c)² + (se_0/m_0)²). Quelle: das Optimum je Instanz (alle Instanzen der Zelle)."""
    def stats(cid):
        o = np.array(cells[cid]["per_instance"]["opt"], float)
        return float(o.mean()), float(o.std(ddof=1) / math.sqrt(len(o)))
    m0, s0 = stats("u0")
    out = []
    for cid in ("u1", "u2"):
        mc, sc = stats(cid)
        out.append(100.0 * (mc / m0) * math.sqrt((sc / mc) ** 2 + (s0 / m0) ** 2))
    return out


def open_ids(cells, ids, a, b, n_tune):
    """Zellen, in denen der gepaarte Unterschied der Lücken a − b höchstens zwei Standardfehler beträgt."""
    out = []
    for cid in ids:
        m, s = paired(cells[cid], a, b, n_tune)
        if abs(m) <= 2 * s:
            out.append(cid)
    return out


def cell_label(params):
    """Kurzbeschreibung der Besonderheit einer Regelzelle gegenüber dem Standard."""
    if params["se"] == 1.0:
        return "rauschfrei"
    if params["zipf"] == 0.0:
        return "gleichverteiltes Vorab"
    if params["zipf"] == 1.5:
        return "Vorab sehr schief"
    return f"Se/Sp {fmt1(params['se'])}/{fmt1(params['sp'])}, L {params['loss']}, {params['n']} Diagnosen".replace(".", ",")


def _z(n):
    """„1 Zelle“ / „3 Zellen“ (kein Klammer-Plural)."""
    return f"{n} Zelle" if n == 1 else f"{n} Zellen"


def derived(res):
    """Alle Zahlen, die in der Prosa stehen, aus der Messreihe abgeleitet (Zählungen über ALLE 11 binären Zellen)."""
    cells = res["cells"]
    best = {cid: R.best_rule(cells[cid]) for cid in BIN_IDS}
    state = {cid: R.verdict_state(best[cid][1]) for cid in BIN_IDS}
    sign = {}
    for cid in BIN_IDS:
        d = cells[cid]["diff_MG"]
        sign[cid] = "G" if d["mean"] > 2 * d["se"] else ("M" if d["mean"] < -2 * d["se"] else "offen")
    q2 = cells["q2"]
    return {
        "best": best, "state": state, "sign": sign,
        "n_under5": sum(1 for cid in BIN_IDS if best[cid][1] < 5.0), "under5_ids": [cid for cid in BIN_IDS if best[cid][1] < 5.0],
        "states": {s: sum(1 for cid in BIN_IDS if state[cid] == s) for s in ("reicht", "merklich", "verfehlt")},
        "n_M": sum(1 for v in sign.values() if v == "M"), "n_G": sum(1 for v in sign.values() if v == "G"), "n_open": sum(1 for v in sign.values() if v == "offen"),
        "B_over_G": sum(1 for cid in BIN_IDS if cells[cid]["B"]["mean"] > cells[cid]["G"]["mean"]),
        "B_over_M": sum(1 for cid in BIN_IDS if cells[cid]["B"]["mean"] > cells[cid]["M"]["mean"]),
        "max_best": max(g for _, g in best.values()), "q2": q2,
        "pair_BG": sign_counts(cells, BIN_IDS, "B", "G", res["meta"]["n_tune"]), "pair_BM": sign_counts(cells, BIN_IDS, "B", "M", res["meta"]["n_tune"]),
        "pair_MG_unclear": sign_counts(cells, ("u0", "u1", "u2"), "M", "G", res["meta"]["n_tune"]),
        "pair_BG_unclear": sign_counts(cells, ("u0", "u1", "u2"), "B", "G", res["meta"]["n_tune"]),
        "unclear_inc_se": unclear_increase_se(cells),
        "open_KG": open_ids(cells, BIN_IDS, "M", "G", res["meta"]["n_tune"]),
        "open_BM": open_ids(cells, BIN_IDS, "B", "M", res["meta"]["n_tune"]),
        "genom_mid": [(y, c, cells[f"y{int(y * 100)}c{c}"]["share_first"]) for y in (0.25, 0.5) for c in (1, 2, 4, 8) if 0.2 < cells[f"y{int(y * 100)}c{c}"]["share_first"] < 0.8],
        "stable": {thr: stable_interval([R.best_rule(cells[c])[1] for c in ALL_IDS], thr) for thr in (C.VERDICT_REICHT, C.VERDICT_VERFEHLT)},
        "unclear": [100 * (cells[c]["opt_mean_all"] / cells["u0"]["opt_mean_all"] - 1.0) for c in ("u0", "u1", "u2")],
        "tau_dG": sum(cells[c]["tau_curve"]["G"]["0.95"] - cells[c]["G"]["mean"] for c in BIN_IDS) / len(BIN_IDS),
        "tau_dB": sum(cells[c]["tau_curve"]["B"]["0.95"] - cells[c]["B"]["mean"] for c in BIN_IDS) / len(BIN_IDS),
        "unclear_M_best": sum(1 for c in ("u0", "u1", "u2") if cells[c]["M"]["mean"] < min(cells[c]["G"]["mean"], cells[c]["B"]["mean"])),
    }


def _alt_neu_rows(res):
    rows = []
    for cid in ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "g1", "g2"):
        c, old = res["cells"][cid], VORMESSUNG_REGELN[cid]
        rows.append([UI._short(c["params"]), f"{fmt2(old[0])} → {fmt2(c['opt_mean'])}", f"{fmt1(old[1])} → {fmt1(c['M']['mean'])}", f"{fmt1(old[2])} → {fmt1(c['G']['mean'])}",
                     f"{fmt1(old[3])} → {fmt1(c['B']['mean'])}"])
    return UI.md_table(["Zelle", "Optimum (Vormessung → Messreihe)", "kurzsichtig (%)", "Info/Kosten (%)", "billigster nützlicher Test (%)"], rows)


def _alt_neu_genom(res):
    rows = []
    for cid in ("y25c1", "y25c2", "y50c1", "y50c2", "y50c4"):
        c, old = res["cells"][cid], VORMESSUNG_GENOM[cid]
        rows.append([f"Ausbeute {fmt2(c['params']['g_yield'])}, Kosten {fmt1(c['params']['g_cost'])}", f"{old[0]} % → {c['share_first'] * 100:.0f} %", f"{fmt1(old[1])} → {fmt1(c['forced']['mean'])}",
                     f"{fmt1(old[2])} → {fmt1(c['never']['mean'])}"])
    return UI.md_table(["Genom-Zelle", "Genom ist erster Test (Vormessung → Messreihe)", "„immer zuerst“ (% mehr)", "„nie“ (% mehr)"], rows)


def render(res: dict, n_tests: int) -> str:
    t = UI.readme_tables(res)
    d = derived(res)
    cells = res["cells"]
    q0, q1, q2, q5, g1, g2 = (cells[c] for c in ("q0", "q1", "q2", "q5", "g1", "g2"))
    qa = d["q2"]
    st = d["states"]
    ppv_i2, ppv_s2 = BR.ppv(1e-3, 0.99, 0.01, 2), BR.ppv(1e-3, 0.99, 0.01, 2, True)
    k_i, k_s = BR.tests_until(1e-3, 0.99, 0.01, 0.9, False), BR.tests_until(1e-3, 0.99, 0.01, 0.9, True)
    y5, y25 = (lambda c: cells[f"y50c{c}"]), (lambda c: cells[f"y25c{c}"])
    sh = lambda c: f"{c['share_first'] * 100:.0f} %"  # noqa: E731
    return f"""# Diagnose-Testfolge – welcher Test als Nächster? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-diagnose-testfolge-demo.streamlit.app/)** -->

Demo zur **Entscheidungsanalyse** (das zwölfte Stück der gleichnamigen Linie) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Eine Person, mehrere mögliche Diagnosen, mehrere Tests mit Kosten und Fehlern: Die Demo rechnet die **beste Folge exakt** (dynamische Optimierung über alle Zustände mit optimalem Stoppen) und stellt drei **einfache Regeln** daneben. Dazu kommen ein **Genomtest** (ein breiter, teurer Test mit vielen Ausgängen), ein dritter Ausgang **„unklar“** und ein Rechner zur **Basisratenfalle**. **Lehrmodell mit erfundenen Daten, keine medizinische Beratung** und keine Empfehlung für eine Untersuchungsfolge.

Sie setzt dort an, wo [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo) aufhört: Diese Demo nennt sequentielle Tests, die Wahl zwischen verschiedenartigen Tests und mehr als zwei Zustände selbst als nicht gerechnet; hier sind sie gerechnet. Das Stoppen nach jedem Test mit bis zu vier gleichartigen Tests rechnet die [`entscheidung-verfahrenswahl-demo`](https://github.com/sebastian-hanisch/entscheidung-verfahrenswahl-demo) (Verweis unten).

## Warum dieses Problem

Welcher Test als Nächster, und wann reicht es? Wer Tests mit Kosten und Fehlern einsetzt, muss die Reihenfolge und den Abbruch wählen. Kernfragen: **Wie weit liegen einfache Regeln über dem Optimum? Wann lohnt ein teurer breiter Test zuerst? Wann reichen zwei positive Tests nicht, weil sie sich einen Fehler teilen?** Die Antworten stehen vorgerechnet in der Messreihe ({res['meta']['n_instances']} Instanzen je Zelle); die Instanz oben auf der Seite ist nur ein Beispiel.

## Befunde und Korrekturen gegenüber der Vormessung

Dem Bau ging eine Vormessung mit dem Zufallsgenerator von NumPy voraus (andere Instanzen). Die Messreihe der Demo rechnet mit dem eigenen SplitMix64 neu ({res['meta']['n_instances']} Instanzen je Zelle, Stoppschwelle auf den ersten {res['meta']['n_tune']} abgestimmt, Lücke auf den übrigen). **Die Befunde bleiben der Richtung nach bestehen (Rauschen öffnet die Lücke von Info/Kosten und billigstem Test, bei kurzsichtig wächst sie nicht durchgehend; Irrtumspreis 100 und gleichverteiltes Vorab kehren die Reihenfolge von kurzsichtig und Info/Kosten; der Genom-Kipppunkt liegt an denselben Kostenstufen), einzelne Zahlen verschieben sich, und zwei Aussagen der Vormessung gelten so nicht mehr:**

{_alt_neu_rows(res)}

{_alt_neu_genom(res)}

- **„Keine der drei Faustregeln liegt unter 5 %“ gilt nicht.** In {d['n_under5']} von {len(BIN_IDS)} binären Zellen liegt die beste einfache Regel unter 5 % (Zellen {', '.join(d['under5_ids'])}: {', '.join(fmt1(d['best'][c][1]) for c in d['under5_ids'])} %). Die Vormessung nannte nur Zellen mit Rauschen und mittlerem Irrtumspreis.
- **Kurzsichtig gegen Info/Kosten im Standard (Se/Sp 0,9/0,9, L 20):** Vormessung −8,1 ± 1,0, jetzt {_diff(q2)}; das Vorzeichen bleibt, der Abstand ist kleiner. Bei Se/Sp 0,8/0,9 liegt der Abstand jetzt bei {_diff(cells['q3'])} (Vormessung −1,9 ± 1,1, damals innerhalb von zwei Standardfehlern); bei sehr schiefem Vorab (Zipf 1,5) bei {_diff(g2)} (Vormessung −1,9 ± 2,2): dort bleibt es offen.
- **Schwelle der Meldung „verfehlt deutlich“:** Ursprünglich waren 15 % vorgesehen. Die größte Lücke der besten einfachen Regel in der Messreihe ist {fmt1(d['max_best'])} %, der Zustand wäre nie erschienen; die Schwelle steht darum bei {C.VERDICT_VERFEHLT:.0f} % (und „reicht“ unter {C.VERDICT_REICHT:.0f} %). Das ist eine Setzung nach dem Blick auf die Messung, keine Messung. Sie liegt aber in einer Lücke der Messwerte und nicht an einem Grenzfall: In den {len(ALL_IDS)} Zellen mit Regelvergleich gibt jede Schwelle von {fmt1(d['stable'][C.VERDICT_REICHT][0])} bis {fmt1(d['stable'][C.VERDICT_REICHT][1])} % („reicht“) beziehungsweise von {fmt1(d['stable'][C.VERDICT_VERFEHLT][0])} bis {fmt1(d['stable'][C.VERDICT_VERFEHLT][1])} % („verfehlt deutlich“) dieselbe Einteilung; die Lücken gehören zu diesen Zellen und können sich bei einer Neumessung verschieben (der Test `test_the_thresholds_sit_inside_gaps_of_the_cell_values` fängt das).
- **Rechenzeit:** Sie schwankt stark mit der Last der Maschine (die Vormessung auf ruhiger Maschine und der Bau unter Last lagen weit auseinander). Zeiten sind kein Befund und stehen in keinem Test.

## Modell

Jede Instanz hat 4, 6 oder 10 Diagnosen (Standard {C.DEFAULT_N}) mit einem Zipf-Vorab (gleichverteilt, schief mit Exponent 1,0 oder sehr schief mit 1,5), bis zu {C.M_MAX} einfache Tests und einen Irrtumspreis L in mittleren Testkosten. Test t spricht jede Diagnose unabhängig mit Wahrscheinlichkeit {fmt1(C.DENSITY)} an; er fällt bei einer angesprochenen Diagnose mit Wahrscheinlichkeit Se positiv aus, sonst mit 1 − Sp; Kosten lognormal (σ {fmt1(C.COST_SIGMA)}), im Mittel 1. Jeder Test wird höchstens einmal gemacht. Am Ende wird die wahrscheinlichste Diagnose genannt; ein Irrtum kostet L. **Gesamtkosten = Testkosten + L · P(Irrtum).** Der **Zustand** merkt sich je Test, ob er gemacht wurde und mit welchem Ausgang; daraus folgt die Wahrscheinlichkeit jeder Diagnose nach Bayes (Tests unabhängig gegeben die Diagnose).

- **Unklar:** dritter Ausgang jedes einfachen Tests mit Wahrscheinlichkeit u (0; 0,1; 0,3), unabhängig von der Diagnose.
- **Genomtest:** n + 1 Ausgänge; mit Wahrscheinlichkeit y (Ausbeute 0,25 oder 0,5) wird die Diagnose eindeutig gefunden, sonst „kein Befund“ (das ändert die Wahrscheinlichkeiten der Diagnosen nicht; der Test gilt danach als gemacht). Kosten in mittleren Testkosten (1, 2, 4, 8, in der Messreihe auch 16).
- **Rechengrenzen:** höchstens {C.M_MAX} einfache Tests (3^m Zustände), mit „unklar“ höchstens {C.M_CAP_UNCLEAR} (4^m), mit Genomtest höchstens {C.M_CAP_GENOME} einfache Tests; die App sagt es, wenn sie weniger rechnet als eingestellt.
- Vollständige Formeln: Expander **📐 Mathematische Formulierung** der App. Alle Parameter sind angenommen.

## Methodik

| Strategie | Wie sie arbeitet | Grenze |
|---|---|---|
| **Optimum (exakt)** | Rückwärtsrechnung über alle Zustände: f(s) = min(Verlust bei Stopp, min über freie Tests von Kosten plus Erwartung über die Ausgänge); `dtf_solver.py` | 3^m Zustände, deshalb höchstens 8 Tests |
| **kurzsichtig** | noch ein Test, wenn der erwartete Verlust nach *einem* Test plus dessen Kosten unter dem jetzigen liegt | sieht nicht, dass erst zwei Tests zusammen die Entscheidung kippen |
| **Info/Kosten** | der Test mit dem größten Entropiegewinn je Kosten; Stopp bei Wahrscheinlichkeit der besten Diagnose ≥ τ | die Regel benutzt L nicht; τ wird je Zelle abgestimmt und folgt so indirekt dem Irrtumspreis (in der Messreihe {fmt2(cells['q4']['tau']['G'])} / {fmt2(cells['q2']['tau']['G'])} / {fmt2(cells['q5']['tau']['G'])} bei L 5 / 20 / 100) |
| **billigster nützlicher Test** | der billigste Test mit Informationsgewinn; derselbe Stopp | wie Info/Kosten, ignoriert die Höhe der Information (verlangt nur, dass sie positiv ist) |

τ (0,6 bis 0,99) wurde je gemessener Zelle auf den ersten {res['meta']['n_tune']} Instanzen abgestimmt und auf den übrigen gemessen. Eine feste Schwelle 0,95 statt der abgestimmten hätte im Standard Info/Kosten von {_pm(qa, 'G')} % auf {fmt1(qa['tau_curve']['G']['0.95'])} % und den billigsten nützlichen Test von {_pm(qa, 'B')} % auf {fmt1(qa['tau_curve']['B']['0.95'])} % angehoben; im Mittel über die {len(BIN_IDS)} binären Zellen liegt Info/Kosten mit τ = 0,95 um {fmt1(d['tau_dG'])} Punkte und der billigste nützliche Test um {fmt1(d['tau_dB'])} Punkte höher als mit der abgestimmten Schwelle (Mittel der Zellmittel). Alle Regeln werden **exakt** ausgewertet (über alle erreichbaren Zustände), nicht simuliert. Zufall nur über den ganzzahligen SplitMix64 mit `decimal`-Ersatz für Logarithmus, Kosinus und Exponentialfunktion (`dtf_rng.py`; die Suite lief unter Windows und im Linux-Docker-Lauf grün, Neu-Rechnen-Tests als Band); Messreihe: `tools/sweep.py` (Seeds ab {C.SWEEP_SEED_BASE}, die Anzeige-Seeds der App sind 0 bis {C.SEED_MAX}).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Messreihe `data/dtf_results.json` ({res['meta']['n_instances']} Instanzen je Zelle; Einheit: **Instanz**; Lücke = 100 · (Kosten der Regel / Kosten des Optimums − 1), Mittel ± Standardfehler über die letzten {res['meta']['n_instances'] - res['meta']['n_tune']} Instanzen (Regelzellen) beziehungsweise über alle {res['meta']['n_instances']} (Genom-Zellen und die Zeile „Optimum (alle Instanzen)“ der Tabelle „unklar“); Optimum: Mittel über dieselben Instanzen in mittleren Testkosten). Jede Tabelle wird beim Erzeugen der README aus der Datei geschrieben (`tools/make_readme.py`) und von `tests/test_claims.py` auf Gleichheit geprüft.

**Faustregeln gegen das Optimum** (jede Zeile nennt ihre Zelle; Se/Sp: Sensitivität/Spezifität, L: Irrtumspreis in mittleren Testkosten):

{t['regeln']}

- **Im Standard (Se/Sp 0,9/0,9, L 20)** liegt kurzsichtig {_pm(qa, 'M')} %, Info/Kosten {_pm(qa, 'G')} % und der billigste nützliche Test {_pm(qa, 'B')} % über dem Optimum von {fmt2(qa['opt_mean'])} (im Mittel {fmt1(qa['opt_tests_mean'])} Tests, Irrtumswahrscheinlichkeit {fmt1(100 * qa['opt_perr_mean'])} %).
- **Rauschfrei (Se/Sp 1,0/1,0)** liegt Info/Kosten bei {_pm(q0, 'G')} %; schon bei 0,95/0,95 sind es {_pm(q1, 'G')} %. In den {len(BIN_IDS) - 1} Zellen mit Se/Sp unter 1 (sie unterscheiden sich zusätzlich im Irrtumspreis, im Vorab oder in der Zahl der Diagnosen) liegt Info/Kosten zwischen {fmt1(min(cells[c]['G']['mean'] for c in BIN_IDS if c != 'q0'))} und {fmt1(max(cells[c]['G']['mean'] for c in BIN_IDS if c != 'q0'))} %.
- **Welche Regel die kleinere Lücke hat, hängt an den Einstellungen.** Kurzsichtig hat in {d['n_M']} der {len(BIN_IDS)} Zellen eine um mehr als zwei Standardfehler kleinere Lücke als Info/Kosten, Info/Kosten in {d['n_G']} (Se/Sp 1,0/1,0; Irrtumspreis 100; Vorab gleichverteilt); in {_z(d['n_open'])} ist der Unterschied offen. Bei Irrtumspreis 100 liegt kurzsichtig bei {_pm(q5, 'M')} % und Info/Kosten bei {_pm(q5, 'G')} % (Differenz {_diff(q5)}); bei gleichverteiltem Vorab {_pm(g1, 'M')} % gegen {_pm(g1, 'G')} % (Differenz {_diff(g1)}).
- **Der billigste nützliche Test** hat in {d['pair_BG']['over']} von {len(BIN_IDS)} Zellen eine um mehr als zwei Standardfehler größere Lücke als Info/Kosten (gepaart auf denselben Instanzen) und in {d['pair_BM']['over']} Zellen eine größere Lücke als kurzsichtig; gegenüber kurzsichtig ist seine Lücke in {_z(d['pair_BM']['under'])} kleiner und in {_z(d['pair_BM']['open'])} ist der Unterschied offen (nach den Mittelwerten allein ist sie in {d['B_over_G']} beziehungsweise {d['B_over_M']} Zellen größer).
- **Meldung der App** (Lücke der besten einfachen Regel in der Zelle): „reicht“ (unter {C.VERDICT_REICHT:.0f} %) in {st['reicht']} Zellen, „kostet merklich“ in {st['merklich']}, „verfehlt deutlich“ (über {C.VERDICT_VERFEHLT:.0f} %) in {st['verfehlt']}.

**Ergebnis „unklar“** (6 Diagnosen, 6 Tests, Vorab schief, Se/Sp 0,9/0,9, L 20):

{t['unklar']}

- Der Optimalwert steigt bei u = 0,1 um rund {fmt1(d['unclear'][1])} % und bei u = 0,3 um rund {fmt1(d['unclear'][2])} % (Mittel über alle {res['meta']['n_instances']} Instanzen; **ungepaart**: die drei Zellen nutzen verschiedene Instanzen, der Standardfehler des Zuwachses liegt nach der Delta-Methode für das Verhältnis ungepaarter Mittel bei etwa ± {fmt1(d['unclear_inc_se'][0])} beziehungsweise ± {fmt1(d['unclear_inc_se'][1])} Punkten).
- Kurzsichtig hat in {d['pair_MG_unclear']['under']} der drei Zellen eine um mehr als zwei Standardfehler kleinere Lücke als Info/Kosten (gepaart) und nach den Mittelwerten in {d['unclear_M_best']} die kleinste Lücke aller drei Regeln; Info/Kosten hat gegenüber dem billigsten nützlichen Test in {d['pair_BG_unclear']['under']} der drei Zellen eine um mehr als zwei Standardfehler kleinere Lücke, in {_z(d['pair_BG_unclear']['open'])} ist der Unterschied offen.

**Genomtest zuerst** (6 einfache Tests, 6 Diagnosen, Se/Sp 0,9/0,9, L 20; Mehrkosten in Prozent über dem Optimum der Instanz):

{t['genom']}

- Bei Ausbeute 0,50 ist der Genomtest bei Kosten 1 in {sh(y5(1))}, bei 2 in {sh(y5(2))}, bei 4 in {sh(y5(4))} der Instanzen der erste Test; bei Ausbeute 0,25 bei Kosten 1 in {sh(y25(1))}, bei 2 in {sh(y25(2))}. **Der Wechsel (der Anteil fällt auf 0 %) liegt bei Ausbeute 0,25 zwischen den Kosten 1 und 2, bei Ausbeute 0,50 zwischen den Kosten 2 und 4 mittleren Testkosten.** An Stufen mit Anteilen zwischen 20 und 80 % ({'; '.join(f"Ausbeute {fmt2(y)}, Kosten {c}: {s * 100:.0f} %" for y, c, s in d['genom_mid'])}) entscheidet die Instanz, nicht die Kostenstufe. Zwischen den Stufen ist nichts gemessen.
- „Genom nie“ kostet bei Ausbeute 0,50 und Kosten 1 {fmt1(y5(1)['never']['mean'])} % mehr, „Genom immer zuerst“ bei Kosten 4 {fmt1(y5(4)['forced']['mean'])} %: Eine einfache Politik am falschen Ende der Kostenskala ist teuer.

**Basisratenfalle** (Sensitivität 0,99, Falsch-positiv-Rate 0,01 je Test; gemeinsamer Fehler: {BR.DISTURB_SHARE * 100:.0f} % der Gesunden tragen einen Störfaktor, der jeden Test mit Wahrscheinlichkeit {fmt1(BR.DISTURB_PROB)} positiv macht; Wahrscheinlichkeit, krank zu sein; **die Parameter des Störfaktors sind angenommen, nicht gemessen**):

{t['ppv']}

- Bei 1 von 1.000 reichen unter der Unabhängigkeitsannahme {k_i} positive Tests für eine Wahrscheinlichkeit von mindestens 0,9 ({fmt2(ppv_i2)} nach zwei), mit gemeinsamem Fehler sind es {k_s} (nach zwei Tests {fmt2(ppv_s2)}). Nach **einem** positiven Test sind beide Rechnungen gleich.

## Ehrliche Grenzen

- **Die Messreihe gilt für diese Welt:** 4, 6 oder 10 Diagnosen, bis zu 8 einfache Tests, die gemessenen Reglerstufen; zwischen den Stufen und für andere Kombinationen zeigt die App die nächstliegende gemessene Zelle und nennt die abweichenden Parameter.
- **Die Regeln gelten nur mit ihrer abgestimmten Schwelle** und nur für die genannten drei; keine Aussage über die Methodenklasse. Die Stoppschwelle wurde aus sechs Werten gewählt; ein feineres Gitter könnte die Lücken verschieben.
- **Unterschiede unter zwei Standardfehlern sind nicht zu deuten:** kurzsichtig gegen Info/Kosten in {_z(len(d['open_KG']))} ({', '.join(cell_label(cells[c]['params']) for c in d['open_KG'])}), der billigste nützliche Test gegen kurzsichtig in {_z(len(d['open_BM']))} ({' und '.join(cell_label(cells[c]['params']) for c in d['open_BM'])}).
- **Nicht gerechnet:** Wiederholung desselben Tests, gemeinsame Fehlerursachen zwischen Tests (außer im Rechner der Basisratenfalle), Reihenfolgebedingungen, Dauer und Belastung, ein Irrtumspreis je Diagnosenpaar, mehr als zwei Ausgänge außer „unklar“ und Genom, mehr als 8 Tests; die Genom-Ausbeute ist für alle Diagnosen gleich.
- **Nur an erfundenen Instanzen geprüft;** keine echten Tests, keine echten Daten, keine Literaturzahlen als Beleg.

## Verwandte Demos mit demselben mathematischen Modell

- [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo): ein Test, Bayes-Update im umgedrehten Baum, Wert der Stichprobeninformation, n gleichartige unabhängige Tests; dort ausdrücklich nicht gerechnet: sequentielle und abhängige Tests, verschiedenartige Tests, mehr als zwei Zustände. Dasselbe Bayes-Update, hier mit Folge und Stoppen.
- [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo): aus einer Wahrscheinlichkeit eine Entscheidung über das Kostenverhältnis (Schwelle 1/(1+r)). Hier ist die Stopp-Entscheidung derselbe Vergleich von Stopp-Verlust und erwartetem Verlust beim Weitermachen; sie ergibt keine feste Schwelle auf die Wahrscheinlichkeit der besten Diagnose, sondern hängt vom ganzen Zustand ab.
- [`entscheidung-verfahrenswahl-demo`](https://github.com/sebastian-hanisch/entscheidung-verfahrenswahl-demo): vergleicht **Rechenverfahren** (Entscheidungsbaum, Zustandsrechnung, Szenario-LP, gemischt-ganzzahliges Modell) für ein gegebenes Entscheidungsmodell am Kapazitätsfall mit zwei Zuständen; dort darf nach jedem von bis zu vier gleichartigen Tests (gleiche Trefferquote, gleiche Kosten) gestoppt werden. Hier geht es um das **Entscheidungsmodell** selbst: mehrere verschiedenartige Tests mit eigenen Kosten und Fehlern, mehr als zwei Diagnosen, einfache Regeln gegen die exakte Strategie, ein breiter Test, der Ausgang „unklar“ und die Basisratenfalle. Beide rechnen das Stoppen mit einer Rückwärtsrechnung über Zustände; die Zahlen sind nicht vergleichbar.
- [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo): Entscheidungsbaum, Strategie und Rollback; die exakte Strategie hier ist ein Rollback über die Zustandsmenge.
- Nachbar, keine Doppelung: das Vorhaben zu Gruppentests (Mischproben bei vielen Personen; hier eine Person und mehrere Tests).

## Verifikation ({n_tests} Tests)

- **Kern gegen Handrechnung:** ein Test (Optimum 2), zu teurer Test (5), zwei gleiche Tests (11, erzwungen 12), Genomtest mit Ausbeute 1 und 0,5 (1 und 3,5), reiner „unklar“-Test (5); Zerlegung der Kosten in Testkosten und Irrtumsverlust.
- **Orakel** (`tests/test_oracle_diagnose_testfolge.py`): (1) **Aufzählung aller Strategien** als explizite Entscheidungsbäume (9 Strategien bei 2, 244 bei 3 Tests; Wert je Baum über gemeinsame Wahrscheinlichkeiten ohne Posterior) gegen die DP auf 24 Kleinstinstanzen (binär, „unklar“, Genom); (2) **Simulation** der Strategien mit festem Zufalls-Seed gegen die exakte Auswertung innerhalb von vier Standardfehlern; (3) **rauschfreier Grenzfall** gegen eine Teilmengen-DP über die Diagnosenmengen; (4) **Basisratenrechnung** gegen Bruchrechnung und gegen die Simulation einer Bevölkerung mit Störfaktor.
- **Eigenschaften und Zweige:** keine Regel liegt je unter dem Optimum (25 Instanzen, mehrere τ); jede Regel weicht auf mindestens einer Instanz von jeder anderen ab; τ ändert Info/Kosten und billigsten Test; „unklar“ und der Genomtest ändern das Optimum; ein Genomtest, dessen Kosten den Irrtumspreis übersteigen, wird nie gewählt (Kosten 30 bei L 20); Irrtumspreis 10^6 nähert sich dem Bayes-Fehler aller Tests (an einer Instanz auf 1e-4 geprüft).
- **Messreihe:** Struktur und innere Stimmigkeit der Datei, **Neu-Rechnen einzelner Instanzen als Band** (relative Toleranz 1e-9), Aggregation (Schwelle nur auf der Abstimmungshälfte), Zell-Zuordnung jedes Presets (jedes liegt genau auf einer gemessenen Zelle), Schwellen der Meldung inklusive.
- **Presets und Verdikte:** Abnahmekriterium je Preset gegen die Messreihe und jedes Kriterium an seiner Schwelle mit künstlichen Werten (`tests/test_stories.py`); jedes Verdikt an gesetzten Werten (`tests/test_verdicts.py`).
- **Oberfläche:** AppTest (jeder Button, jeder Regler am Minimum und Maximum einzeln und zusammen, Permalink mit Einrasten und Begrenzen, Rechner der Basisratenfalle, Hinweis bei gekürzter Testzahl), Layout (Schnellstart, fester Kopf, aktives Preset hervorgehoben; jede Regleränderung hebt die Markierung auf), Footer, Laufzeit-Importe.
- **README:** wird aus `data/dtf_results.json` erzeugt; `tests/test_claims.py` prüft, dass die README dem Erzeugnis entspricht und die Testzahl stimmt.
- Keine Mutationstests (seit 2026-10-28 nicht mehr Teil der Demo-Prüfung). Nicht in der CI laufen: die Messreihe selbst (`tools/sweep.py`, einige Minuten).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | Git-Ausnahmen, Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Seitenaufbau nach DEMO-PLAYBOOK §2) |
| `dtf_constants.py` | Reglerstufen, feste Annahmen, Presets, gemessene Zellen |
| `dtf_rng.py` | SplitMix64 (`decimal`-Ersatz für Logarithmus, Kosinus, Exponentialfunktion), wörtliche Kopie der Vorlage |
| `dtf_model.py` | Einstellungen, Instanz-Erzeugung, Likelihood-Tabellen, Problem (Posterior, Zustände) |
| `dtf_solver.py` | exakte DP mit Stoppen, Auswertung beliebiger Strategien, Baum der besten Folge |
| `dtf_rules.py` | die drei Faustregeln als Strategien |
| `dtf_basisrate.py` | Basisratenfalle: PPV, gemeinsamer Fehler, Zahl positiver Tests bis 0,9 |
| `dtf_measure.py` | Messung einer Instanz und Aggregation je Zelle |
| `dtf_results.py` | Messreihe laden, nächstliegende Zelle, Urteil in drei Zuständen |
| `dtf_evaluation.py` | Auswertung einer Instanz, Verdikte als reine Funktionen |
| `dtf_stories.py` | Abnahmekriterien der Presets |
| `dtf_presets.py`, `dtf_layout.py` | Permalink, Presets, fester Kopf |
| `dtf_visualization.py`, `dtf_ui_panel.py`, `dtf_pdf_export.py` | Figuren, Tabellen und Textbausteine, PDF-Auswertung |
| `data/dtf_results.json` | vorgerechnete Messreihe (24 Zellen) |
| `tools/sweep.py`, `tools/report.py`, `tools/make_readme.py` | Messreihe reproduzieren, berichten (mit den Zahlen der Vormessung), README erzeugen |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken, plus pytest |
| `tests/` | siehe Verifikation |

## Bewusst nicht umgesetzt

- **Wiederholung desselben Tests und abhängige Fehler zwischen Tests** (außer im Rechner der Basisratenfalle): würde das Modell zu einem Problem mit gemeinsamer Ursache machen und die Zustände vervielfachen.
- **Mehr als 8 einfache Tests:** 3^m Zustände; bei 9 Tests wären es 19.683, bei 10 etwa 59.000, deutlich langsamer in der App.
- **Irrtumspreis je Diagnosenpaar, Dauer und Belastung der Person, Reihenfolgebedingungen:** hier ein pauschaler Irrtumspreis.
- **Krankheitsabhängige Ausbeute des Genomtests** und echte Wissensbasen: die Ausbeute ist ein Regler, für alle Diagnosen gleich.
- **Echte Tests, echte Daten, Literaturwerte als Eingabe:** erfundene Instanzen.

## Lokal ausführen

```bash
python -m venv venv
venv\\Scripts\\activate           # Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

{FOOTER}
"""


def main():
    res = json.loads((ROOT / "data" / "dtf_results.json").read_text(encoding="utf-8"))
    text = render(res, count_tests())
    target = ROOT / "README.md"
    if "--check" in sys.argv:
        ok = target.read_text(encoding="utf-8").replace("\r\n", "\n") == text
        print("README stimmt" if ok else "README weicht ab")
        sys.exit(0 if ok else 1)
    target.write_text(text, encoding="utf-8", newline="\n")
    print("README.md geschrieben")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
