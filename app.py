"""Diagnose-Testfolge – welcher Test als Nächster? - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Eine Person, mehrere mögliche Diagnosen, mehrere Tests mit Kosten und Fehlern: die exakte beste Folge (DP mit optimalem Stoppen) gegen einfache Regeln.
Lehrmodell mit erfundenen Daten, keine medizinische Beratung. Aufbau nach DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""


import streamlit as st

import dtf_basisrate as BR
import dtf_constants as C
import dtf_results as R
import dtf_ui_panel as UI
from dtf_evaluation import analyse, fmt1, fmt2, verdict_genome, verdict_ppv, verdict_rules
from dtf_layout import dunkles_design, kopf_css
from dtf_model import effective_m
from dtf_pdf_export import build_pdf
from dtf_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    settings_from_state,
    sidebar_presets,
    sync_query_params,
)
from dtf_rules import RULE_NAMES
from dtf_visualization import build_gap_bars, build_genome_curve, build_genome_excess, build_ppv, build_prior, build_regime, build_tree

st.set_page_config(page_title="Diagnose-Testfolge – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne die beste Folge (alle Zustände) ...")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _pdf(settings):
    return build_pdf(analyse(settings), UI.settings_text(settings))


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in dtf_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔬 Diagnose-Testfolge – welcher Test als Nächster?")
st.markdown(
    """
**Eine Person, mehrere mögliche Diagnosen, mehrere Tests mit Kosten und Fehlern:** Welcher Test kommt als Nächster, und wann hört man auf? Die Demo rechnet die **beste Folge exakt**
(dynamische Optimierung mit optimalem Stoppen) und stellt **einfache Regeln** daneben. Sie fragt: Wie weit liegen die Regeln über dem Optimum? Wann lohnt ein **teurer breiter Test** (ein Genomtest) zuerst?
Und wann reichen **zwei positive Tests** nicht, weil sie sich einen Fehler teilen? **Lehrmodell mit erfundenen Daten, keine medizinische Beratung.**

Direkt darunter erklärt **💬 In einfachen Worten** die Aufgabe ohne Formeln. Weiter unten: die drei **Kernfragen** und die Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**, **Wie funktioniert diese Demo?**, **📐 Mathematische Formulierung** und **⚠️ Annahmen, Grenzen und bekannte Probleme**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/diagnose-testfolge-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich vor, ein Gerät fällt aus und es kommen mehrere Ursachen in Frage. Jede Prüfung kostet etwas, und keine irrt nie. Wer am Ende die falsche Ursache nennt, zahlt dafür. Die Demo rechnet nach, in welcher Reihenfolge man prüfen sollte und wann man aufhört; sie erzählt das als Diagnose mit Tests und rechnet mit erfundenen Zahlen. Die **beste Folge** bestimmt sie exakt, indem sie in jedem Zwischenstand alle noch möglichen Prüfungen durchprobiert und den billigsten Gesamtweg nimmt. Daneben stehen einfache Regeln, ein breiter, teurer Test, ein dritter Ausgang „unklar“ und ein Rechner dafür, warum zwei positive Meldungen nicht reichen, wenn sie sich einen Fehler teilen. Es ist ein Rechenmodell und keine medizinische Beratung.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/diagnose-testfolge-erklaert.html)
        """
    )
# erklaerseite:einfach:end

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Tests und Diagnosen**")
    st.slider("Zahl der Tests m", *bounds("m_slider"), key="m_slider", help="Wie viele einfache Tests zur Wahl stehen. Bei „unklar“ werden höchstens 7, mit Genomtest höchstens 6 gerechnet (Rechenzeit).")
    st.select_slider("Zahl der Diagnosen n", options=options("n_select"), key="n_select", help="Wie viele Diagnosen in Frage kommen.")
    st.select_slider("Vorab-Verteilung", options=options("zipf_select"), key="zipf_select", format_func=lambda z: C.ZIPF_LABELS[z],
                     help="Wie ungleich die Diagnosen vorab wahrscheinlich sind (Zipf-Exponent 0, 1,0 oder 1,5).")
    st.markdown("**Testgüte und Irrtum**")
    st.select_slider("Testgüte (Sensitivität / Spezifität)", options=options("quality_select"), key="quality_select",
                     help="Wahrscheinlichkeit, bei einer angesprochenen Diagnose positiv zu sein, und negativ, wenn sie nicht angesprochen ist.")
    st.select_slider("Irrtumspreis L (in mittleren Testkosten)", options=options("loss_select"), key="loss_select",
                     help="Was die Nennung der falschen Diagnose kostet, gemessen in mittleren Kosten eines Tests.")
    st.select_slider("Anteil „unklar“", options=options("unclear_select"), key="unclear_select", format_func=lambda u: str(u).replace(".", ","),
                     help="Wahrscheinlichkeit, dass ein Test ein unklares Ergebnis liefert, unabhängig von der Diagnose.")
    st.markdown("**Genomtest**")
    genome_on = st.checkbox("Genomtest dazunehmen", key="genome_check", help="Ein breiter, teurer Test, der die Diagnose mit Wahrscheinlichkeit y eindeutig findet, sonst „kein Befund“.")
    st.select_slider("Kosten des Genomtests", options=options("gcost_select"), key="gcost_select", format_func=lambda c: fmt1(c), disabled=not genome_on,
                     help="In mittleren Testkosten; wirkt nur mit Genomtest (die Kernfrage 2 zeigt die Kosten auch ohne).")
    st.select_slider("Ausbeute des Genomtests", options=options("gyield_select"), key="gyield_select", format_func=lambda y: fmt2(y), disabled=not genome_on,
                     help="Wahrscheinlichkeit, dass der Genomtest die Diagnose eindeutig findet.")
    st.markdown("**Instanz**")
    st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

settings = settings_from_state()
sync_query_params({k: st.session_state[k] for k in ("m_slider", "n_select", "zipf_select", "quality_select", "loss_select", "unclear_select", "genome_check",
                                                      "gcost_select", "gyield_select", "seed_input")})
a = _analysis(settings)
res = R.load()
m_eff = effective_m(settings.m, settings.u, settings.genome)

# --- Eine Instanz (live) -------------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Eine Instanz (live)")
st.caption("Eine Instanz belegt nichts: die Aussagen der Kernfragen kommen aus der vorgerechneten Messreihe (Zahl der Instanzen steht dabei).")
if m_eff != settings.m:
    st.info(f"Es werden {m_eff} einfache Tests gerechnet, nicht {settings.m}: " + ("mit Genomtest höchstens 6, " if settings.genome else "bei „unklar“ höchstens 7, ") + "damit die exakte Rechnung schnell bleibt.")
best = a.best_rule()
c1, c2 = st.columns(2)
c1.metric("Erwartete Gesamtkosten (Optimum)", fmt2(a.opt.total), delta=f"Testkosten {fmt2(a.opt.test_cost)} + Irrtumsverlust {fmt2(a.opt.loss)}", delta_color="off")
c2.metric("Wahrscheinlichkeit des Irrtums", UI.pct(100 * a.opt.p_error), delta=f"{fmt2(a.opt.n_tests)} Tests im Mittel", delta_color="off")
c3, c4 = st.columns(2)
c3.metric("Beste einfache Regel", RULE_NAMES[best], delta=f"{a.gap(best):+.1f} % gegen das Optimum".replace(".", ","), delta_color="inverse")
c4.metric("Bester erster Test", a.first_test or "keiner (sofort Stopp)", delta=f"{len(a.prior)} Diagnosen, {a.m_total} Tests", delta_color="off")
st.markdown("**Baum der besten Folge** (Knoten: Test oder Stopp mit genannter Diagnose; Kanten: Ergebnis mit Wahrscheinlichkeit)")
st.plotly_chart(build_tree(a.tree, settings.u > 0, len(a.prior)), width="stretch", key="tree_chart")
st.caption("Kanten: – negativ, + positiv, ? unklar; beim Genomtest die gefundene Diagnose oder „kein“ (kein Befund). Äste mit weniger als 2 % der Fälle und Ebenen unter der dritten sind gekürzt. D1 bis D" + str(len(a.prior)) + " sind die Diagnosen, T1 bis T" + str(m_eff) + " die einfachen Tests.")
st.download_button("📄 Auswertung als PDF", data=_pdf(settings), file_name="diagnose-testfolge-auswertung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.markdown("## ❓ Kernfrage 1: Wie weit liegen einfache Regeln über dem Optimum?")
cell_res = res["cells"][a.rule_cell]
rule, gap_val = R.best_rule(cell_res)
kind, text = verdict_rules(R.verdict_state(gap_val), rule, gap_val, cell_res[rule]["se"], cell_res["n_instances"] - cell_res["n_tune"], not a.rule_diff)
getattr(st, kind)(text)
st.plotly_chart(build_gap_bars(cell_res, "Gemessene Zelle: " + UI.cell_label(res["cells"][a.rule_cell]["params"])), width="stretch", key="gap_chart")
note = UI.diff_note(a.rule_diff)
st.caption((note + " " if note else "") + "Optimum der Zelle: " + fmt2(cell_res["opt_mean"]) + f" mittlere Testkosten (Mittel über {cell_res['n_instances'] - cell_res['n_tune']} Instanzen; die Stoppschwelle der Regeln wurde auf {cell_res['n_tune']} anderen Instanzen abgestimmt).")

st.markdown("## ❓ Kernfrage 2: Wann lohnt ein teurer breiter Test zuerst?")
g_id, g_diff = UI.genome_cell_for(settings)
g = res["cells"][g_id]
kind, text = verdict_genome(g["share_first"], g["forced"]["mean"], g["never"]["mean"], settings.g_yield, settings.g_cost, not g_diff)
getattr(st, kind)(text)
if settings.genome and a.genome is not None:
    st.caption(f"Diese Instanz: Der Genomtest ist {'der erste Test' if a.genome['first'] else 'nicht der erste Test'} der besten Folge; „Genom immer zuerst“ kostet {fmt1(a.genome['forced_gap'])} %, „Genom nie“ {fmt1(a.genome['never_gap'])} % mehr als das Optimum (eine Instanz).")
else:
    st.caption("Der Genomtest ist ausgeschaltet; die Aussage oben kommt aus der gemessenen Zelle zu den Genom-Reglern (Seitenleiste).")
st.plotly_chart(build_genome_curve(res, settings.g_yield), width="stretch", key="genome_chart")
st.caption(f"Gemessene Zelle: {UI.cell_label(g['params'])}; Anteil in {g['n_instances']} Instanzen. " + UI.diff_note(g_diff))

st.markdown("## ❓ Kernfrage 3: Wann reichen zwei positive Tests nicht?")
st.caption("Rechner zur Basisratenfalle: Wie wahrscheinlich ist man krank, wenn k Tests positiv sind? Unabhängig gerechnet, oder wenn die Tests einen Fehler teilen.")
p1, p2, p3, p4 = st.columns(4)
prev = p1.select_slider("Anteil Kranke in der getesteten Gruppe", options=BR.PREVALENCE_OPTIONS, value=1e-3, format_func=lambda x: BR.PREVALENCE_LABELS[x], key="ppv_prev")
se_t = p2.select_slider("Sensitivität", options=BR.SE_OPTIONS, value=0.99, format_func=lambda x: fmt2(x), key="ppv_se")
fpr_t = p3.select_slider("Falsch-positiv-Rate je Test", options=BR.FPR_OPTIONS, value=0.01, format_func=lambda x: fmt2(x), key="ppv_fpr")
k_t = p4.slider("Positive Tests k", 1, 6, 2, key="ppv_k")
ppv_i, ppv_s = BR.ppv(prev, se_t, fpr_t, k_t), BR.ppv(prev, se_t, fpr_t, k_t, True)
kind, text = verdict_ppv(prev, k_t, ppv_i, ppv_s)
getattr(st, kind)(text)
ks = list(range(1, 7))
st.plotly_chart(build_ppv(ks, [BR.ppv(prev, se_t, fpr_t, k) for k in ks], [BR.ppv(prev, se_t, fpr_t, k, True) for k in ks]), width="stretch", key="ppv_chart")
st.dataframe(UI.tests_until_rows(prev, se_t, fpr_t), hide_index=True, width="stretch")
st.caption(f"Gemeinsamer Fehler (angenommen, nicht gemessen): {BR.DISTURB_SHARE * 100:.0f} % der Gesunden tragen einen Störfaktor, der jeden Test mit Wahrscheinlichkeit {fmt1(BR.DISTURB_PROB)} positiv macht; die Falsch-positiv-Rate je Test bleibt gleich.".replace(".0 %", " %"))

st.markdown("---")

# --- Methodenvergleich ---------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    tab_tree, tab_rules, tab_gen, tab_ppv, tab_sweep = st.tabs(["🌳 Baum und Tests", "📏 Regeln", "🧬 Genom und ❓ unklar", "🧪 Basisratenfalle", "📊 Messreihe"])
    with tab_tree:
        st.markdown("**Die Tests dieser Instanz** (Kosten in mittleren Testkosten; „spricht an“: die Diagnosen, bei denen der Test mit Wahrscheinlichkeit Sensitivität positiv ausfällt, sonst mit 1 − Spezifität):")
        st.dataframe(UI.tests_rows(a), hide_index=True, width="stretch")
        st.markdown("**Vorab-Wahrscheinlichkeit der Diagnosen**")
        st.plotly_chart(build_prior(a.prior), width="stretch", key="prior_chart")
    with tab_rules:
        st.markdown("**Optimum und drei Regeln auf dieser Instanz** (eine Instanz, kein Beleg; Kosten in mittleren Testkosten):")
        st.dataframe(UI.rules_rows(a), hide_index=True, width="stretch")
        st.markdown(
            "- **kurzsichtig:** noch einen Test, wenn der erwartete Verlust nach *einem* Test plus dessen Kosten unter dem jetzigen liegt.\n"
            "- **Info/Kosten:** der Test mit dem größten Entropiegewinn je Kosten; Stopp, sobald die beste Diagnose mindestens die Stoppschwelle τ erreicht.\n"
            "- **billigster nützlicher Test:** der billigste Test mit Informationsgewinn; derselbe Stopp.\n\n"
            "τ ist keine Kostengröße: sie wurde je gemessener Zelle auf anderen Instanzen abgestimmt und steht in der Tabelle."
        )
    with tab_gen:
        st.markdown("**Genomtest zuerst** (gemessene Zellen zu Ihrer Ausbeute; Kosten in mittleren Testkosten):")
        st.plotly_chart(build_genome_excess(res, settings.g_yield), width="stretch", key="genome_excess_chart")
        st.dataframe(UI.genome_rows(res, settings.g_yield), hide_index=True, width="stretch")
        st.markdown("**Ergebnis „unklar“** (dritter Ausgang, gemessene Zellen mit 6 Tests, Testgüte 0,9 / 0,9, Irrtumspreis 20):")
        st.dataframe(UI.unclear_rows(res), hide_index=True, width="stretch")
    with tab_ppv:
        st.markdown("**Derselbe Test, andere Gruppe** (Sensitivität 0,99, Falsch-positiv-Rate 0,01): Wahrscheinlichkeit krank zu sein nach einem positiven Test")
        st.dataframe(UI.referral_rows(), hide_index=True, width="stretch")
        st.markdown("Die Wahrscheinlichkeit nach einem positiven Test hängt am Anteil der Kranken in der getesteten Gruppe, nicht nur am Test. Wer überwiesen wurde, gehört schon zu einer anderen Gruppe als die Bevölkerung.")
    with tab_sweep:
        st.markdown(f"**Vorgerechnete Messreihe** ({res['meta']['n_instances']} Instanzen je Zelle; Stoppschwelle auf den ersten {res['meta']['n_tune']} abgestimmt, Lücke auf den übrigen; Mittel ± Standardfehler):")
        st.plotly_chart(build_regime(res, [f"q{i}" for i in range(7)], ["1,0/1,0, L 20", "0,95/0,95, L 20", "0,9/0,9, L 20", "0,8/0,9, L 20", "0,9/0,9, L 5", "0,9/0,9, L 100", "0,7/0,8, L 20"]),
                        width="stretch", key="regime_chart")
        st.dataframe(UI.messreihe_rows(res, [f"q{i}" for i in range(7)] + [f"g{i}" for i in range(4)]), hide_index=True, width="stretch")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        """
**Zustand.** Jeder Test wird höchstens einmal gemacht. Der Zustand merkt sich, welche Tests gemacht wurden und mit welchem Ergebnis. Daraus folgt nach Bayes die Wahrscheinlichkeit jeder Diagnose.

**Stoppen gehört dazu.** Nach jedem Ergebnis gibt es zwei Möglichkeiten: jetzt die wahrscheinlichste Diagnose nennen (dann droht der **Irrtumsverlust** L · (1 − Wahrscheinlichkeit der besten Diagnose)) oder noch einen Test machen (dann kostet er etwas, und es folgt wieder dieselbe Wahl). Die **exakte Strategie** probiert in jedem Zustand alle noch freien Tests durch und nimmt den billigsten Gesamtweg; es gibt 3^m Zustände (4^m bei „unklar“), deshalb höchstens 8 einfache Tests.

**Faustregeln.** Dieselbe Entscheidung ohne Rechnung über alle Zustände: *kurzsichtig* schaut einen Test voraus, *Info/Kosten* nimmt den Test mit dem größten Informationsgewinn je Kosten, *billigster nützlicher Test* nimmt den billigsten mit Information. Die beiden letzten brauchen eine Stoppschwelle τ; eine feste Schwelle (zum Beispiel 0,95) hätte beide Regeln schlechter aussehen lassen (Zahlen in der README), darum wurde τ je gemessener Zelle auf anderen Instanzen abgestimmt.

**Genomtest.** Ein Test mit n + 1 Ausgängen: mit Wahrscheinlichkeit y wird die Diagnose eindeutig gefunden, sonst „kein Befund“, und die Wahrscheinlichkeiten der Diagnosen ändern sich nicht (der Test gilt danach als gemacht). **„Unklar“** ist ein dritter Ausgang jedes einfachen Tests ohne Information.

**Basisratenfalle.** Kernfrage 3 rechnet die Wahrscheinlichkeit, krank zu sein, nach k positiven Tests: einmal mit unabhängigen Fehlern, einmal mit einem Störfaktor, den die Tests teilen.

**Messreihe.** Die Aussagen der Kernfragen sind vorgerechnet (Zahl der Instanzen und Seeds stehen in der README); die Instanz oben ist nur ein Beispiel.
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/diagnose-testfolge-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ mögliche Diagnosen mit Vorab $p_d$, $m$ Tests mit Kosten $c_t$ und Ausgängen $o$ sowie ein Irrtumspreis $L$. Gesucht ist eine Strategie, die jedem Zustand den nächsten Test oder das Aufhören zuordnet, mit den kleinsten erwarteten Gesamtkosten.

| Symbol | Bedeutung |
|---|---|
| $d$, $n$ | Diagnose, Zahl der Diagnosen |
| $p_d$ | Vorab-Wahrscheinlichkeit der Diagnose $d$, Summe 1 |
| $t$, $m$ | Test, Zahl der einfachen Tests |
| $c_t$ | Kosten von Test $t$, im Mittel 1 |
| $o$ | Ausgang eines Tests (negativ, positiv; mit „unklar“ drei; Genomtest $n+1$) |
| $\ell_t(o \mid d)$ | Wahrscheinlichkeit des Ausgangs $o$ von Test $t$, wenn Diagnose $d$ vorliegt |
| $s$ | Zustand: für jeden Test, ob er gemacht wurde und mit welchem Ausgang |
| $P(d \mid s)$ | Wahrscheinlichkeit der Diagnose $d$ im Zustand $s$ |
| $L$ | Irrtumspreis in mittleren Testkosten |
| $f(s)$ | kleinste erwartete Gesamtkosten ab Zustand $s$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/diagnose-testfolge-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown("Wahrscheinlichkeit der Diagnose im Zustand s, wenn bisher die Ausgänge o_t der gemachten Tests T(s) beobachtet wurden (Tests unabhängig gegeben die Diagnose):")
    st.latex(r"P(d \mid s) = \frac{p_d \prod_{t \in T(s)} \ell_t(o_t \mid d)}{\sum_{d'} p_{d'} \prod_{t \in T(s)} \ell_t(o_t \mid d')}")
    st.markdown("Optimale Strategie (Stopp: die wahrscheinlichste Diagnose nennen, Verlust L mal Gegenwahrscheinlichkeit):")
    st.latex(r"f(s) = \min\Big( L\,\big(1 - \max_d P(d \mid s)\big),\ \min_{t \notin T(s)} \Big[ c_t + \sum_{o} P(o \mid s)\, f(s \cup \{(t,o)\}) \Big] \Big)")
    st.markdown("Wahrscheinlichkeit des Ausgangs o von Test t im Zustand s:")
    st.latex(r"P(o \mid s) = \sum_d P(d \mid s)\, \ell_t(o \mid d)")
    st.markdown("Einfacher Test mit Sensitivität Se und Spezifität Sp, der die Diagnosen D_t anspricht; mit Anteil u unklar:")
    st.latex(r"\ell_t(\text{pos} \mid d) = (1-u)\,\mathrm{Se}\ \text{für}\ d \in D_t,\quad (1-u)\,(1-\mathrm{Sp})\ \text{sonst};\qquad \ell_t(\text{unklar} \mid d) = u")
    st.markdown("Genomtest mit Ausbeute y (Ausgang d: Diagnose d gefunden, Ausgang n + 1: kein Befund):")
    st.latex(r"\ell_G(d \mid d) = y,\qquad \ell_G(\text{kein Befund} \mid d) = 1 - y")
    st.markdown("Wahrscheinlichkeit krank zu sein nach k positiven Tests bei Anteil Kranker π (Basisratenfalle, unabhängige Fehler; Falsch-positiv-Rate φ):")
    st.latex(r"\mathrm{PPV}_k = \frac{\pi\,\mathrm{Se}^k}{\pi\,\mathrm{Se}^k + (1-\pi)\,\varphi^k}")
    st.markdown("Mit gemeinsamem Störfaktor (Anteil a der Gesunden, jeder Test dann mit Wahrscheinlichkeit h positiv; Rest mit Wahrscheinlichkeit r, so dass a·h + (1 − a)·r = φ):")
    st.latex(r"\mathrm{PPV}_k = \frac{\pi\,\mathrm{Se}^k}{\pi\,\mathrm{Se}^k + (1-\pi)\,\big(a\,h^k + (1-a)\,r^k\big)}")
    st.markdown("Dies ist das vollständige Modell der Demo (die Rechnung ist genau diese Gleichung, aufgezählt über alle Zustände). Nicht enthalten: Wiederholung desselben Tests, abhängige Fehler zwischen Tests, Reihenfolgebedingungen, Irrtumspreis je Diagnosenpaar.")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen**
- Die Tests sind gegeben die Diagnose unabhängig; kein Test wird wiederholt; jede Verwechslung kostet gleich viel (L); der Vorab ist bekannt.
- Alle Zahlen (Kosten, Sensitivität, Spezifität, Irrtumspreis, Ausbeute, Störfaktor) sind **angenommen**, nicht aus Studien. Das ist ein Lehrmodell mit erfundenen Daten, **keine medizinische Beratung** und keine Empfehlung für eine Untersuchungsfolge.
- Die Genom-Ausbeute ist für alle Diagnosen gleich, und „kein Befund“ lehrt nichts.

**Grenzen**
- Die Messreihe gilt für diese Welt: 6 beziehungsweise 4 oder 10 Diagnosen, höchstens 8 einfache Tests, die gemessenen Stufen der Regler. Für andere Einstellungen zeigt die App die nächstliegende gemessene Zelle und sagt es.
- Nicht gerechnet: Wiederholung desselben Tests, gemeinsame Fehlerursachen zwischen Tests (außer in Kernfrage 3), Dauer und Belastung der Person, Reihenfolgebedingungen, Irrtumspreis je Diagnosenpaar, mehr als zwei Ausgänge außer „unklar“ und Genom, mehr als 8 Tests.
- Aussagen über die Regeln gelten für die **gemessene Regel mit der abgestimmten Stoppschwelle**, nicht für die Methodenklasse.

**Bekannte Probleme und Unsicherheiten**
- Die Messreihe ist klein (120 Instanzen je Zelle); Unterschiede unter zwei Standardfehlern sind nicht zu deuten.
- Die Stoppschwelle wurde auf einer Hälfte der Instanzen abgestimmt und auf der anderen gemessen; eine andere Wahl des Gitters könnte die Lücken verschieben.
- Die Sprünge zwischen den Stufen der Regler (zum Beispiel Genomkosten 2 und 4) sind gemessen, dazwischen gibt es keine Messung.
- Nur an erfundenen Instanzen geprüft; echte Tests und Daten wurden nicht verwendet.
"""
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
