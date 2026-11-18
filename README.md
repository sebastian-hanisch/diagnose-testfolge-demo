# Diagnose-Testfolge – welcher Test als Nächster? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-diagnose-testfolge-demo.streamlit.app/)** -->

Demo zum **Erkennen unter Unsicherheit** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Eine Person, mehrere mögliche Diagnosen, mehrere Tests mit Kosten und Fehlern: Die Demo rechnet die **beste Folge exakt** (dynamische Optimierung über alle Zustände mit optimalem Stoppen) und stellt drei **einfache Regeln** daneben. Dazu kommen ein **Genomtest** (ein breiter, teurer Test mit vielen Ausgängen), ein dritter Ausgang **„unklar“** und ein Rechner zur **Basisratenfalle**. **Lehrmodell mit erfundenen Daten, keine medizinische Beratung** und keine Empfehlung für eine Untersuchungsfolge.

Sie setzt dort an, wo [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo) aufhört: Diese Demo nennt sequentielle Tests, die Wahl zwischen verschiedenartigen Tests und mehr als zwei Zustände selbst als nicht gerechnet; hier sind sie gerechnet. Das Stoppen nach jedem Test mit bis zu vier gleichartigen Tests rechnet die [`entscheidung-verfahrenswahl-demo`](https://github.com/sebastian-hanisch/entscheidung-verfahrenswahl-demo) (Verweis unten).

## Warum dieses Problem

Welcher Test als Nächster, und wann reicht es? Wer Tests mit Kosten und Fehlern einsetzt, muss die Reihenfolge und den Abbruch wählen. Kernfragen: **Wie weit liegen einfache Regeln über dem Optimum? Wann lohnt ein teurer breiter Test zuerst? Wann reichen zwei positive Tests nicht, weil sie sich einen Fehler teilen?** Die Antworten stehen vorgerechnet in der Messreihe (120 Instanzen je Zelle); die Instanz oben auf der Seite ist nur ein Beispiel.

## Befunde und Korrekturen gegenüber der ersten Messung

Dem Bau ging eine erste Messung mit dem Zufallsgenerator von NumPy voraus (andere Instanzen). Die Messreihe der Demo rechnet mit dem eigenen SplitMix64 neu (120 Instanzen je Zelle, Stoppschwelle auf den ersten 60 abgestimmt, Lücke auf den übrigen). **Die Befunde bleiben der Richtung nach bestehen (Rauschen öffnet die Lücke von Info/Kosten und billigstem Test, bei kurzsichtig wächst sie nicht durchgehend; Irrtumspreis 100 und gleichverteiltes Vorab kehren die Reihenfolge von kurzsichtig und Info/Kosten; der Genom-Kipppunkt liegt an denselben Kostenstufen), einzelne Zahlen verschieben sich, und zwei Aussagen der ersten Messung gelten so nicht mehr:**

| Zelle | Optimum (erste Messung → Messreihe) | kurzsichtig (%) | Info/Kosten (%) | billigster nützlicher Test (%) |
|---|---|---|---|---|
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 1,0/1,0, L 20 | 1,13 → 1,06 | 2,7 → 3,4 | 1,4 → 1,8 | 4,0 → 3,6 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,95/0,95, L 20 | 3,13 → 2,86 | 6,2 → 6,0 | 14,6 → 15,1 | 34,1 → 32,1 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 20 | 4,43 → 4,45 | 5,0 → 6,7 | 13,2 → 12,1 | 24,1 → 23,4 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,8/0,9, L 20 | 6,20 → 6,04 | 10,0 → 8,6 | 11,9 → 12,9 | 18,5 → 19,2 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 5 | 2,17 → 2,01 | 0,8 → 1,8 | 10,8 → 12,8 | 20,1 → 20,0 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 100 | 12,20 → 13,23 | 25,0 → 21,8 | 12,8 → 12,2 | 15,3 → 15,1 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,7/0,8, L 20 | 9,34 → 9,46 | 5,4 → 4,8 | 12,6 → 12,2 | 16,2 → 15,3 |
| 6 Diagnosen, 8 Tests, Vorab gleich, Se/Sp 0,9/0,9, L 20 | 5,23 → 5,20 | 22,6 → 22,2 | 14,4 → 13,3 | 21,2 → 19,8 |
| 6 Diagnosen, 8 Tests, Vorab sehr schief, Se/Sp 0,9/0,9, L 20 | 3,69 → 3,78 | 17,1 → 13,8 | 19,0 → 13,4 | 27,4 → 25,4 |

| Genom-Zelle | Genom ist erster Test (erste Messung → Messreihe) | „immer zuerst“ (% mehr) | „nie“ (% mehr) |
|---|---|---|---|
| Ausbeute 0,25, Kosten 1,0 | 43 % → 45 % | 1,1 → 1,2 | 7,9 → 8,2 |
| Ausbeute 0,25, Kosten 2,0 | 0 % → 0 % | 13,2 → 13,9 | 0,7 → 0,6 |
| Ausbeute 0,50, Kosten 1,0 | 97 % → 96 % | 0,1 → 0,0 | 46,5 → 46,2 |
| Ausbeute 0,50, Kosten 2,0 | 42 % → 46 % | 3,0 → 2,4 | 18,9 → 18,3 |
| Ausbeute 0,50, Kosten 4,0 | 5 % → 0 % | 25,9 → 26,1 | 1,7 → 1,4 |

- **„Keine der drei Faustregeln liegt unter 5 %“ gilt nicht.** In 3 von 11 binären Zellen liegt die beste einfache Regel unter 5 % (Zellen q0, q4, q6: 1,8, 1,8, 4,8 %). Die erste Messung nannte nur Zellen mit Rauschen und mittlerem Irrtumspreis.
- **Kurzsichtig gegen Info/Kosten im Standard (Se/Sp 0,9/0,9, L 20):** erste Messung −8,1 ± 1,0, jetzt −5,4 ± 0,9; das Vorzeichen bleibt, der Abstand ist kleiner. Bei Se/Sp 0,8/0,9 liegt der Abstand jetzt bei −4,2 ± 1,2 (erste Messung −1,9 ± 1,1, damals innerhalb von zwei Standardfehlern); bei sehr schiefem Vorab (Zipf 1,5) bei +0,4 ± 2,0 (erste Messung −1,9 ± 2,2): dort bleibt es offen.
- **Schwelle der Meldung „verfehlt deutlich“:** Ursprünglich waren 15 % vorgesehen. Die größte Lücke der besten einfachen Regel in der Messreihe ist 13,4 %, der Zustand wäre nie erschienen; die Schwelle steht darum bei 10 % (und „reicht“ unter 3 %). Das ist eine Setzung nach dem Blick auf die Messung, keine Messung. Sie liegt aber in einer Lücke der Messwerte und nicht an einem Grenzfall: In den 14 Zellen mit Regelvergleich gibt jede Schwelle von 2,7 bis 4,7 % („reicht“) beziehungsweise von 8,7 bis 12,2 % („verfehlt deutlich“) dieselbe Einteilung; die Lücken gehören zu diesen Zellen und können sich bei einer Neumessung verschieben (der Test `test_the_thresholds_sit_inside_gaps_of_the_cell_values` fängt das).
- **Rechenzeit:** Sie schwankt stark mit der Last der Maschine (die erste Messung auf ruhiger Maschine und der Bau unter Last lagen weit auseinander). Zeiten sind kein Befund und stehen in keinem Test.

## Modell

Jede Instanz hat 4, 6 oder 10 Diagnosen (Standard 6) mit einem Zipf-Vorab (gleichverteilt, schief mit Exponent 1,0 oder sehr schief mit 1,5), bis zu 8 einfache Tests und einen Irrtumspreis L in mittleren Testkosten. Test t spricht jede Diagnose unabhängig mit Wahrscheinlichkeit 0,5 an; er fällt bei einer angesprochenen Diagnose mit Wahrscheinlichkeit Se positiv aus, sonst mit 1 − Sp; Kosten lognormal (σ 1,0), im Mittel 1. Jeder Test wird höchstens einmal gemacht. Am Ende wird die wahrscheinlichste Diagnose genannt; ein Irrtum kostet L. **Gesamtkosten = Testkosten + L · P(Irrtum).** Der **Zustand** merkt sich je Test, ob er gemacht wurde und mit welchem Ausgang; daraus folgt die Wahrscheinlichkeit jeder Diagnose nach Bayes (Tests unabhängig gegeben die Diagnose).

- **Unklar:** dritter Ausgang jedes einfachen Tests mit Wahrscheinlichkeit u (0; 0,1; 0,3), unabhängig von der Diagnose.
- **Genomtest:** n + 1 Ausgänge; mit Wahrscheinlichkeit y (Ausbeute 0,25 oder 0,5) wird die Diagnose eindeutig gefunden, sonst „kein Befund“ (das ändert die Wahrscheinlichkeiten der Diagnosen nicht; der Test gilt danach als gemacht). Kosten in mittleren Testkosten (1, 2, 4, 8, in der Messreihe auch 16).
- **Rechengrenzen:** höchstens 8 einfache Tests (3^m Zustände), mit „unklar“ höchstens 7 (4^m), mit Genomtest höchstens 6 einfache Tests; die App sagt es, wenn sie weniger rechnet als eingestellt.
- Vollständige Formeln: Expander **📐 Mathematische Formulierung** der App. Alle Parameter sind angenommen.

## Methodik

| Strategie | Wie sie arbeitet | Grenze |
|---|---|---|
| **Optimum (exakt)** | Rückwärtsrechnung über alle Zustände: f(s) = min(Verlust bei Stopp, min über freie Tests von Kosten plus Erwartung über die Ausgänge); `dtf_solver.py` | 3^m Zustände, deshalb höchstens 8 Tests |
| **kurzsichtig** | noch ein Test, wenn der erwartete Verlust nach *einem* Test plus dessen Kosten unter dem jetzigen liegt | sieht nicht, dass erst zwei Tests zusammen die Entscheidung kippen |
| **Info/Kosten** | der Test mit dem größten Entropiegewinn je Kosten; Stopp bei Wahrscheinlichkeit der besten Diagnose ≥ τ | die Regel benutzt L nicht; τ wird je Zelle abgestimmt und folgt so indirekt dem Irrtumspreis (in der Messreihe 0,60 / 0,80 / 0,90 bei L 5 / 20 / 100) |
| **billigster nützlicher Test** | der billigste Test mit Informationsgewinn; derselbe Stopp | wie Info/Kosten, ignoriert die Höhe der Information (verlangt nur, dass sie positiv ist) |

τ (0,6 bis 0,99) wurde je gemessener Zelle auf den ersten 60 Instanzen abgestimmt und auf den übrigen gemessen. Eine feste Schwelle 0,95 statt der abgestimmten hätte im Standard Info/Kosten von 12,1 ± 0,9 % auf 44,3 % und den billigsten nützlichen Test von 23,4 ± 1,3 % auf 55,9 % angehoben; im Mittel über die 11 binären Zellen liegt Info/Kosten mit τ = 0,95 um 34,4 Punkte und der billigste nützliche Test um 36,3 Punkte höher als mit der abgestimmten Schwelle (Mittel der Zellmittel). Alle Regeln werden **exakt** ausgewertet (über alle erreichbaren Zustände), nicht simuliert. Zufall nur über den ganzzahligen SplitMix64 mit `decimal`-Ersatz für Logarithmus, Kosinus und Exponentialfunktion (`dtf_rng.py`; die Suite lief unter Windows und im Linux-Docker-Lauf grün, Neu-Rechnen-Tests als Band); Messreihe: `tools/sweep.py` (Seeds ab 100000, die Anzeige-Seeds der App sind 0 bis 299).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Messreihe `data/dtf_results.json` (120 Instanzen je Zelle; Einheit: **Instanz**; Lücke = 100 · (Kosten der Regel / Kosten des Optimums − 1), Mittel ± Standardfehler über die letzten 60 Instanzen (Regelzellen) beziehungsweise über alle 120 (Genom-Zellen und die Zeile „Optimum (alle Instanzen)“ der Tabelle „unklar“); Optimum: Mittel über dieselben Instanzen in mittleren Testkosten). Jede Tabelle wird beim Erzeugen der README aus der Datei geschrieben (`tools/make_readme.py`) und von `tests/test_claims.py` auf Gleichheit geprüft.

**Faustregeln gegen das Optimum** (jede Zeile nennt ihre Zelle; Se/Sp: Sensitivität/Spezifität, L: Irrtumspreis in mittleren Testkosten):

| Zelle | Optimum | kurzsichtig (%) | Info/Kosten (%) | billigster nützlicher Test (%) | kurzsichtig minus Info/Kosten | τ (Info/Kosten / billigster) |
|---|---|---|---|---|---|---|
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 1,0/1,0, L 20 | 1,06 | 3,4 ± 0,6 | 1,8 ± 0,4 | 3,6 ± 0,7 | +1,6 ± 0,5 | 0,90 / 0,90 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,95/0,95, L 20 | 2,86 | 6,0 ± 0,5 | 15,1 ± 1,5 | 32,1 ± 1,8 | −9,1 ± 1,7 | 0,90 / 0,90 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 20 | 4,45 | 6,7 ± 0,6 | 12,1 ± 0,9 | 23,4 ± 1,3 | −5,4 ± 0,9 | 0,80 / 0,80 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,8/0,9, L 20 | 6,04 | 8,6 ± 0,9 | 12,9 ± 1,0 | 19,2 ± 0,8 | −4,2 ± 1,2 | 0,70 / 0,70 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 5 | 2,01 | 1,8 ± 0,3 | 12,8 ± 2,4 | 20,0 ± 2,6 | −11,0 ± 2,4 | 0,60 / 0,60 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 100 | 13,23 | 21,8 ± 1,8 | 12,2 ± 0,7 | 15,1 ± 0,7 | +9,6 ± 1,5 | 0,90 / 0,90 |
| 6 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,7/0,8, L 20 | 9,46 | 4,8 ± 0,6 | 12,2 ± 0,6 | 15,3 ± 0,6 | −7,5 ± 0,9 | 0,60 / 0,60 |
| 4 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 20 | 3,36 | 7,3 ± 0,9 | 13,4 ± 1,4 | 25,3 ± 1,5 | −6,1 ± 1,3 | 0,80 / 0,80 |
| 6 Diagnosen, 8 Tests, Vorab gleich, Se/Sp 0,9/0,9, L 20 | 5,20 | 22,2 ± 1,5 | 13,3 ± 1,0 | 19,8 ± 0,8 | +8,9 ± 1,7 | 0,80 / 0,80 |
| 6 Diagnosen, 8 Tests, Vorab sehr schief, Se/Sp 0,9/0,9, L 20 | 3,78 | 13,8 ± 1,7 | 13,4 ± 1,6 | 25,4 ± 1,9 | +0,4 ± 2,0 | 0,80 / 0,80 |
| 10 Diagnosen, 8 Tests, Vorab schief, Se/Sp 0,9/0,9, L 20 | 5,80 | 6,6 ± 0,5 | 14,9 ± 1,2 | 21,1 ± 1,1 | −8,3 ± 1,2 | 0,70 / 0,70 |

- **Im Standard (Se/Sp 0,9/0,9, L 20)** liegt kurzsichtig 6,7 ± 0,6 %, Info/Kosten 12,1 ± 0,9 % und der billigste nützliche Test 23,4 ± 1,3 % über dem Optimum von 4,45 (im Mittel 3,6 Tests, Irrtumswahrscheinlichkeit 12,4 %).
- **Rauschfrei (Se/Sp 1,0/1,0)** liegt Info/Kosten bei 1,8 ± 0,4 %; schon bei 0,95/0,95 sind es 15,1 ± 1,5 %. In den 10 Zellen mit Se/Sp unter 1 (sie unterscheiden sich zusätzlich im Irrtumspreis, im Vorab oder in der Zahl der Diagnosen) liegt Info/Kosten zwischen 12,1 und 15,1 %.
- **Welche Regel die kleinere Lücke hat, hängt an den Einstellungen.** Kurzsichtig hat in 7 der 11 Zellen eine um mehr als zwei Standardfehler kleinere Lücke als Info/Kosten, Info/Kosten in 3 (Se/Sp 1,0/1,0; Irrtumspreis 100; Vorab gleichverteilt); in 1 Zelle ist der Unterschied offen. Bei Irrtumspreis 100 liegt kurzsichtig bei 21,8 ± 1,8 % und Info/Kosten bei 12,2 ± 0,7 % (Differenz +9,6 ± 1,5); bei gleichverteiltem Vorab 22,2 ± 1,5 % gegen 13,3 ± 1,0 % (Differenz +8,9 ± 1,7).
- **Der billigste nützliche Test** hat in 11 von 11 Zellen eine um mehr als zwei Standardfehler größere Lücke als Info/Kosten (gepaart auf denselben Instanzen) und in 8 Zellen eine größere Lücke als kurzsichtig; gegenüber kurzsichtig ist seine Lücke in 1 Zelle kleiner und in 2 Zellen ist der Unterschied offen (nach den Mittelwerten allein ist sie in 11 beziehungsweise 9 Zellen größer).
- **Meldung der App** (Lücke der besten einfachen Regel in der Zelle): „reicht“ (unter 3 %) in 2 Zellen, „kostet merklich“ in 6, „verfehlt deutlich“ (über 10 %) in 3.

**Ergebnis „unklar“** (6 Diagnosen, 6 Tests, Vorab schief, Se/Sp 0,9/0,9, L 20):

| Anteil unklar | Optimum (alle Instanzen) | Zuwachs | kurzsichtig (%) | Info/Kosten (%) | billigster nützlicher Test (%) |
|---|---|---|---|---|---|
| 0,00 | 5,45 | 0,0 % | 2,6 ± 0,3 | 10,2 ± 0,9 | 18,1 ± 1,2 |
| 0,10 | 6,12 | 12,2 % | 1,8 ± 0,2 | 11,3 ± 0,9 | 15,9 ± 1,0 |
| 0,30 | 7,71 | 41,3 % | 0,9 ± 0,1 | 12,7 ± 0,9 | 14,3 ± 0,9 |

- Der Optimalwert steigt bei u = 0,1 um rund 12,2 % und bei u = 0,3 um rund 41,3 % (Mittel über alle 120 Instanzen; **ungepaart**: die drei Zellen nutzen verschiedene Instanzen, der Standardfehler des Zuwachses liegt nach der Delta-Methode für das Verhältnis ungepaarter Mittel bei etwa ± 1,9 beziehungsweise ± 2,1 Punkten).
- Kurzsichtig hat in 3 der drei Zellen eine um mehr als zwei Standardfehler kleinere Lücke als Info/Kosten (gepaart) und nach den Mittelwerten in 3 die kleinste Lücke aller drei Regeln; Info/Kosten hat gegenüber dem billigsten nützlichen Test in 0 der drei Zellen eine um mehr als zwei Standardfehler kleinere Lücke, in 1 Zelle ist der Unterschied offen.

**Genomtest zuerst** (6 einfache Tests, 6 Diagnosen, Se/Sp 0,9/0,9, L 20; Mehrkosten in Prozent über dem Optimum der Instanz):

| Ausbeute y | Genomkosten | Genom ist erster Test | „immer zuerst“ (% mehr) | „nie“ (% mehr) | Optimum ohne Genom | Optimum mit Genom |
|---|---|---|---|---|---|---|
| 0,25 | 1,0 | 45 % | 1,2 ± 0,1 | 8,2 ± 0,2 | 5,52 | 5,09 |
| 0,25 | 2,0 | 0 % | 13,9 ± 0,5 | 0,6 ± 0,1 | 5,35 | 5,32 |
| 0,25 | 4,0 | 0 % | 48,8 ± 1,1 | 0,0 ± 0,0 | 5,58 | 5,58 |
| 0,25 | 8,0 | 0 % | 123,1 ± 2,3 | 0,0 ± 0,0 | 5,56 | 5,56 |
| 0,25 | 16,0 | 0 % | 275,0 ± 4,1 | 0,0 ± 0,0 | 5,45 | 5,45 |
| 0,50 | 1,0 | 96 % | 0,0 ± 0,0 | 46,2 ± 0,6 | 5,53 | 3,76 |
| 0,50 | 2,0 | 46 % | 2,4 ± 0,3 | 18,3 ± 0,5 | 5,55 | 4,67 |
| 0,50 | 4,0 | 0 % | 26,1 ± 1,1 | 1,4 ± 0,1 | 5,51 | 5,43 |
| 0,50 | 8,0 | 0 % | 101,1 ± 2,0 | 0,0 ± 0,0 | 5,42 | 5,42 |
| 0,50 | 16,0 | 0 % | 246,0 ± 3,7 | 0,0 ± 0,0 | 5,51 | 5,51 |

- Bei Ausbeute 0,50 ist der Genomtest bei Kosten 1 in 96 %, bei 2 in 46 %, bei 4 in 0 % der Instanzen der erste Test; bei Ausbeute 0,25 bei Kosten 1 in 45 %, bei 2 in 0 %. **Der Wechsel (der Anteil fällt auf 0 %) liegt bei Ausbeute 0,25 zwischen den Kosten 1 und 2, bei Ausbeute 0,50 zwischen den Kosten 2 und 4 mittleren Testkosten.** An Stufen mit Anteilen zwischen 20 und 80 % (Ausbeute 0,25, Kosten 1: 45 %; Ausbeute 0,50, Kosten 2: 46 %) entscheidet die Instanz, nicht die Kostenstufe. Zwischen den Stufen ist nichts gemessen.
- „Genom nie“ kostet bei Ausbeute 0,50 und Kosten 1 46,2 % mehr, „Genom immer zuerst“ bei Kosten 4 26,1 %: Eine einfache Politik am falschen Ende der Kostenskala ist teuer.

**Basisratenfalle** (Sensitivität 0,99, Falsch-positiv-Rate 0,01 je Test; gemeinsamer Fehler: 2 % der Gesunden tragen einen Störfaktor, der jeden Test mit Wahrscheinlichkeit 0,4 positiv macht; Wahrscheinlichkeit, krank zu sein; **die Parameter des Störfaktors sind angenommen, nicht gemessen**):

| Anteil Kranke | 1 positiver Test | 2 positive Tests (unabhängig / gemeinsamer Fehler) | 3 positive Tests (unabhängig / gemeinsamer Fehler) |
|---|---|---|---|
| 1 von 10.000 | 0,010 | 0,495 / 0,030 | 0,990 / 0,070 |
| 1 von 1.000 | 0,090 | 0,907 / 0,234 | 0,999 / 0,431 |
| 1 von 100 | 0,500 | 0,990 / 0,755 | 1,000 / 0,884 |
| 1 von 20 | 0,839 | 0,998 / 0,942 | 1,000 / 0,976 |

- Bei 1 von 1.000 reichen unter der Unabhängigkeitsannahme 2 positive Tests für eine Wahrscheinlichkeit von mindestens 0,9 (0,91 nach zwei), mit gemeinsamem Fehler sind es 6 (nach zwei Tests 0,23). Nach **einem** positiven Test sind beide Rechnungen gleich.

## Ehrliche Grenzen

- **Die Messreihe gilt für diese Welt:** 4, 6 oder 10 Diagnosen, bis zu 8 einfache Tests, die gemessenen Reglerstufen; zwischen den Stufen und für andere Kombinationen zeigt die App die nächstliegende gemessene Zelle und nennt die abweichenden Parameter.
- **Die Regeln gelten nur mit ihrer abgestimmten Schwelle** und nur für die genannten drei; keine Aussage über die Methodenklasse. Die Stoppschwelle wurde aus sechs Werten gewählt; ein feineres Gitter könnte die Lücken verschieben.
- **Unterschiede unter zwei Standardfehlern sind nicht zu deuten:** kurzsichtig gegen Info/Kosten in 1 Zelle (Vorab sehr schief), der billigste nützliche Test gegen kurzsichtig in 2 Zellen (rauschfrei und gleichverteiltes Vorab).
- **Nicht gerechnet:** Wiederholung desselben Tests, gemeinsame Fehlerursachen zwischen Tests (außer im Rechner der Basisratenfalle), Reihenfolgebedingungen, Dauer und Belastung, ein Irrtumspreis je Diagnosenpaar, mehr als zwei Ausgänge außer „unklar“ und Genom, mehr als 8 Tests, Gruppentests (Mischproben bei vielen Personen; hier geht es um eine Person und mehrere Tests); die Genom-Ausbeute ist für alle Diagnosen gleich.
- **Nur an erfundenen Instanzen geprüft;** keine echten Tests, keine echten Daten, keine Literaturzahlen als Beleg.

## Verwandte Demos mit demselben mathematischen Modell

- [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo): ein Test, Bayes-Update im umgedrehten Baum, Wert der Stichprobeninformation, n gleichartige unabhängige Tests; dort ausdrücklich nicht gerechnet: sequentielle und abhängige Tests, verschiedenartige Tests, mehr als zwei Zustände. Dasselbe Bayes-Update, hier mit Folge und Stoppen.
- [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo): aus einer Wahrscheinlichkeit eine Entscheidung über das Kostenverhältnis (Schwelle 1/(1+r)). Hier ist die Stopp-Entscheidung derselbe Vergleich von Stopp-Verlust und erwartetem Verlust beim Weitermachen; sie ergibt keine feste Schwelle auf die Wahrscheinlichkeit der besten Diagnose, sondern hängt vom ganzen Zustand ab.
- [`entscheidung-verfahrenswahl-demo`](https://github.com/sebastian-hanisch/entscheidung-verfahrenswahl-demo): vergleicht **Rechenverfahren** (Entscheidungsbaum, Zustandsrechnung, Szenario-LP, gemischt-ganzzahliges Modell) für ein gegebenes Entscheidungsmodell am Kapazitätsfall mit zwei Zuständen; dort darf nach jedem von bis zu vier gleichartigen Tests (gleiche Trefferquote, gleiche Kosten) gestoppt werden. Hier geht es um das **Entscheidungsmodell** selbst: mehrere verschiedenartige Tests mit eigenen Kosten und Fehlern, mehr als zwei Diagnosen, einfache Regeln gegen die exakte Strategie, ein breiter Test, der Ausgang „unklar“ und die Basisratenfalle. Beide rechnen das Stoppen mit einer Rückwärtsrechnung über Zustände; die Zahlen sind nicht vergleichbar.
- [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo): Entscheidungsbaum, Strategie und Rollback; die exakte Strategie hier ist ein Rollback über die Zustandsmenge.
- [`basisratenfalle-demo`](https://github.com/sebastian-hanisch/basisratenfalle-demo): die Basisratenfalle als eigene Demo (Bevölkerung als Zählung, exakte Rechnung, gemeinsamer Fehler gegen unabhängige Tests, Zuweisungsfilter); hier ist sie nur ein Rechner neben der Testfolge.

## Verifikation (251 Tests)

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
| `tools/sweep.py`, `tools/report.py`, `tools/make_readme.py` | Messreihe reproduzieren, berichten (mit den Zahlen der ersten Messung), README erzeugen |
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
venv\Scripts\activate           # Linux/macOS: source venv/bin/activate

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

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Erkennen unter Unsicherheit: Hinweis und Ursache](https://sebastianhanisch.net/konzepte-erkennen-unsicherheit.html).
