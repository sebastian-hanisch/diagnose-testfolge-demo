"""Berichtet data/dtf_results.json: je Zelle die Lücken der Regeln, daneben (Vergleichsspalten „erste Messung“) die Zahlen der ersten Messung mit dem Zufallsgenerator von NumPy.

    python tools/report.py

Die Zahlen der ersten Messung stammen aus einem früheren Lauf (andere Instanzen, 120 beziehungsweise 60 Instanzen);
sie dienen nur dem ehrlichen Vergleich in der README („Befunde und Korrekturen“), nicht der App."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Zelle -> (Optimum, kurzsichtig, Info/Kosten, billigster) der ersten Messung; für die Zellen u* ist das Optimum das Mittel aller 60 Instanzen
VORMESSUNG_REGELN = {
    "q0": (1.13, 2.7, 1.4, 4.0), "q1": (3.13, 6.2, 14.6, 34.1), "q2": (4.43, 5.0, 13.2, 24.1), "q3": (6.20, 10.0, 11.9, 18.5), "q4": (2.17, 0.8, 10.8, 20.1),
    "q5": (12.20, 25.0, 12.8, 15.3), "q6": (9.34, 5.4, 12.6, 16.2), "g0": (3.19, 8.2, 15.1, 23.9), "g1": (5.23, 22.6, 14.4, 21.2), "g2": (3.69, 17.1, 19.0, 27.4),
    "g3": (5.93, 6.6, 13.1, 21.0), "u0": (5.31, 2.6, 14.6, 21.2), "u1": (5.95, 2.2, 12.8, 18.4), "u2": (7.44, 1.0, 14.2, 16.8),
}
# Genom-Zelle -> (Anteil erster Test in %, „immer zuerst“ in %, „nie“ in %)
VORMESSUNG_GENOM = {
    "y25c1": (43, 1.1, 7.9), "y25c2": (0, 13.2, 0.7), "y25c4": (0, 49.9, 0.0), "y25c8": (0, 124.8, 0.0), "y25c16": (0, 274.6, 0.0),
    "y50c1": (97, 0.1, 46.5), "y50c2": (42, 3.0, 18.9), "y50c4": (5, 25.9, 1.7), "y50c8": (0, 97.8, 0.0), "y50c16": (0, 245.6, 0.0),
}


def main():
    data = json.loads((ROOT / "data" / "dtf_results.json").read_text(encoding="utf-8"))
    cells = data["cells"]
    print(f"{data['meta']['n_instances']} Instanzen je Zelle, τ auf den ersten {data['meta']['n_tune']} abgestimmt, Lücke auf den übrigen; Seeds ab {data['meta']['seed_base']}")
    print("Regeln: Optimum | kurzsichtig | Info/Kosten | billigster | τ(G/B) | kurzsichtig minus Info/Kosten   [erste Messung in Klammern]")
    for cid, old in VORMESSUNG_REGELN.items():
        c = cells[cid]
        opt = c["opt_mean_all"] if cid.startswith("u") else c["opt_mean"]
        print(f"{cid:3} {opt:6.2f} ({old[0]:5.2f}) | " + " | ".join(f"{c[r]['mean']:5.1f}±{c[r]['se']:3.1f} ({o:4.1f})" for r, o in zip("MGB", old[1:]))
              + f" | τ {c['tau']['G']}/{c['tau']['B']} | {c['diff_MG']['mean']:+5.1f}±{c['diff_MG']['se']:3.1f}")
    print("Genom: Anteil erster Test % | immer zuerst % | nie %   [erste Messung in Klammern]")
    for cid, old in VORMESSUNG_GENOM.items():
        c = cells[cid]
        print(f"{cid:7} {c['share_first'] * 100:4.0f} ({old[0]:3d}) | {c['forced']['mean']:6.1f}±{c['forced']['se']:3.1f} ({old[1]:5.1f}) | {c['never']['mean']:5.1f}±{c['never']['se']:3.1f} ({old[2]:4.1f})"
              f" | opt ohne {c['opt_without_mean']:.2f} mit {c['opt_mean']:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
