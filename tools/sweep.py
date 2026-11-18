"""Reproduziert die Messreihe: jede Zelle (dtf_constants.CELLS) mit N_INSTANCES Instanzen, schreibt data/dtf_results.json.

    python tools/sweep.py              # alle Zellen (mehrere Minuten, parallel); nicht in der CI
    python tools/sweep.py --only q2    # nur eine Zelle (Zwischenstand, schreibt nichts)

Die Seeds der Messreihe beginnen bei C.SWEEP_SEED_BASE und überschneiden sich nie mit den Anzeige-Seeds 0 bis 299 der App."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dtf_constants as C  # noqa: E402
import dtf_measure as M  # noqa: E402


def _task(args):
    cell_id, i = args
    cell = next(c for c in C.CELLS if c["id"] == cell_id)
    return cell_id, i, (M.measure_genome_instance(cell, i) if cell["kind"] == "genom" else M.measure_rules_instance(cell, i))


def run(cells, workers):
    tasks = [(c["id"], i) for c in cells for i in range(C.N_INSTANCES)]
    rows = {c["id"]: [None] * C.N_INSTANCES for c in cells}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (cid, i, r) in enumerate(ex.map(_task, tasks, chunksize=6)):
            rows[cid][i] = r
            if (k + 1) % 240 == 0:
                print(f"  {k + 1}/{len(tasks)} Instanzen, {time.time() - t0:.0f} s", flush=True)
    out = {}
    for c in cells:
        agg = M.aggregate_genome(rows[c["id"]]) if c["kind"] == "genom" else M.aggregate_rules(rows[c["id"]])
        out[c["id"]] = {"params": {k: v for k, v in c.items() if k != "id"}, **agg}
    return out


def _round(x):
    if isinstance(x, float):
        return round(x, 12)
    if isinstance(x, dict):
        return {str(k): _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="nur diese Zelle messen und ausgeben (schreibt data/dtf_results.json nicht)")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    args = ap.parse_args()
    cells = [c for c in C.CELLS if args.only in (None, c["id"])]
    res = run(cells, args.workers)
    if args.only:
        for cid, r in res.items():
            print(cid, json.dumps({k: v for k, v in _round(r).items() if k != "per_instance"}, ensure_ascii=False))
        return
    meta = {"n_instances": C.N_INSTANCES, "n_tune": C.N_TUNE, "taus": list(C.TAUS), "seed_base": C.SWEEP_SEED_BASE, "gap_mark": C.GAP_MARK}
    path = ROOT / "data" / "dtf_results.json"
    path.write_text(json.dumps({"meta": meta, "cells": _round(res)}, ensure_ascii=False, indent=None, separators=(",", ":")), encoding="utf-8")
    print("geschrieben:", path, f"({path.stat().st_size / 1024:.0f} KB, {len(res)} Zellen)")


if __name__ == "__main__":
    main()
