"""Aggregate the multi-seed repeat of one variant (default: image-only) on the TEST split.

Runs: the original seed-42 run (models/<variant>) plus models/<variant>_seed<N> for each finished seed. The data split is
fixed (seed 42); only training randomness (head initialisation, batch order, augmentation) changes between runs.
Per-run cells show the point estimate with its 95% group-bootstrap CI (test-sampling uncertainty of that fixed model).
The summary column shows mean +/- sample sd and min-max of the point estimates across runs (training randomness); with
only a handful of runs the sd itself is imprecise.

Usage: python src/seed_repeat.py [--variant image] [--seeds 1 2 3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset import build_table  # noqa: E402
from evaluate import acc_stat, collect, elig_corr, fmt, save_table, sev_f1_stat  # noqa: E402
from infer import get_device  # noqa: E402
from log_util import log  # noqa: E402


def metrics(P):
    ec = elig_corr(P)
    g = P["group_id"].values
    lab = (P["acquisition"] == "lab").values
    fld = ~lab
    unh = (P["health"] == "Unhealthy").values
    common = sorted(set(P.loc[lab & unh, "condition"]) & set(P.loc[fld & unh, "condition"]))
    inc = P["condition"].isin(common).values & unh
    out = {}

    def add(name, head, mask=None):
        e, c = ec[head]
        out[name] = acc_stat(g, e if mask is None else e & mask, c)[:4]

    add("Crop accuracy (all)", "crop")
    add("Health accuracy (all)", "health")
    add("Condition accuracy (all)", "condition")
    add("Condition accuracy (lab)", "condition", lab)
    add("Condition accuracy (field)", "condition", fld)
    add("Health accuracy (field)", "health", fld)
    add("Like-for-like biotic condition acc (lab)", "condition", lab & inc)
    add("Like-for-like biotic condition acc (field)", "condition", fld & inc)
    out["Severity macro-F1 (heuristic labels)"] = sev_f1_stat(P)[:4]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="image")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    a = ap.parse_args()
    dev = get_device()
    df, maps = build_table()
    stats = json.load(open(C.MODELS / "env_stats.json"))
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    runs = [(a.variant, f"{a.variant} seed 42 (original)")]
    runs += [(f"{a.variant}_seed{s}", f"seed {s}") for s in a.seeds]
    results, labels, epochs = [], [], {}
    for run, label in runs:
        if not (C.MODELS / run / "DONE.json").exists():
            print(f"skip {run}: training not complete")
            continue
        P, info = collect(a.variant, test_df, maps, stats, dev, a.workers, run=run)
        results.append(metrics(P))
        labels.append(label)
        epochs[label] = info["epoch"]
        print(f"{run}: evaluated (best epoch {info['epoch']})", flush=True)
    if len(results) < 2:
        sys.exit("need at least two finished runs")
    rows = []
    for m in results[0]:
        pts = np.array([r[m][0] for r in results])
        row = {"Metric": m, **{lab: fmt(*r[m]) for lab, r in zip(labels, results)},
               f"Mean +/- sd over {len(results)} runs": f"{pts.mean():.3f} +/- {pts.std(ddof=1):.3f}",
               "Min - max": f"{pts.min():.3f} - {pts.max():.3f}"}
        rows.append(row)
    out = pd.DataFrame(rows)
    save_table(out, f"seed_repeat_{a.variant}", f"Multi-seed repeat of the {a.variant} model (test split)",
               "Data split fixed (seed 42); only training randomness differs. Per-run intervals = 95% group-bootstrap CI "
               "(test sampling); the summary columns show variability across training runs. Best epochs: "
               + ", ".join(f"{k}: {v}" for k, v in epochs.items()) + ".")
    log(f"seed_repeat {a.variant}: {len(results)} runs evaluated; outputs/tables/seed_repeat_{a.variant}.md")


if __name__ == "__main__":
    main()
