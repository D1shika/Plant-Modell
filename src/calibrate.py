"""Validation-only temperature scaling for the uncertainty output of predict.py.  NOT RUN until approved.

Runs the chosen checkpoint once over the VALIDATION split (3,460 images; ~1-3 min on the RTX 3050, ~10-20 min on CPU,
<1 GB VRAM, no internet), saves the raw logits to outputs/repair/val_logits_<variant>.npz (new file), fits one
temperature per head (condition, crop, health) by minimising validation negative log-likelihood over a grid, and
reports expected calibration error (ECE) before/after for all / lab / field validation images.

It never reads the test split. With --write it also saves models/calibration/<variant>.json (new file; predict.py then
reports calibrated confidences). Thresholds stay at the provisional 0.5 unless --write is given together with
--threshold, and the report shows, on VALIDATION data, how many wrong predictions each candidate threshold would flag.
Caveat: validation is 94% lab images (211 field), so field calibration is weakly determined.

Usage: python -I src/calibrate.py [--variant image] [--write] [--threshold 0.6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from class_meta import CONDITION_META  # noqa: E402
from dataset import LeafDataset, build_table  # noqa: E402
from infer import get_device, load_variant, predict_loader  # noqa: E402
from reconcile import softmax  # noqa: E402

GRID = np.round(np.arange(0.25, 3.01, 0.01), 2)
OUT = C.OUTPUTS / "repair"


def nll(logits, y, t):
    p = softmax(logits, t)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(conf, ok, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="image")
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--write", action="store_true", help="save models/calibration/<variant>.json")
    ap.add_argument("--threshold", type=float, default=None, help="condition-confidence threshold to save with --write")
    a = ap.parse_args()
    dev = get_device()
    df, maps = build_table()
    stats = json.load(open(C.MODELS / "env_stats.json"))
    va = df[df["split"] == "val"].reset_index(drop=True)
    assert (va["split"] == "val").all()
    model, maps_ck, cfg = load_variant(a.variant, "best", dev)
    assert maps_ck["condition"] == maps["condition"], "label maps differ from checkpoint"
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"val_logits_{a.variant}.npz"
    if cache.exists():
        lg = dict(np.load(cache))
        print(f"using cached validation logits {cache}")
    else:
        ds = LeafDataset(va, maps, stats, train=False, load_image=(a.variant != "env"))
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)
        lg = {k: v.numpy() for k, v in predict_loader(model, dl, dev).items()}
        np.savez(cache, **lg)
    conds, crops = maps["condition"], maps["crop"]
    y = {"condition": va["condition"].map({c: i for i, c in enumerate(conds)}).values,
         "crop": va["crop"].map({c: i for i, c in enumerate(crops)}).values,
         "health": va["health"].map({"Healthy": 0, "Unhealthy": 1}).values}
    temps, lines = {}, [f"# Validation-only calibration ({a.variant}; n={len(va)}, field {int((va.acquisition == 'field').sum())})", ""]
    for h in ("condition", "crop", "health"):
        t = float(GRID[int(np.argmin([nll(lg[h], y[h], g) for g in GRID]))])
        temps[h] = t
        lines.append(f"## {h}: temperature {t:.2f} (NLL {nll(lg[h], y[h], 1.0):.4f} -> {nll(lg[h], y[h], t):.4f})")
        for scope, m in (("all", np.ones(len(va), bool)), ("lab", (va.acquisition == "lab").values),
                         ("field", (va.acquisition == "field").values)):
            p0, p1 = softmax(lg[h][m]), softmax(lg[h][m], t)
            ok = p0.argmax(1) == y[h][m]
            lines.append(f"- {scope} n={m.sum()}: acc {ok.mean():.3f}, ECE raw {ece(p0.max(1), ok):.3f} -> "
                         f"calibrated {ece(p1.max(1), ok):.3f}")
        lines.append("")
    # how many wrong (val) predictions a condition-confidence threshold would flag, after calibration
    pc = softmax(lg["condition"], temps["condition"])
    conf, ok = pc.max(1), pc.argmax(1) == y["condition"]
    fld = (va.acquisition == "field").values
    lines.append("## Condition-confidence threshold (calibrated, VALIDATION): share of wrong / correct predictions flagged")
    for th in (0.4, 0.5, 0.6, 0.7, 0.8):
        f = conf < th
        lines.append(f"- {th:.1f}: all wrong {int((f & ~ok).sum())}/{int((~ok).sum())}, correct flagged "
                     f"{int((f & ok).sum())}/{int(ok.sum())}; field wrong {int((f & ~ok & fld).sum())}/{int((~ok & fld).sum())}")
    report = "\n".join(lines) + "\n"
    (OUT / f"calibration_{a.variant}.md").write_text(report, encoding="utf-8")
    print(report)
    if a.write:
        cal = {"variant": a.variant, "fitted_on": "val", "n_val": int(len(va)), "temperature": temps,
               "method": "per-head temperature scaling, grid-search NLL on the validation split"}
        if a.threshold is not None:
            cal["thresholds"] = {"condition": a.threshold, "crop": 0.5, "health": 0.5}
        (C.MODELS / "calibration").mkdir(exist_ok=True)
        json.dump(cal, open(C.MODELS / "calibration" / f"{a.variant}.json", "w"), indent=2)
        print("wrote", C.MODELS / "calibration" / f"{a.variant}.json")


if __name__ == "__main__":
    main()
