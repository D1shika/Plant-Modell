"""Heuristic visual-damage severity (NOT expert ground truth; does not establish real-world severity accuracy).

1. Segment the leaf with GrabCut (central-rectangle init) on an image resized to max side 256.
   If the GrabCut leaf mask covers < 5% of the frame (failure), fall back to an HSV mask: broad-green pixels,
   morphologically closed so lesions inside the leaf are included, largest connected component. Flag "hsv_fallback".
2. Low-saturation (S < 40) non-green pixels are removed from the leaf mask (cast shadows / gray backdrop), then the mask
   is eroded (5x5, 2 iterations) to remove stem/edge rims that inflated damage in the pilot.
3. Within the leaf, "healthy green" = HSV hue 35-85 (OpenCV 0-179), saturation >= 40, value >= 40.
4. damage_fraction = non_green_leaf_pixels / total_leaf_pixels
5. Mild < 0.10 <= Moderate <= 0.30 < Severe. Healthy leaves get severity "None".
severity_valid = True only when the GrabCut mask is ok (5%..97% of frame); fallback/failed masks are flagged and
excluded from severity training and evaluation.

Usage: python src/severity.py --index data/processed/index.csv --out data/processed/severity.csv [--debug 24] [--force]
"""
import argparse
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS, ROOT, SEED, SEVERITY_THRESHOLDS  # noqa: E402
from log_util import log  # noqa: E402

MAX_SIDE = 256
GREEN_H = (35, 85)
MIN_SV = 40
ERODE = np.ones((5, 5), np.uint8)


def severity_class(frac):
    lo, hi = SEVERITY_THRESHOLDS
    return "Mild" if frac < lo else ("Moderate" if frac <= hi else "Severe")


def _hsv_fallback(hsv):
    broad = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= MIN_SV) & (hsv[..., 2] >= MIN_SV)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n <= 1:
        return np.zeros(broad.shape, bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == biggest


def analyse(path):
    cv2.setNumThreads(1)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * .05), int(h * .05), int(w * .9), int(h * .9))
    try:
        cv2.grabCut(img, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
        leaf = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        leaf = np.zeros((h, w), bool)
    share = leaf.sum() / (h * w)
    if share < 0.05:
        leaf, flag = _hsv_fallback(hsv), "hsv_fallback"
    elif share > 0.97:
        flag = "leaf_fills_frame"
    else:
        flag = "ok"
    green = (hsv[..., 0] >= GREEN_H[0]) & (hsv[..., 0] <= GREEN_H[1]) & (hsv[..., 1] >= MIN_SV) & (hsv[..., 2] >= MIN_SV)
    # low-saturation gray pixels (cast shadows, backdrop) are treated as background, not leaf or damage
    leaf = leaf & (green | (hsv[..., 1] >= MIN_SV))
    leaf = cv2.erode(leaf.astype(np.uint8), ERODE, iterations=2).astype(bool)
    damage = leaf & ~green
    area = int(leaf.sum())
    if area < 0.02 * h * w:
        return img, leaf, damage, float("nan"), area / (h * w), "mask_failed"
    return img, leaf, damage, float(damage.sum() / area), area / (h * w), flag


def _worker(p):
    _, _, _, frac, share, flag = analyse(ROOT / p)
    return frac, share, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(ROOT / a.index)
    un = df[df["health"] == "Unhealthy"].reset_index(drop=True)
    out = ROOT / a.out
    if out.exists() and not a.force and len(pd.read_csv(out)) == len(un):
        print("severity output already exists with matching row count; use --force to recompute")
        un = pd.read_csv(out)
    else:
        with Pool(a.workers) as pool:
            recs = []
            for i, r in enumerate(pool.imap(_worker, list(un["path"]), chunksize=16), 1):
                recs.append(r)
                if i % 2000 == 0:
                    print(f"  {i}/{len(un)}", flush=True)
                    log(f"severity progress {i}/{len(un)}")
        un["damage_fraction"] = [r[0] for r in recs]
        un["leaf_share"] = [r[1] for r in recs]
        un["mask_flag"] = [r[2] for r in recs]
        un["severity"] = [severity_class(f) if f == f else "Invalid" for f in un["damage_fraction"]]
        un["severity_valid"] = (un["mask_flag"] == "ok") & (un["severity"] != "Invalid")
        un.to_csv(out, index=False)
    val = un[un["severity_valid"]]
    cnt = Counter(un["severity"])
    vcnt = Counter(val["severity"])
    print("unhealthy images:", len(un), "| severity_valid:", len(val))
    print("severity counts (all):", dict(cnt))
    print("severity counts (valid only):", dict(vcnt), {k: round(v / max(len(val), 1), 3) for k, v in vcnt.items()})
    for k in ("Mild", "Moderate", "Severe"):
        if vcnt.get(k, 0) / max(len(val), 1) < 0.05:
            print(f"WARNING: severity class {k} has < 5% of valid unhealthy examples")
    print("mask quality:", dict(Counter(un["mask_flag"])))
    print("by source/mask_flag:\n", pd.crosstab(un["source"], un["mask_flag"]))
    print("by source/severity (valid only):\n", pd.crosstab(val["source"], val["severity"]))
    d = OUTPUTS / "severity_debug"
    d.mkdir(parents=True, exist_ok=True)
    idx = random.Random(SEED).sample(range(len(un)), min(a.debug, len(un)))
    for k, i in enumerate(sorted(idx)):
        img, leaf, damage, frac, share, flag = analyse(ROOT / un.loc[i, "path"])
        lm = cv2.cvtColor((leaf * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        dm = img.copy()
        dm[damage] = (0, 0, 255)
        panel = np.hstack([img, lm, dm])
        sev = severity_class(frac) if frac == frac else "Invalid"
        label = f"{un.loc[i, 'condition'][:22]}|{un.loc[i, 'source'][:5]}|{frac:.2f} {sev}|{flag}"
        cv2.putText(panel, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(panel, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        ok, buf = cv2.imencode(".png", panel)
        (d / f"debug_{k:02d}.png").write_bytes(buf.tobytes())
    print(f"wrote {len(idx)} debug images to", d)
    log(f"severity {a.index}: unhealthy={len(un)} valid={len(val)} counts_valid={dict(vcnt)} flags={dict(Counter(un['mask_flag']))}")


if __name__ == "__main__":
    main()
