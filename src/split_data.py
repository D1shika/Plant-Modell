"""Leakage-resistant 70/15/15 split (seed 42).

Grouping levels (merged with union-find):
  1. PlantVillage original-leaf id from leaf-map.json (PV:...)            -> grouping by original leaf
  2. PlantVillage original-image name fallback (PVK:...)                   -> grouping by original image only
  3. exact duplicates (identical file MD5)                                 -> exact duplicate prevention
  4. near-duplicates: perceptual hash (64-bit pHash), Hamming <= PHASH_T   -> approximate image grouping
Plant-level (physical plant) independence is NOT established by any of these.

Usage: python src/split_data.py --index data/processed/index_pilot.csv --out data/processed/splits_pilot.csv
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, SEED  # noqa: E402
from log_util import log  # noqa: E402

PHASH_T = 6  # Hamming distance threshold (of 64 bits); documented choice, conservative-ish


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bits(p):
    with Image.open(p) as im:
        return np.array(imagehash.phash(im.convert("RGB")).hash, dtype=bool).ravel()


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(ROOT / a.index).reset_index(drop=True)
    n = len(df)
    paths = [ROOT / p for p in df["path"]]
    uf = UF(n)

    # initial groups from provided ids
    first = {}
    for i, g in enumerate(df["group_id"].fillna("")):
        if g:
            if g in first:
                uf.union(first[g], i)
            else:
                first[g] = i
    n_id_merges = len({uf.find(i) for i in range(n)})

    # exact duplicates
    seen = {}
    exact = 0
    for i, p in enumerate(paths):
        h = md5(p)
        if h in seen:
            uf.union(seen[h], i)
            exact += 1
        else:
            seen[h] = i

    # perceptual near-duplicates (all images; O(n^2) in numpy, chunked)
    bits = np.stack([phash_bits(p) for p in paths]).astype(np.uint8)
    near = 0
    for s in range(0, n, 128):  # small chunks keep memory low (128 x n x 64 bytes)
        d = (bits[s:s + 128, None, :] != bits[None, :, :]).sum(-1)
        ii, jj = np.where(d <= PHASH_T)
        for i, j in zip(ii + s, jj):
            if j > i:
                uf.union(int(i), int(j))
                near += 1
    roots = [uf.find(i) for i in range(n)]
    df["final_group"] = [f"G{r}" for r in roots]
    ng = df["final_group"].nunique()
    print(f"images={n}  groups after id-merge={n_id_merges}  exact-duplicate pairs={exact}  "
          f"phash near-dup pairs (<= {PHASH_T})={near}  final groups={ng}")

    # stratified group split: 20 folds -> 14 train / 3 val / 3 test (~70/15/15), stratified by condition
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.full(n, -1)
    for k, (_, te) in enumerate(sgkf.split(df, df["condition"], df["final_group"])):
        fold[te] = k
    df["split"] = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))

    # assertions: no path or group in more than one split
    assert df["path"].is_unique
    assert (df.groupby("final_group")["split"].nunique() == 1).all(), "group leaks across splits"
    assert (df.groupby("path")["split"].nunique() == 1).all()
    print("ASSERT OK: no path and no group id appears in more than one split")

    print("\nsplit sizes:", dict(df["split"].value_counts()),
          {k: round(v / n, 3) for k, v in df["split"].value_counts().items()})
    print("\nper-source:\n", pd.crosstab(df["source"], df["split"]))
    ct = pd.crosstab(df["condition"], df["split"])
    print("\nper-condition:\n", ct.to_string())
    low = ct[(ct.get("val", 0) == 0) | (ct.get("test", 0) == 0)]
    if len(low):
        print("\nWARNING conditions with no val or no test samples:", list(low.index))
    out = ROOT / a.out
    df.drop(columns=["group_id"]).rename(columns={"final_group": "group_id"}).to_csv(out, index=False)
    print("wrote", out)
    log(f"split_data {a.index}: n={n} groups={ng} exact_dup_pairs={exact} phash_pairs={near} "
        f"splits={dict(df['split'].value_counts())} out={Path(a.out).name}")


if __name__ == "__main__":
    main()
