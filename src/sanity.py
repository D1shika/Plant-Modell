"""Mandatory sanity checks (Part 12.1 and the duplicate part of 12.2). CPU only.

A. Background/source probe. Tiny classifiers see ONLY (i) a heavily blurred 16x16 RGB version of the image or
   (ii) the outer border strips (24 px of the 256 px cached image, centre removed). Targets: cause label (None vs
   Biotic; Abiotic does not exist in the data), source dataset, and (extra) condition. Trained on the train split,
   evaluated on the group-aware test split. If cause is predicted suspiciously well (balanced accuracy >= 0.70 or
   accuracy >= majority baseline + 0.10) the warning below is printed.
B. Cross-split near-duplicates: for every test image the Hamming distance (64-bit pHash of the 256 px cached image)
   to its nearest TRAIN image; plus an independent MD5 exact-duplicate check across splits on the original files.

Usage: python src/sanity.py
"""
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset import cache_path  # noqa: E402
from log_util import log  # noqa: E402

TAB = C.OUTPUTS / "tables"
WARN = "Cause accuracy may reflect dataset background or source differences rather than leaf symptoms."
BORDER = 24


def feats(rel):
    im = Image.open(cache_path(rel)).convert("RGB")
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(8)).resize((16, 16), Image.BOX), np.float32).ravel() / 255
    a = np.asarray(im, np.float32) / 255
    t, b = a[:BORDER], a[-BORDER:]
    l, r = a[:, :BORDER], a[:, -BORDER:]

    def small(x, shape):
        return np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize(shape, Image.BOX), np.float32).ravel() / 255

    border = np.concatenate([small(t, (32, 4)), small(b, (32, 4)), small(l, (4, 32)), small(r, (4, 32))])
    ph = np.asarray(imagehash.phash(im).hash, bool).ravel()
    return blur, border, ph


def md5(rel):
    h = hashlib.md5()
    with open(C.ROOT / rel, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(Xtr, Xte, ytr, yte, kind):
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=C.SEED)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
    else:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=300, C=0.1)
        clf.fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(Xte))
    maj = pd.Series(ytr).mode()[0]
    return (accuracy_score(yte, pred), balanced_accuracy_score(yte, pred), float((np.asarray(yte) == maj).mean()))


def main():
    df = pd.read_csv(C.DATA_PROCESSED / "splits.csv")
    df["cause"] = df["cause"].fillna("None")  # pandas parses the label string 'None' (healthy leaves) as NaN
    rels = list(df["path"])
    with Pool(6) as pool:
        out = pool.map(feats, rels, chunksize=64)
    blur = np.stack([o[0] for o in out])
    border = np.stack([o[1] for o in out])
    ph = np.stack([o[2] for o in out])
    tr, te = (df["split"] == "train").values, (df["split"] == "test").values
    rows = []
    for fname, X in (("blurred 16x16", blur), ("border strips", border)):
        for target, kind in (("cause", "hgb"), ("source", "hgb"), ("condition", "lr")):
            y = df[target].values
            acc, bacc, maj = probe(X[tr], X[te], y[tr], y[te], kind)
            rows.append({"features": fname, "target": target, "classifier": "HistGB" if kind == "hgb" else "LogReg",
                         "n_train": int(tr.sum()), "n_test": int(te.sum()), "accuracy": round(acc, 3),
                         "balanced_acc": round(bacc, 3), "majority_baseline": round(maj, 3),
                         "classes": df[target].nunique()})
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "sanity_probe.csv", index=False)
    sus = res[(res["target"] == "cause") & ((res["balanced_acc"] >= 0.70) | (res["accuracy"] >= res["majority_baseline"] + 0.10))]
    md = ["# Background/source probe (Part 12.1)", "",
          "Inputs are ONLY blurred 16x16 images or border strips; train/test are the leakage-resistant group splits. "
          "'cause' = None vs Biotic (no abiotic data exist).", "",
          "| features | target | classifier | classes | n_train | n_test | accuracy | balanced acc | majority baseline |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r.features} | {r.target} | {r.classifier} | {r.classes} | {r.n_train} | {r.n_test} | {r.accuracy} | "
           f"{r.balanced_acc} | {r.majority_baseline} |" for r in res.itertuples()]
    if len(sus):
        md += ["", "**WARNING: " + WARN + "**"]
        print("\nWARNING:", WARN)
        log("SANITY PROBE: " + WARN)
    else:
        md += ["", "Cause was not predicted suspiciously well by background-only inputs under the stated criterion."]
    src = res[res["target"] == "source"]
    if (src["accuracy"] >= 0.90).any():
        md += ["", "Source dataset is highly identifiable from background/border pixels alone (accuracy >= 0.90): "
                   "lab vs field differences are partly a background effect."]
    cnd = res[res["target"] == "condition"]
    md += ["", "Condition (27 classes) from background-only inputs: " +
           ", ".join(f"{r.features}: acc {r.accuracy} (balanced {r.balanced_acc}, majority {r.majority_baseline})"
                     for r in cnd.itertuples())]

    # ---- B. cross-split near duplicates (test -> nearest train)
    packed = np.packbits(ph, axis=1)
    lut = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    pt, pte = packed[tr], packed[te]
    mind = np.empty(len(pte), int)
    arg = np.empty(len(pte), int)
    for s in range(0, len(pte), 128):
        d = lut[pte[s:s + 128, None, :] ^ pt[None, :, :]].sum(-1)
        mind[s:s + 128], arg[s:s + 128] = d.min(1), d.argmin(1)
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    cs = ["", "# Cross-split near-duplicate check (test images vs nearest TRAIN image, pHash of the 256 px cache)", "",
          "| Hamming <= | test images | share |", "|---|---|---|"]
    for t in (4, 6, 8, 10, 12):
        cs.append(f"| {t} | {int((mind <= t).sum())} | {(mind <= t).mean():.4f} |")
    order = np.argsort(mind)[:25]
    pairs = pd.DataFrame({"test_path": df.loc[te_idx[order], "path"].values,
                          "nearest_train_path": df.loc[tr_idx[arg[order]], "path"].values,
                          "hamming": mind[order],
                          "test_condition": df.loc[te_idx[order], "condition"].values,
                          "train_condition": df.loc[tr_idx[arg[order]], "condition"].values,
                          "test_group": df.loc[te_idx[order], "group_id"].values,
                          "train_group": df.loc[tr_idx[arg[order]], "group_id"].values})
    pairs.to_csv(TAB / "crosssplit_nearest_pairs.csv", index=False)
    cs.append("")
    cs.append("The 25 closest cross-split pairs are in crosssplit_nearest_pairs.csv for visual inspection.")
    close = int((mind <= 6).sum())
    if close:
        cs.append(f"**FLAG: {close} test images have a train image within Hamming 6 (the grouping threshold).**")
        log(f"SANITY FLAG: {close} test images within Hamming<=6 of a train image")
    with Pool(6) as pool:
        hashes = pool.map(md5, rels, chunksize=32)
    df["md5"] = hashes
    leak = df.groupby("md5")["split"].nunique()
    cs.append(f"Exact duplicates (MD5) across splits: {int((leak > 1).sum())} hash(es) appear in more than one split; "
              f"{int(df['md5'].duplicated().sum())} duplicate files in total.")
    (TAB / "sanity_probe.md").write_text("\n".join(md + cs) + "\n", encoding="utf-8")
    print("\n".join(md + cs))
    log(f"sanity: probe rows={len(res)}, cause_suspicious={len(sus) > 0}, test imgs within H<=6 of train={close}, "
        f"md5 cross-split={int((leak > 1).sum())}")


if __name__ == "__main__":
    main()
