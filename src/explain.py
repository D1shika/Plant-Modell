"""Explainability: Grad-CAM (health and condition heads) and environment permutation importance.

Grad-CAM target layer: the last convolutional block of EfficientNet-B0 (model.features[-1], 1280 x 7 x 7). The CAM
explains the PREDICTED class of the chosen head. For each crop and head, up to 5 correctly classified and up to 5
misclassified TEST images are saved (seed 42; up to 2 of the 5 slots go to field/PlantDoc images when they exist, since
they are only ~6% of the test set; nothing is duplicated to reach a count).
Heatmaps show where the image branch's evidence for that logit is concentrated; they do NOT prove causal reasoning.

attention_summary: per overlay, the share of CAM energy in (a) the outer 10% frame, (b) the heuristic leaf mask,
(c) the heuristic damage mask (enrichment = energy share / area share; >1 means more attention than area alone).

Permutation importance: one environment feature at a time is shuffled across the test set (10 repeats, seed 42) with the
image features held fixed; importance = baseline accuracy - permuted accuracy. Environment values are SIMULATED, so this
measures reliance on synthetic inputs, not real environmental causation.

Usage: python src/explain.py [--variant fusion] [--no-gradcam] [--no-perm]    (run evaluate.py first)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset import LeafDataset, build_table, load_rgb_256, make_transform  # noqa: E402
from evaluate import md_table  # noqa: E402
from infer import GradCAM, get_device, load_variant, overlay  # noqa: E402
from log_util import log  # noqa: E402
from severity import analyse  # noqa: E402

TAB, EXP = C.OUTPUTS / "tables", C.OUTPUTS / "explain"
N_PER = 5


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:60]


def attention_stats(path, cam224):
    e = cam224.sum() + 1e-8
    m = int(0.1 * 224)
    edge = np.ones_like(cam224, bool)
    edge[m:-m, m:-m] = False
    res = {"edge_energy": float(cam224[edge].sum() / e), "edge_area": float(edge.mean())}
    try:
        _, leaf, dmg, frac, share, flag = analyse(C.ROOT / path)
        if flag == "ok":
            lf = cv2.resize(leaf.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
            dm = cv2.resize(dmg.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
            res["leaf_energy"], res["leaf_area"] = float(cam224[lf].sum() / e), float(lf.mean())
            if dm.mean() > 0.005:
                res["damage_energy"], res["damage_area"] = float(cam224[dm].sum() / e), float(dm.mean())
    except Exception:  # noqa: BLE001
        pass
    return res


def gradcam_run(variant, P, df, maps, stats, dev):
    model, _, _ = load_variant(variant, "best", dev)
    if not model.use_img:
        print("Grad-CAM needs an image branch; skipping", variant)
        return
    for p in model.parameters():
        p.requires_grad_(True)
    gc = GradCAM(model)
    tf = make_transform(False)
    env_n = ((df[maps["env_features"]].values - np.array(stats["mean"])) / np.array(stats["std"])).astype(np.float32)
    env_by_path = dict(zip(df["path"], env_n))
    rng = np.random.default_rng(C.SEED)
    rows, saved = [], 0
    for head, tcol, pcol in (("health", "health", "pred_health"), ("condition", "condition", "pred_condition")):
        for crop in sorted(P["crop"].unique()):
            sub = P[P["crop"] == crop]
            for status, m in (("correct", sub[tcol] == sub[pcol]), ("wrong", sub[tcol] != sub[pcol])):
                cand = sub[m].sort_values("path")
                if len(cand) == 0:
                    continue
                # field (PlantDoc) images are only ~6% of the test set: reserve up to 2 of the 5 slots for them
                fld = cand[cand["source"] == "PlantDoc"]
                lab = cand[cand["source"] != "PlantDoc"]
                fld = fld.iloc[rng.permutation(len(fld))]
                lab = lab.iloc[rng.permutation(len(lab))]
                pick = pd.concat([fld.iloc[:2], lab, fld.iloc[2:]]).iloc[:N_PER]
                for k, r in enumerate(pick.itertuples()):
                    img = load_rgb_256(C.ROOT / r.path)
                    x = tf(img).unsqueeze(0).to(dev)
                    env = torch.from_numpy(env_by_path[r.path]).unsqueeze(0).to(dev)
                    cam, idx = gc(x, env, head)
                    cam224 = cv2.resize(cam[0].float().cpu().numpy(), (224, 224), interpolation=cv2.INTER_CUBIC).clip(0, 1)
                    base = np.asarray(img.resize((224, 224)))
                    panel = np.hstack([base, overlay(base, cam224)])
                    d = EXP / variant / head / safe(crop)
                    d.mkdir(parents=True, exist_ok=True)
                    name = f"{status}_{k}_true-{safe(getattr(r, tcol))}_pred-{safe(getattr(r, pcol))}.png"
                    ok, buf = cv2.imencode(".png", panel[:, :, ::-1])
                    (d / name).write_bytes(buf.tobytes())
                    saved += 1
                    st = attention_stats(r.path, cam224)
                    rows.append({"variant": variant, "head": head, "crop": crop, "status": status, "source": r.source,
                                 "health": r.health, "path": r.path, **st})
    A = pd.DataFrame(rows)
    A.to_csv(EXP / f"attention_{variant}.csv", index=False)
    g = A.groupby(["head", "source", "status"]).agg(
        n=("path", "count"), edge_energy=("edge_energy", "mean"), edge_area=("edge_area", "mean"),
        leaf_energy=("leaf_energy", "mean"), leaf_area=("leaf_area", "mean"),
        damage_energy=("damage_energy", "mean"), damage_area=("damage_area", "mean")).round(3).reset_index()
    g.to_csv(EXP / f"attention_summary_{variant}.csv", index=False)
    print(f"\nGrad-CAM: saved {saved} overlays to {EXP / variant}")
    print(g.to_string(index=False))
    log(f"explain gradcam {variant}: {saved} overlays; attention summary in outputs/explain/attention_summary_{variant}.csv")


@torch.no_grad()
def perm_run(variant, test_df, maps, stats, dev, workers):
    model, _, _ = load_variant(variant, "best", dev)
    if not (model.use_img and model.use_env):
        print("permutation importance needs a fusion model; skipping", variant)
        return None
    ds = LeafDataset(test_df, maps, stats, train=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    feats = []
    for b in dl:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            feats.append(model.pool(model.features(b["image"].to(dev))).flatten(1).float())
    feats = torch.cat(feats)
    env0 = torch.from_numpy(ds.env).to(dev)
    y = {k: torch.from_numpy(np.asarray(v)).to(dev) for k, v in ds.y.items()}

    def accs(env):
        h = torch.cat([feats, model.env(env)], 1)
        out = {"health": model.health(h), "cause": model.cause(h), "condition": model.condition(h)}
        r = {}
        for k, lg in out.items():
            m = y[k] != -1
            r[k] = float((lg.argmax(1)[m] == y[k][m]).float().mean())
        return r

    base = accs(env0)
    n_el = {k: int((y[k] != -1).sum()) for k in base}
    rng = np.random.default_rng(C.SEED)
    rows = []
    for j, f in enumerate(maps["env_features"]):
        drops = {k: [] for k in base}
        for _ in range(10):
            e = env0.clone()
            e[:, j] = e[torch.from_numpy(rng.permutation(len(e))).to(dev), j]
            a = accs(e)
            for k in base:
                drops[k].append(base[k] - a[k])
        rows.append({"variant": variant, "feature": f,
                     **{f"d_{k}_acc_mean": round(float(np.mean(v)), 4) for k, v in drops.items()},
                     **{f"d_{k}_acc_sd": round(float(np.std(v)), 4) for k, v in drops.items()}})
    out = pd.DataFrame(rows)
    out.attrs["base"], out.attrs["n"] = base, n_el
    return out, base, n_el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fusion")
    ap.add_argument("--no-gradcam", action="store_true")
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    a = ap.parse_args()
    dev = get_device()
    df, maps = build_table()
    stats = json.load(open(C.MODELS / "env_stats.json"))
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    EXP.mkdir(parents=True, exist_ok=True)
    if not a.no_gradcam:
        pp = TAB / f"preds_{a.variant}.csv"
        if not pp.exists():
            sys.exit(f"{pp} missing: run evaluate.py first")
        gradcam_run(a.variant, pd.read_csv(pp), df, maps, stats, dev)
    if not a.no_perm:
        md = ["# Environment permutation importance (SIMULATED env inputs)", "",
              "Drop in accuracy when ONE env feature is shuffled across the test set (mean/sd of 10 repeats). "
              "Cause accuracy is trivial (only Biotic labels exist), so its drop is uninformative.", ""]
        allrows = []
        for v in ("fusion", "fusion_shuffled_env"):
            if not (C.MODELS / v / "DONE.json").exists():
                continue
            res = perm_run(v, test_df, maps, stats, dev, a.workers)
            if res is None:
                continue
            out, base, n_el = res
            allrows.append(out)
            md += [f"## {v}", f"Baseline test accuracy: " + ", ".join(f"{k} {base[k]:.4f} (n={n_el[k]})" for k in base), "",
                   md_table(out.drop(columns="variant")), ""]
            print(f"\n{v} permutation importance:\n", out.to_string(index=False))
        if allrows:
            pd.concat(allrows).to_csv(TAB / "env_permutation_importance.csv", index=False)
            (TAB / "env_permutation_importance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
            log("explain: env permutation importance saved")


if __name__ == "__main__":
    main()
