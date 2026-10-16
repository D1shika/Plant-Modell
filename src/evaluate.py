"""Evaluate trained variants on the held-out TEST split only (checkpoints were selected on validation loss).

Bootstrap: 1,000 resamples, seed 42, resampling whole GROUPS (split_data.py group ids) with replacement, because images
in a group are near-duplicates and plain image resampling would understate uncertainty. Accuracy CI = 2.5/97.5
percentiles. Macro-F1 CI: per resample, classes absent from that resample's true labels are excluded from the average
(a present class with no correct predictions scores 0). A CI is reported only when the eligible sample has >= 30
images in >= 10 groups; otherwise the CI is shown as 'n/a' (no interval is invented).

Eligibility: crop/health/condition = all test samples; cause = unhealthy samples; severity = unhealthy samples with a
valid heuristic mask. Cause has only Biotic labels (no abiotic data exist), so cause accuracy is trivial.

Usage: python src/evaluate.py [--variants image env fusion fusion_shuffled_env] [--workers 3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset import LeafDataset, build_table  # noqa: E402
from infer import get_device, load_variant, predict_loader  # noqa: E402
from log_util import log  # noqa: E402
from model import CAUSE, CROPS, HEALTH, SEVERITY, VARIANTS, decode_severity  # noqa: E402

NBOOT, MIN_N, MIN_GROUPS = 1000, 30, 10
TAB, FIG = C.OUTPUTS / "tables", C.OUTPUTS / "figures"
NAMES = {"image": "Image-only", "env": "Environment-only", "fusion": "Fusion",
         "fusion_shuffled_env": "Fusion (shuffled env)"}
SEV_IDX = {s: i for i, s in enumerate(SEVERITY)}


# ------------------------------------------------------------------ helpers
def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def save_table(df, name, title, note=""):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    md = f"### {title}\n\n{md_table(df)}\n" + (f"\n{note}\n" if note else "")
    (TAB / f"{name}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def fmt(pt, lo, hi, n):
    if n == 0 or pt != pt:
        return "-"
    ci = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else "[CI n/a]"
    return f"{pt:.3f} {ci} n={n}"


def _draws(g, seed=C.SEED):
    codes, uniq = pd.factorize(pd.Series(g).astype(str))
    rng = np.random.default_rng(seed)
    return codes, len(uniq), rng.integers(0, len(uniq), size=(NBOOT, len(uniq)))


def acc_stat(groups, elig, corr, seed=C.SEED):
    """-> (point, lo, hi, n, n_groups, boot_values or None)."""
    elig = np.asarray(elig, bool)
    corr = np.asarray(corr, bool)
    n = int(elig.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0, None
    g = np.asarray(groups)[elig]
    c = corr[elig]
    pt = float(c.mean())
    codes, G, draws = _draws(g, seed)
    if n < MIN_N or G < MIN_GROUPS:
        return pt, float("nan"), float("nan"), n, G, None
    ng = np.bincount(codes, minlength=G).astype(float)
    cg = np.bincount(codes, weights=c.astype(float), minlength=G)
    vals = cg[draws].sum(1) / ng[draws].sum(1)
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), n, G, vals


def _f1_from_cm(cm):
    tp = np.einsum("...ii->...i", cm)
    sup, prd = cm.sum(-1), cm.sum(-2)
    den = sup + prd
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    present = sup > 0
    return (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1), present.sum(-1)


def sev_f1_stat(P, seed=C.SEED):
    el = (P["health"] == "Unhealthy") & P["severity_valid"]
    n = int(el.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    yt = P.loc[el, "severity"].map(SEV_IDX).values
    yp = P.loc[el, "pred_severity"].map(SEV_IDX).values
    cm = np.zeros((3, 3))
    np.add.at(cm, (yt, yp), 1)
    pt = float(_f1_from_cm(cm)[0])
    codes, G, draws = _draws(P.loc[el, "group_id"].values, seed)
    if n < MIN_N or G < MIN_GROUPS:
        return pt, float("nan"), float("nan"), n, G
    cmg = np.zeros((G, 3, 3))
    np.add.at(cmg, (codes, yt, yp), 1)
    vals = []
    for s in range(0, NBOOT, 100):
        m, present = _f1_from_cm(cmg[draws[s:s + 100]].sum(1))
        vals.extend(m[present > 0].tolist())
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), n, G


def elig_corr(P):
    unh = (P["health"] == "Unhealthy").values
    one = np.ones(len(P), bool)
    return {"crop": (one, (P["crop"] == P["pred_crop"]).values),
            "health": (one, (P["health"] == P["pred_health"]).values),
            "cause": (unh, (P["cause"] == P["pred_cause"]).values),
            "condition": (one, (P["condition"] == P["pred_condition"]).values)}


# ------------------------------------------------------------------ predictions
def collect(variant, test_df, maps, stats, dev, workers, run=None):
    """Predict the test split with models/<run>/best.pt (run defaults to the variant name)."""
    run = run or variant
    model, maps_ck, cfg = load_variant(variant, "best", dev, run=run)
    assert maps_ck["condition"] == maps["condition"], "label maps differ from checkpoint"
    ds = LeafDataset(test_df, maps, stats, train=False, load_image=(variant != "env"))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    lg = predict_loader(model, dl, dev)
    P = test_df[["path", "source", "acquisition", "crop", "condition", "health", "cause", "group_id",
                 "severity", "severity_valid"]].reset_index(drop=True).copy()
    P["severity_valid"] = P["severity_valid"].fillna(False).astype(bool)
    for h, names in (("crop", CROPS), ("health", HEALTH), ("cause", CAUSE), ("condition", maps["condition"])):
        pr = torch.softmax(lg[h], 1)
        P["pred_" + h] = [names[i] for i in pr.argmax(1).tolist()]
        P["conf_" + h] = pr.max(1).values.numpy()
    sp = torch.sigmoid(lg["severity"]).numpy()
    P["p_gt_mild"], P["p_gt_moderate"] = sp[:, 0], sp[:, 1]
    P["pred_severity"] = [SEVERITY[i] for i in decode_severity(lg["severity"]).tolist()]
    P.to_csv(TAB / f"preds_{run}.csv", index=False)
    return P, {"epoch": torch.load(C.MODELS / run / "best.pt", map_location="cpu", weights_only=False)["epoch"]}


# ------------------------------------------------------------------ tables
def table3(preds, train_df, test_df):
    rows, num = [], []
    for v in VARIANTS:
        if v not in preds:
            rows.append({"Model": NAMES[v], **{k: "not trained" for k in
                                               ("Crop accuracy", "Health accuracy", "Cause accuracy",
                                                "Condition accuracy", "Severity macro-F1")}})
            continue
        P, ec = preds[v], None
        ec = elig_corr(P)
        row = {"Model": NAMES[v]}
        for col, h in (("Crop accuracy", "crop"), ("Health accuracy", "health"), ("Cause accuracy", "cause"),
                       ("Condition accuracy", "condition")):
            pt, lo, hi, n, G, _ = acc_stat(P["group_id"].values, *ec[h])
            row[col] = fmt(pt, lo, hi, n)
            num.append(dict(variant=v, metric=h, value=pt, ci_lo=lo, ci_hi=hi, n=n, n_groups=G))
        pt, lo, hi, n, G = sev_f1_stat(P)
        row["Severity macro-F1"] = fmt(pt, lo, hi, n)
        num.append(dict(variant=v, metric="severity_macro_f1", value=pt, ci_lo=lo, ci_hi=hi, n=n, n_groups=G))
        rows.append(row)
    # majority-class reference (train majority applied to the test set)
    base = {"Model": "Majority-class baseline"}
    unh_tr, unh_te = train_df["health"] == "Unhealthy", test_df["health"] == "Unhealthy"
    for col, h in (("Crop accuracy", "crop"), ("Health accuracy", "health"), ("Condition accuracy", "condition")):
        m = train_df[h].mode()[0]
        base[col] = f"{(test_df[h] == m).mean():.3f} n={len(test_df)}"
    base["Cause accuracy"] = f"{(test_df.loc[unh_te, 'cause'] == 'Biotic').mean():.3f} n={int(unh_te.sum())}"
    sv_tr = train_df[unh_tr & train_df["severity_valid"].fillna(False).astype(bool)]["severity"].mode()[0]
    sv_te = test_df[unh_te & test_df["severity_valid"].fillna(False).astype(bool)]
    p, r, f, _ = precision_recall_fscore_support(sv_te["severity"].map(SEV_IDX), [SEV_IDX[sv_tr]] * len(sv_te),
                                                 labels=[0, 1, 2], zero_division=0)
    base["Severity macro-F1"] = f"{f.mean():.3f} n={len(sv_te)}"
    rows.append(base)
    df = pd.DataFrame(rows)
    pd.DataFrame(num).to_csv(TAB / "table3_numeric.csv", index=False)
    save_table(df, "table3_modality_ablation", "Table 3 - Modality ablation (test split; 95% group-bootstrap CI)",
               "Cause accuracy is trivial: only Biotic labels exist (no abiotic data). Environment values are SIMULATED. "
               "Severity labels are heuristic. The shuffled-env row is the control: compare it with Fusion.")
    return num


def diff_cell(a, b):
    """a=field stat tuple, b=lab stat tuple -> pp difference with independent-bootstrap CI."""
    if a[3] == 0 or b[3] == 0:
        return "-"
    d = (a[0] - b[0]) * 100
    if a[5] is not None and b[5] is not None:
        dv = (a[5] - b[5]) * 100
        return f"{d:+.1f} pp [{np.percentile(dv, 2.5):+.1f}, {np.percentile(dv, 97.5):+.1f}]"
    return f"{d:+.1f} pp [CI n/a]"


def table4(P):
    lab, fld = P["acquisition"] == "lab", P["acquisition"] == "field"
    unh = P["health"] == "Unhealthy"
    rows = []

    def add(name, h, extra=None, label=""):
        ec = elig_corr(P)[h]
        cells = []
        stats = []
        for m in (lab, fld):
            mm = m.values if extra is None else (m & extra).values
            e, c = ec[0] & mm, ec[1]
            st = acc_stat(P["group_id"].values, e, c)
            stats.append(st)
            cells.append(fmt(*st[:4]))
        rows.append({"Comparison": name, "Lab (PlantVillage)": cells[0], "Field (PlantDoc)": cells[1],
                     "Field - Lab": diff_cell(stats[1], stats[0])})

    add("Health accuracy (all samples)", "health")
    add("Cause accuracy (unhealthy; trivial)", "cause")
    add("Condition accuracy (all samples)", "condition")
    common = sorted(set(P.loc[lab & unh, "condition"]) & set(P.loc[fld & unh, "condition"]))
    inc = P["condition"].isin(common)
    add("Like-for-like biotic: health accuracy", "health", extra=unh & inc)
    add("Like-for-like biotic: cause accuracy", "cause", extra=unh & inc)
    add("Like-for-like biotic: condition accuracy", "condition", extra=unh & inc)
    df = pd.DataFrame(rows)
    save_table(df, "table4_lab_vs_field", "Table 4 - Lab vs field (Fusion model, test split)",
               f"Like-for-like biotic subset = unhealthy samples of conditions present in both sources' test sets "
               f"({len(common)}): {', '.join(common) if common else 'none'}. Differences are Field minus Lab in "
               f"percentage points with independent group-bootstrap intervals. '-' = unavailable.")


def table5(P, maps):
    note = ("Required classes Disease / N_Deficiency / NK_Deficiency are NOT available: no abiotic (nutrient) data "
            "were obtained (OLID I is Kaggle-only). N and NK deficiency were not relabelled as anything else.")
    (TAB / "table5_tomato_required_UNAVAILABLE.md").write_text("### Table 5 - Tomato Disease/N/NK confusion\n\n"
                                                              "UNAVAILABLE.\n\n" + note + "\n", encoding="utf-8")
    print("\nTable 5 (Disease / N_Deficiency / NK_Deficiency): UNAVAILABLE -", note)
    T = P[P["crop"] == "Tomato"]
    # alternative A: tomato healthy vs disease from the health head (NOT equivalent to Table 5)
    cm = pd.crosstab(T["health"].map({"Healthy": "Healthy", "Unhealthy": "Disease"}),
                     T["pred_health"].map({"Healthy": "Healthy", "Unhealthy": "Disease"}))
    cm = cm.reindex(index=["Healthy", "Disease"], columns=["Healthy", "Disease"]).fillna(0).astype(int)
    cm.insert(0, "n_true", cm.sum(axis=1))
    cm = cm.reset_index().rename(columns={"health": "True \\ Predicted"})
    save_table(cm, "table5_alt_tomato_healthy_vs_disease",
               "Table 5 (ALTERNATIVE, not equivalent) - Tomato Healthy vs Disease, health head (Fusion)")
    # alternative B: tomato per-condition confusion (condition head)
    tcs = [c for c in maps["condition"] if c.startswith("Tomato___")]
    cols = tcs + ["other_crop"]
    pc = T["pred_condition"].where(T["pred_condition"].isin(tcs), "other_crop")
    cmx = pd.crosstab(T["condition"], pc).reindex(index=tcs, columns=cols).fillna(0).astype(int)
    cmx.insert(0, "n_true", cmx.sum(axis=1))
    cmx = cmx.reset_index().rename(columns={"condition": "True \\ Predicted"})
    save_table(cmx, "table5_alt_tomato_per_condition", "Table 5 (ALTERNATIVE) - Tomato per-condition confusion (Fusion)")


def full_confusion(P, maps, variant):
    cs = maps["condition"]
    cm = pd.crosstab(P["condition"], P["pred_condition"]).reindex(index=cs, columns=cs).fillna(0).astype(int)
    cm.insert(0, "n_true", cm.sum(axis=1))
    cm.reset_index().rename(columns={"condition": "true\\pred"}).to_csv(TAB / f"confusion_condition_{variant}.csv",
                                                                         index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm[cs].div(cm["n_true"].replace(0, 1), axis=0).values
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cs)), cs, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cs)), [f"{c} (n={n})" for c, n in zip(cs, cm["n_true"])], fontsize=6)
    ax.set(title=f"Condition confusion, {NAMES[variant]} (row-normalised, test)", xlabel="predicted", ylabel="true")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"confusion_condition_{variant}.png", dpi=130)
    plt.close(fig)


def table6(preds):
    rows = []
    for v in VARIANTS:
        if v not in preds:
            continue
        P = preds[v]
        for scope, mask in (("All unhealthy", P["health"] == "Unhealthy"), ("Biotic", P["cause"] == "Biotic"),
                            ("Abiotic", P["cause"] == "Abiotic")):
            el = mask & P["severity_valid"]
            n = int(el.sum())
            if n == 0:
                rows.append({"Model": NAMES[v], "Scope": scope, "Class": "-", "Precision": "-", "Recall": "-",
                             "F1": "-", "Support": 0})
                continue
            yt, yp = P.loc[el, "severity"].map(SEV_IDX), P.loc[el, "pred_severity"].map(SEV_IDX)
            p, r, f, s = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], zero_division=0)
            for i, c in enumerate(SEVERITY):
                rows.append({"Model": NAMES[v], "Scope": scope, "Class": c, "Precision": f"{p[i]:.3f}",
                             "Recall": f"{r[i]:.3f}", "F1": f"{f[i]:.3f}", "Support": int(s[i])})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table6_severity_all_models.csv", index=False)
    has_fusion = (df["Model"] == NAMES["fusion"]).any()
    main = df[df["Model"] == NAMES["fusion"]] if has_fusion else df
    save_table(main, "table6_severity",
               f"Table 6 - Severity precision/recall/F1 ({'Fusion' if has_fusion else 'all available models'}; "
               f"valid heuristic labels only)",
               "Evaluated against HEURISTIC damage-fraction labels (Mild <0.10, Moderate 0.10-0.30, Severe >0.30); this "
               "does not establish real-world severity accuracy. Abiotic: no examples (-). All-models version: "
               "table6_severity_all_models.csv.")


def per_group(preds):
    rows = []
    for v, P in preds.items():
        ec = elig_corr(P)
        for kind, col in (("source", "source"), ("crop", "crop")):
            for val, m in P.groupby(col).groups.items():
                mask = np.zeros(len(P), bool)
                mask[list(m)] = True
                row = {"model": NAMES[v], "group_type": kind, "group": val, "n": int(mask.sum())}
                for h in ("crop", "health", "condition", "cause"):
                    e, c = ec[h]
                    pt, lo, hi, n, G, _ = acc_stat(P["group_id"].values, e & mask, c)
                    row[f"{h}_acc"] = "-" if n == 0 else f"{pt:.3f}"
                    row[f"{h}_n"] = n
                rows.append(row)
    df = pd.DataFrame(rows)
    save_table(df, "per_group_metrics", "Per-source and per-crop metrics with sample counts (test split)")
    if "fusion" in preds:
        for _, r in df[(df.model == NAMES["fusion"]) & (df.group_type == "source")].iterrows():
            lab = "Field" if r["group"] == "PlantDoc" else "Lab"
            print(f"{lab} cause accuracy: {r['cause_acc']} on n={r['cause_n']} | health {r['health_acc']} on "
                  f"n={r['health_n']} | condition {r['condition_acc']} on n={r['condition_n']}")


def sanity_flags(preds, num, test_df, train_df):
    flags = []
    nm = pd.DataFrame(num)
    for _, r in nm.iterrows():
        if r["value"] == 1.0:
            flags.append(f"{NAMES[r['variant']]} {r['metric']} = 1.000 on n={r['n']}"
                         + (" (cause is trivial: only Biotic labels exist)" if r["metric"] == "cause" else
                            " -> investigate (possible leakage or trivial task)"))
    for v, P in preds.items():
        ec = elig_corr(P)
        for h in ("health", "condition"):
            if v == "env":  # constant-predictor class-prior artifact, not a source effect (noted in LIMITATIONS)
                continue
            e, c = ec[h]
            a = {k: (c[e & (P["acquisition"] == k).values].mean() if (e & (P["acquisition"] == k).values).any() else np.nan)
                 for k in ("lab", "field")}
            if a["lab"] == a["lab"] and a["field"] == a["field"] and abs(a["lab"] - a["field"]) > 0.10:
                flags.append(f"{NAMES[v]} {h}: lab {a['lab']:.3f} vs field {a['field']:.3f} "
                             f"(gap {abs(a['lab'] - a['field']) * 100:.1f} pp > 10 pp)")
        wrong = P["pred_condition"] != P["condition"]
        if wrong.sum():
            mc = P.loc[wrong, "conf_condition"].mean()
            if mc > 0.8:
                flags.append(f"{NAMES[v]}: mean confidence on WRONG condition predictions = {mc:.2f} (> 0.8, overconfident)")
    if "env" in preds:
        P = preds["env"]
        base = (test_df["health"] == train_df["health"].mode()[0]).mean()
        acc = (P["health"] == P["pred_health"]).mean()
        if acc > base + 0.05:
            flags.append(f"Environment-only health accuracy {acc:.3f} exceeds majority baseline {base:.3f} by > 5 pp "
                         f"-> simulated env may leak labels")
    small = test_df["condition"].value_counts()
    small = small[small < 20]
    if len(small):
        flags.append("Conditions with < 20 test samples: " + ", ".join(f"{k} (n={v})" for k, v in small.items()))
    text = "# Sanity flags (evaluate.py)\n\n" + ("\n".join(f"- {f}" for f in flags) if flags else "- none\n")
    (TAB / "sanity_flags.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    for f in flags:
        log("SANITY FLAG: " + f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    df, maps = build_table()
    stats = json.load(open(C.MODELS / "env_stats.json"))
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"TEST split only: {len(test_df)} images, {test_df['group_id'].nunique()} groups; device {dev}")
    preds, manifest = {}, {}
    for v in a.variants:
        if not (C.MODELS / v / "DONE.json").exists():
            print(f"skip {v}: training not complete (no DONE.json)")
            continue
        preds[v], manifest[v] = collect(v, test_df, maps, stats, dev, a.workers)
        print(f"{v}: predictions collected (best epoch {manifest[v]['epoch']})")
    if not preds:
        sys.exit("No completed variants to evaluate")
    json.dump({"test_n": len(test_df), "variants": manifest, "bootstrap": NBOOT, "seed": C.SEED},
              open(TAB / "eval_manifest.json", "w"), indent=2)
    num = table3(preds, train_df, test_df)
    if "fusion" in preds:
        table4(preds["fusion"])
        table5(preds["fusion"], maps)
    for v in ("fusion", "image"):
        if v in preds:
            full_confusion(preds[v], maps, v)
    table6(preds)
    per_group(preds)
    sanity_flags(preds, num, test_df, train_df)
    log(f"evaluate: variants={list(preds)} test_n={len(test_df)}; tables in outputs/tables")


if __name__ == "__main__":
    main()
