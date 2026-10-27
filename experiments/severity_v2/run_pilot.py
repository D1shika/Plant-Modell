"""Phase C pilot: compare the current severity heuristic (src/severity.py, "v1") with severity_v2 on one seeded sample.

EXPLORATORY. Heuristic labels only: no expert ground truth exists, so agreement between v1 and v2 is NOT accuracy.

Sample (fixed before any v2 result was seen):
  * TRAIN split of data/processed/splits.csv only (no validation/test images; nothing is tuned on held-out data).
  * Seed 42 (config.SEED); within each stratum images are drawn uniformly at random with pandas .sample(random_state=42).
  * Strata, N_PER images each (default 12):
      white_mildew   lab  Cherry___Powdery_mildew
      chlorosis      lab  Orange___Haunglongbing, Tomato___Yellow_Leaf_Curl_Virus (half each)
      necrosis       lab  Tomato___Early_blight, Grape___Black_rot (half each)
      grey_lesions   lab  Tomato___Septoria_leaf_spot, Apple___Apple_scab (half each)
      healthy        lab  all *___healthy conditions pooled (negative check: damage here is false damage)
      field          field all Unhealthy PlantDoc images pooled
  * Exclusions: none by design. Unreadable files are kept in the table with flag 'unreadable'.
Pre-stated expectations (written before running): if the fix works, v2 damage should rise mainly in white_mildew and
grey_lesions, change little in chlorosis/necrosis, and stay low on healthy leaves.

Reads only: data/processed/splits.csv, data/processed/severity.csv, data/raw images. Writes only under --out
(default experiments/severity_v2/out); refuses to write into a folder that already holds results.

Run from the project root:  .venv\\Scripts\\python.exe -I experiments\\severity_v2\\run_pilot.py [--out experiments\\severity_v2\\out_windows]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from severity_v2 import ROOT, VERSION, analyse_v2, label, load, overlay_components  # noqa: E402
from severity import analyse, severity_class  # noqa: E402  (v1, unchanged)

SEED = 42
STRATA = [
    ("white_mildew", "lab", ["Cherry___Powdery_mildew"]),
    ("chlorosis", "lab", ["Orange___Haunglongbing", "Tomato___Yellow_Leaf_Curl_Virus"]),
    ("necrosis", "lab", ["Tomato___Early_blight", "Grape___Black_rot"]),
    ("grey_lesions", "lab", ["Tomato___Septoria_leaf_spot", "Apple___Apple_scab"]),
    ("healthy", "lab", None),
    ("field", "field", None),
]


def draw_sample(n_per):
    sp = pd.read_csv(ROOT / "data" / "processed" / "splits.csv")
    tr = sp[sp["split"] == "train"]
    parts = []
    for name, acq, conds in STRATA:
        pool = tr[tr["acquisition"] == acq]
        if conds is None:
            pool = pool[pool["health"] == ("Healthy" if name == "healthy" else "Unhealthy")]
            picks = [pool.sample(n=min(n_per, len(pool)), random_state=SEED)]
        else:
            k = [n_per // len(conds) + (i < n_per % len(conds)) for i in range(len(conds))]
            picks = []
            for c, kc in zip(conds, k):
                cp = pool[pool["condition"] == c]
                if len(cp) < kc:
                    raise SystemExit(f"stratum {name}: only {len(cp)} train images for {c}, need {kc}")
                picks.append(cp.sample(n=kc, random_state=SEED))
        part = pd.concat(picks)
        part.insert(0, "stratum", name)
        parts.append(part)
    return pd.concat(parts).reset_index(drop=True)


def panel(img, v1_damage, v2_res):
    a = img.copy()
    b = img.copy()
    if v1_damage is not None:
        b[v1_damage] = (0, 0, 255)
    c = overlay_components(v2_res) if v2_res is not None else img.copy()
    return np.hstack([a, b, c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--n_per", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "severity_v2_pilot.csv").exists():
        raise SystemExit(f"{out} already holds pilot results; pass a new --out folder (nothing was overwritten)")
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    sample = draw_sample(a.n_per)
    sample[["stratum", "path", "condition", "health", "acquisition", "source", "split"]].to_csv(
        out / "sample_manifest.csv", index=False)
    stored = pd.read_csv(ROOT / "data" / "processed" / "severity.csv")[["path", "damage_fraction", "mask_flag",
                                                                        "severity", "severity_valid"]]
    stored = stored.rename(columns=lambda c: c if c == "path" else "stored_" + c)

    rows = []
    for i, r in sample.iterrows():
        rec = {"stratum": r["stratum"], "path": r["path"], "condition": r["condition"], "health": r["health"],
               "acquisition": r["acquisition"]}
        try:
            img_raw = load(ROOT / r["path"])
            img1, _, dmg1, f1, share1, flag1 = analyse(ROOT / r["path"])
            res = analyse_v2(img_raw)
        except Exception as e:  # recorded, never hidden
            rec.update(v1_flag="unreadable", v2_flag="unreadable", error=str(e)[:200])
            rows.append(rec)
            continue
        s1 = severity_class(f1) if f1 == f1 else "Invalid"
        s2, valid2 = label(res)
        rec.update(v1_damage_fraction=f1, v1_leaf_share=share1, v1_flag=flag1, v1_severity=s1,
                   v1_valid=bool(flag1 == "ok" and s1 != "Invalid"),
                   v2_damage_fraction=res["damage_fraction"], v2_leaf_share=res["leaf_share"], v2_flag=res["flag"],
                   v2_severity=s2, v2_valid=valid2, v2_restored_lowsat_px=res["restored_lowsat_px"],
                   v2_chlorotic=res["chlorotic"], v2_necrotic=res["necrotic"], v2_pale=res["pale"],
                   v2_dark=res["dark"])
        name = f"{i:03d}_{r['stratum']}_{r['condition']}.png"
        cv2.imwrite(str(out / "overlays" / name), panel(img1, dmg1, res))
        rec["overlay"] = f"overlays/{name}"
        rows.append(rec)
        print(f"{i + 1}/{len(sample)} {r['stratum']:13s} v1={f1:.3f} {flag1:16s} v2={res['damage_fraction']:.3f} "
              f"{res['flag']}", flush=True)

    df = pd.DataFrame(rows).merge(stored, on="path", how="left")
    for c in ("v1_valid", "v2_valid"):
        df[c] = df[c].fillna(False).astype(bool)
    df.insert(0, "method_version", VERSION)
    df.to_csv(out / "severity_v2_pilot.csv", index=False)
    summarise(df, out)


def md_table(ct):
    """DataFrame crosstab -> markdown (no tabulate dependency)."""
    head = "| " + " | ".join([str(ct.index.name or "")] + [str(c) for c in ct.columns]) + " |"
    sep = "|" + "---|" * (len(ct.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(ct.index, ct.values)]
    return "\n".join([head, sep] + body)


def summarise(df, out):
    L = [f"# severity_v2 pilot summary ({VERSION})", "",
         "Heuristic comparison only. Agreement between v1 and v2 is heuristic agreement, not accuracy.", ""]
    # reproducibility of v1 on this machine vs the stored labels (unhealthy images only are stored)
    m = df.dropna(subset=["stored_damage_fraction", "v1_damage_fraction"])
    if len(m):
        d = (m["v1_damage_fraction"] - m["stored_damage_fraction"]).abs()
        L += ["## v1 reproducibility (recomputed v1 vs stored data/processed/severity.csv)", "",
              f"n = {len(m)}; |diff| median {d.median():.4f}, max {d.max():.4f}; "
              f"same category {int((m['v1_severity'] == m['stored_severity']).sum())}/{len(m)}; "
              f"same flag {int((m['v1_flag'] == m['stored_mask_flag']).sum())}/{len(m)}", ""]
    L += ["## Per stratum", "",
          "| stratum | n | v1 median frac | v2 median frac | median v2-v1 | images with v2>v1+0.01 | v2<v1-0.01 |"
          " median restored px | v1 ok | v2 ok | v2 pale_dominant | category changed (both valid) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, g in df.groupby("stratum", sort=False):
        dd = g["v2_damage_fraction"] - g["v1_damage_fraction"]
        both = g[g["v1_valid"] & g["v2_valid"]]
        L.append(f"| {s} | {len(g)} | {g['v1_damage_fraction'].median():.3f} | {g['v2_damage_fraction'].median():.3f} | "
                 f"{dd.median():+.3f} | {int((dd > .01).sum())} | {int((dd < -.01).sum())} | "
                 f"{g['v2_restored_lowsat_px'].median():.0f} | {int((g['v1_flag'] == 'ok').sum())} | "
                 f"{int((g['v2_flag'] == 'ok').sum())} | {int((g['v2_flag'] == 'pale_dominant').sum())} | "
                 f"{int((both['v1_severity'] != both['v2_severity']).sum())}/{len(both)} |")
    L += ["", "## Category distributions (all rows; Invalid = mask failed)", ""]
    for s, g in df.groupby("stratum", sort=False):
        L.append(f"**{s}**: v1 {dict(g['v1_severity'].value_counts())}; v2 {dict(g['v2_severity'].value_counts())}")
        L.append("")
    L += ["## Heuristic agreement v1 (rows) x v2 (columns), unhealthy strata, both valid", ""]
    u = df[(df["stratum"] != "healthy") & df["v1_valid"] & df["v2_valid"]]
    L += [md_table(pd.crosstab(u["v1_severity"], u["v2_severity"])), ""]
    for acq in ("lab", "field"):
        ua = u[u["acquisition"] == acq]
        L.append(f"{acq}: same category {int((ua['v1_severity'] == ua['v2_severity']).sum())}/{len(ua)}")
    L += ["", "## Flags", "", md_table(pd.crosstab(df["v1_flag"], df["v2_flag"])), ""]
    (out / "pilot_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
