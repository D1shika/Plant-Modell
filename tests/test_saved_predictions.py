"""Regression checks on the SAVED test-split predictions (outputs/tables/preds_<variant>.csv; read-only, nothing written).

The saved files hold each head's argmax and the two raw severity sigmoids, which is enough to apply the reconciliation
rule exactly (final crop/health = metadata of the condition-head argmax) and the ordinal projection. Full probability
vectors are not saved, so reconciled CONFIDENCES cannot be recomputed here (that needs the venv inference test).

These numbers are REPORTING on the held-out test split; nothing here is tuned on them.
Run:  .venv\\Scripts\\python.exe -I tests\\test_saved_predictions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from class_meta import CONDITION_META  # noqa: E402
from reconcile import count_inconsistent_raw, decode_ordinal, project_ordinal  # noqa: E402

SEV = ["Mild", "Moderate", "Severe"]
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))


def boot_acc(correct, groups, n_boot=1000, seed=42):
    """Point estimate and 95% group-bootstrap CI (same scheme as evaluate.acc_stat)."""
    c = np.asarray(correct, float)
    codes, uniq = pd.factorize(pd.Series(groups).astype(str))
    ng = np.bincount(codes, minlength=len(uniq)).astype(float)
    cg = np.bincount(codes, weights=c, minlength=len(uniq))
    d = np.random.default_rng(seed).integers(0, len(uniq), (n_boot, len(uniq)))
    v = cg[d].sum(1) / ng[d].sum(1)
    return c.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def macro_f1(t, p):
    f = []
    for c in SEV:
        tp, fp, fn = ((t == c) & (p == c)).sum(), ((t != c) & (p == c)).sum(), ((t == c) & (p != c)).sum()
        if (t == c).sum():
            f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def fmt(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


for variant in ("image", "fusion"):
    path = ROOT / "outputs" / "tables" / f"preds_{variant}.csv"
    if not path.exists():
        print(f"SKIP {variant}: {path} not found")
        continue
    P = pd.read_csv(path)
    P["severity_valid"] = P["severity_valid"].fillna(False).astype(bool)
    print(f"\n===== {variant}: {len(P)} test images ({(P.acquisition == 'field').sum()} field)")
    P["rec_crop"] = [CONDITION_META[c][0] for c in P["pred_condition"]]
    P["rec_health"] = ["Healthy" if CONDITION_META[c][1] else "Unhealthy" for c in P["pred_condition"]]
    before = count_inconsistent_raw(P["pred_crop"], P["pred_health"], P["pred_condition"])
    after = count_inconsistent_raw(P["rec_crop"], P["rec_health"], P["pred_condition"])
    fb = count_inconsistent_raw(*(P.loc[P.acquisition == "field", c] for c in ("pred_crop", "pred_health", "pred_condition")))
    print(f"inconsistent crop/health/condition: before {before} (field {fb}), after {after}")
    check(f"{variant}: zero inconsistencies after reconciliation", after == 0)
    if variant == "fusion":
        check("fusion: 'before' count matches the analysis report (19)", before == 19, f"got {before}")

    # recognition accuracy: raw heads vs reconciled (condition is unchanged by construction)
    for scope, m in (("all", np.ones(len(P), bool)), ("lab", (P.acquisition == "lab").values),
                     ("field", (P.acquisition == "field").values)):
        g = P.loc[m, "group_id"].values
        line = [f"{scope:5s} n={m.sum():4d}"]
        for h in ("crop", "health"):
            raw = boot_acc((P.loc[m, h] == P.loc[m, "pred_" + h]).values, g)
            rec = boot_acc((P.loc[m, h] == P.loc[m, "rec_" + h]).values, g)
            line.append(f"{h}: raw {fmt(raw)} -> reconciled {fmt(rec)}")
        cond = boot_acc((P.loc[m, "condition"] == P.loc[m, "pred_condition"]).values, g)
        line.append(f"condition (unchanged) {fmt(cond)}")
        print(" | ".join(line))

    # ordinal severity
    p1, p2, viol = project_ordinal(P["p_gt_mild"].values, P["p_gt_moderate"].values)
    print(f"ordinal violations P(>Moderate) > P(>Mild): before {int(viol.sum())}, after {int((p2 > p1).sum())}")
    check(f"{variant}: no ordinal violations after projection", int((p2 > p1).sum()) == 0)
    dec = np.array(SEV)[decode_ordinal(p1, p2)]
    check(f"{variant}: projected decoding equals the saved pred_severity", (dec == P["pred_severity"].values).all())

    # severity, (a) as reported before: true-unhealthy with valid heuristic label; (b) the user pathway: severity is only
    # shown when the RECONCILED health is Unhealthy, so a true-unhealthy leaf predicted healthy counts as a miss.
    el = (P.health == "Unhealthy") & P.severity_valid
    t = P.loc[el, "severity"].values
    a = macro_f1(t, dec[el.values])
    shown = np.where(P.loc[el, "rec_health"].values == "Unhealthy", dec[el.values], "None")
    b = macro_f1(t, shown)
    print(f"severity macro-F1 vs HEURISTIC labels (n={el.sum()}): true-health gating {a:.3f}; "
          f"predicted-health gating (user pathway) {b:.3f}; hidden because predicted healthy: {(shown == 'None').sum()}")
    for acq in ("lab", "field"):
        m = el & (P.acquisition == acq)
        sh = np.where(P.loc[m, "rec_health"].values == "Unhealthy", dec[m.values], "None")
        print(f"   {acq}: n={m.sum()} true-gated {macro_f1(P.loc[m, 'severity'].values, dec[m.values]):.3f}, "
              f"user-pathway {macro_f1(P.loc[m, 'severity'].values, sh):.3f}")

n_ok = sum(ok for _, ok in RESULTS)
print(f"\n{n_ok}/{len(RESULTS)} saved-prediction checks passed")
sys.exit(0 if n_ok == len(RESULTS) else 1)
