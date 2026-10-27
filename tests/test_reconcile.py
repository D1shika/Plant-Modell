"""Unit tests for src/reconcile.py and src/class_meta.py with SYNTHETIC logits (NumPy only; no torch, no files written).

Run:  .venv\\Scripts\\python.exe -I tests\\test_reconcile.py      (exit code 0 = all passed)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from class_meta import CONDITION_META, validate  # noqa: E402
from reconcile import (CAUSE_UNAVAILABLE, aggregation_matrices, count_inconsistent_raw, decode_ordinal,  # noqa: E402
                       project_ordinal, reconcile, severity_class_probs, softmax)

MAPS = json.load(open(ROOT / "models" / "label_maps.json"))  # read-only
CONDS, CROPS = MAPS["condition"], MAPS["crop"]
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))


def logits_for(cond_scores, crop=None, health=None, sev=(0.0, 0.0), n=1):
    """Build one-row logits: cond_scores = {condition: logit}; others 0. crop/health set the raw heads' argmax."""
    lg = {"condition": np.full((n, len(CONDS)), -10.0), "crop": np.zeros((n, len(CROPS))),
          "health": np.zeros((n, 2)), "cause": np.array([[5.0, -5.0]] * n), "severity": np.array([sev] * n, float)}
    for c, v in cond_scores.items():
        lg["condition"][:, CONDS.index(c)] = v
    if crop:
        lg["crop"][:, CROPS.index(crop)] = 10.0
    if health:
        lg["health"][:, MAPS["health"].index(health)] = 10.0
    return lg


# ---- metadata
check("metadata matches the production label map exactly", validate(CONDS, CROPS) == [], str(validate(CONDS, CROPS)))
check("metadata: exactly one healthy class per crop except Orange",
      {k: sum(1 for c, (cr, h) in CONDITION_META.items() if cr == k and h) for k in CROPS}
      == {k: (0 if k == "Orange" else 1) for k in CROPS})
check("validate() reports an unknown class", validate(CONDS + ["Mango___Anthracnose"], CROPS) != [])
check("validate() reports a missing class", validate(CONDS[1:], CROPS) != [])
try:
    aggregation_matrices(CONDS + ["Mango___Anthracnose"], CROPS)
    check("aggregation raises on a class without metadata", False)
except ValueError:
    check("aggregation raises on a class without metadata", True)

# ---- probability aggregation
rng = np.random.default_rng(0)
L = rng.normal(0, 3, (50, len(CONDS)))
P = softmax(L)
M, H = aggregation_matrices(CONDS, CROPS)
check("each condition belongs to exactly one crop", np.allclose(M.sum(1), 1))
check("crop marginals sum to 1", np.allclose((P @ M).sum(1), 1))
tom = [i for i, c in enumerate(CONDS) if CONDITION_META[c][0] == "Tomato"]
check("P(Tomato) = sum of tomato condition probabilities", np.allclose((P @ M)[:, CROPS.index("Tomato")], P[:, tom].sum(1)))
check("P(healthy) uses only metadata-healthy classes",
      np.allclose(P[:, H].sum(1), P[:, [i for i, c in enumerate(CONDS) if CONDITION_META[c][1]]].sum(1)))

# ---- crop / health consistency (the report's two examples)
r = reconcile(logits_for({"Apple___Apple_scab": 5.0}, crop="Tomato", health="Unhealthy"), MAPS, ["Biotic"])[0]
check("raw crop Tomato + condition Apple scab -> final crop Apple", r["crop"] == "Apple" and r["condition"] == "Apple___Apple_scab")
check("raw crop kept for diagnostics and disagreement flagged", r["raw_crop"] == "Tomato" and "crop" in r["head_disagreement"])
check("head disagreement makes the result uncertain", r["uncertain"])
r = reconcile(logits_for({"Tomato___Bacterial_spot": 5.0}, crop="Tomato", health="Healthy"), MAPS, ["Biotic"])[0]
check("raw health Healthy + Tomato bacterial spot -> final Unhealthy", r["health"] == "Unhealthy" and r["raw_health"] == "Healthy")
check("health disagreement flagged", "health" in r["head_disagreement"])

# ---- healthy handling
r = reconcile(logits_for({"Grape___healthy": 8.0}, crop="Grape", health="Healthy", sev=(5.0, 5.0)), MAPS, ["Biotic"])[0]
check("healthy condition -> Healthy, cause None, severity None",
      r["health"] == "Healthy" and r["cause"] == "None" and r["severity"] == "None")
check("confident consistent prediction is not uncertain", not r["uncertain"], r["uncertain_reasons"])

# ---- cause
r = reconcile(logits_for({"Apple___Black_rot": 8.0}, crop="Apple", health="Unhealthy"), MAPS, ["Biotic"])[0]
check("single trained cause class -> cause unavailable", r["cause"] == CAUSE_UNAVAILABLE and r["cause_conf"] != r["cause_conf"])
r = reconcile(logits_for({"Apple___Black_rot": 8.0}, crop="Apple", health="Unhealthy"), MAPS, None)[0]
check("missing cause metadata -> cause unavailable", r["cause"] == CAUSE_UNAVAILABLE)
r = reconcile(logits_for({"Apple___Black_rot": 8.0}, crop="Apple", health="Unhealthy"), MAPS, ["Biotic", "Abiotic"])[0]
check("two trained cause classes -> cause head used", r["cause"] == "Biotic")

# ---- ambiguity of marginals (argmax condition is Apple scab, but tomato mass is larger)
sc = {"Apple___Apple_scab": 2.0, "Tomato___Early_blight": 1.5, "Tomato___Late_blight": 1.5, "Tomato___Leaf_Mold": 1.5}
r = reconcile(logits_for(sc, crop="Apple", health="Unhealthy"), MAPS, ["Biotic"])[0]
check("crop marginal disagreement -> crop stays consistent but uncertain",
      r["crop"] == "Apple" and "crop" in r["marginal_disagreement"] and r["uncertain"])
check("low condition confidence reported", "low condition confidence" in r["uncertain_reasons"])

# ---- ordinal severity
p1, p2, v = project_ordinal(np.array([0.3, 0.8, 0.9]), np.array([0.6, 0.2, 0.95]))
check("projection fixes violations and keeps consistent rows", list(v) == [True, False, True]
      and np.allclose(p2, [0.3, 0.2, 0.9]))
sp = severity_class_probs(p1, p2)
check("projected severity class probabilities are valid", np.all(sp >= 0) and np.allclose(sp.sum(1), 1))
z = rng.normal(0, 3, (500, 2))
s = 1 / (1 + np.exp(-z))
a1, a2, viol = project_ordinal(s[:, 0], s[:, 1])
ref = (z[:, 0] > 0).astype(int) * (1 + (z[:, 1] > 0).astype(int))  # model.decode_severity rule on raw logits
check("decoding after projection equals the original decode rule", np.array_equal(decode_ordinal(a1, a2), ref))
check("no violations remain after projection", int((a2 > a1).sum()) == 0, f"raw violations {int(viol.sum())}")
r = reconcile(logits_for({"Tomato___Late_blight": 8.0}, crop="Tomato", health="Unhealthy", sev=(-1.0, 2.0)), MAPS, ["Biotic"])[0]
check("raw severity kept and violation recorded",
      r["ordinal_violation_raw"] and r["raw_p_gt_moderate"] > r["raw_p_gt_mild"] and r["p_gt_moderate_projected"] <= r["raw_p_gt_mild"])

# ---- batch behaviour
idx = rng.integers(0, len(CONDS), 64)
lg = {"condition": rng.normal(0, 1, (64, len(CONDS))), "crop": rng.normal(0, 3, (64, len(CROPS))),
      "health": rng.normal(0, 3, (64, 2)), "cause": rng.normal(0, 1, (64, 2)), "severity": rng.normal(0, 2, (64, 2))}
lg["condition"][np.arange(64), idx] += 6
out = reconcile(lg, MAPS, ["Biotic"])
check("batch: one result per row", len(out) == 64)
check("batch: zero user-facing crop/health/condition inconsistencies",
      count_inconsistent_raw([r["crop"] for r in out], [r["health"] for r in out], [r["condition"] for r in out]) == 0)
raw_bad = count_inconsistent_raw([r["raw_crop"] for r in out], [r["raw_health"] for r in out], [r["condition"] for r in out])
check("batch: random raw heads do disagree (sanity of the test itself)", raw_bad > 0, f"raw inconsistent {raw_bad}/64")
check("batch: final condition is the condition-head argmax", [r["condition"] for r in out] == [CONDS[i] for i in idx])
one = reconcile({k: v[:1] for k, v in lg.items()}, MAPS, ["Biotic"])[0]
same = all((np.isclose(one[k], out[0][k], equal_nan=True) if isinstance(one[k], float) else one[k] == out[0][k])
           for k in one)
check("single-row call equals the same row in a batch", same and one.keys() == out[0].keys())

# ---- temperature (calibration hook)
r1 = reconcile(logits_for({"Apple___Black_rot": 3.0}), MAPS, ["Biotic"])[0]
r2 = reconcile(logits_for({"Apple___Black_rot": 3.0}), MAPS, ["Biotic"], temperature={"condition": 2.0})[0]
check("temperature > 1 lowers confidence without changing the answer",
      r2["condition"] == r1["condition"] and r2["condition_conf"] < r1["condition_conf"])

n_ok = sum(ok for _, ok in RESULTS)
print(f"\n{n_ok}/{len(RESULTS)} reconcile checks passed")
sys.exit(0 if n_ok == len(RESULTS) else 1)
