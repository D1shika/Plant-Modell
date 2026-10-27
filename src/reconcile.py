"""Inference-only post-processing that makes the user-facing answer internally consistent.  NumPy only (no torch), so
it can be unit-tested anywhere.

WHY: the model has independent crop / health / condition heads. Nothing forces them to agree, and on the saved test
predictions 19 of 3,464 (18 of 206 field photos) disagree, e.g. crop "Tomato" with condition "Apple___Apple_scab".

WHAT (no retraining, original head outputs kept for diagnostics):
1. The final condition is the argmax of the condition head (unchanged, so condition accuracy is unchanged).
2. Final crop and health are DEFINED by the final condition through explicit metadata (class_meta.CONDITION_META), so
   crop / health / condition always agree.
3. Their confidences are marginals of the condition softmax:
       P(crop k)   = sum of P(condition c) over conditions c of crop k
       P(healthy)  = sum of P(condition c) over conditions marked healthy;  P(unhealthy) = 1 - P(healthy)
   Classes are taken from the checkpoint's label map; a condition without metadata raises ValueError (never guessed).
   The marginals already sum to 1 because every known condition belongs to exactly one crop and one health state.
4. If the most probable crop (or health state) by marginal differs from the one implied by the final condition, the
   prediction is ambiguous; this is reported as 'marginal_disagreement' and makes the result Uncertain.
5. 'raw_*' fields keep the independent heads' own argmax and softmax confidence, and 'head_disagreement' lists heads
   whose raw answer differs from the reconciled one.

LIMITS: this changes what is DISPLAYED, not what the network recognises. It cannot make a wrong condition right.

Cause: if fewer than two cause classes had training examples the cause head carries no information and is reported as
unavailable.

Severity: the two cumulative outputs should satisfy P(>Mild) >= P(>Moderate). The raw values are kept; the reported
values use the projection P'(>Moderate) = min(P(>Moderate), P(>Mild)) (the closest rank-consistent pair that leaves
P(>Mild) unchanged). Decoding with the projected pair gives the same class as model.decode_severity.

Uncertainty: confidences here are RAW (uncalibrated) softmax values or marginals of them, unless a temperature from
validation-only calibration is supplied. Thresholds below are PROVISIONAL (not fitted on any data) until validation
calibration is run; they do not and cannot catch every wrong prediction.
"""
import numpy as np

from class_meta import CONDITION_META

UNCERTAIN_MSG = "Uncertain prediction - please verify with a clearer image or an expert."
CAUSE_UNAVAILABLE = "unavailable"
CAUSE_UNAVAILABLE_MSG = ("Cause (biotic vs abiotic) is unavailable: the model was trained only on biotic examples, so it "
                         "cannot tell disease from nutrient or water stress.")
SEVERITY = ["Mild", "Moderate", "Severe"]
# PROVISIONAL defaults (not fitted on data); see module docstring.
DEFAULT_THRESHOLDS = {"condition": 0.5, "crop": 0.5, "health": 0.5}


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def aggregation_matrices(conditions, crops, meta=CONDITION_META):
    """-> (C x K crop membership, length-C healthy mask). Raises on any condition without metadata or unknown crop."""
    unknown = [c for c in conditions if c not in meta]
    if unknown:
        raise ValueError(f"no metadata for condition classes {unknown}; add them to class_meta.CONDITION_META")
    crop_idx = {k: i for i, k in enumerate(crops)}
    bad = sorted({meta[c][0] for c in conditions} - set(crop_idx))
    if bad:
        raise ValueError(f"metadata crops {bad} are not in the checkpoint crop list {list(crops)}")
    m = np.zeros((len(conditions), len(crops)))
    for i, c in enumerate(conditions):
        m[i, crop_idx[meta[c][0]]] = 1.0
    healthy = np.array([meta[c][1] for c in conditions], dtype=bool)
    return m, healthy


def project_ordinal(p_gt_mild, p_gt_moderate):
    """Rank-consistent projection. Returns (p_gt_mild, projected p_gt_moderate, raw-violation mask)."""
    p1 = np.asarray(p_gt_mild, dtype=np.float64)
    p2 = np.asarray(p_gt_moderate, dtype=np.float64)
    return p1, np.minimum(p2, p1), p2 > p1


def severity_class_probs(p1, p2_projected):
    """[P(Mild), P(Moderate), P(Severe)] from a rank-consistent cumulative pair (rows sum to 1, all >= 0)."""
    return np.stack([1.0 - p1, p1 - p2_projected, p2_projected], axis=-1)


def decode_ordinal(p1, p2_projected):
    """Class 0 unless P(>Mild) > 0.5; class 2 only if also P(>Moderate) > 0.5 (same rule as model.decode_severity)."""
    gt1 = (np.asarray(p1) > 0.5).astype(int)
    return gt1 * (1 + (np.asarray(p2_projected) > 0.5).astype(int))


def cause_available(trained_causes):
    return trained_causes is not None and len(set(trained_causes)) >= 2


def reconcile(logits, maps, trained_causes=None, temperature=None, thresholds=None):
    """Reconcile a batch of raw head logits.

    logits: dict with numpy arrays 'crop' (N,K), 'health' (N,2), 'cause' (N,2), 'condition' (N,C), 'severity' (N,2).
    maps: label maps from the checkpoint ('crop', 'health', 'cause', 'condition').
    temperature: optional dict {head: T} from validation-only calibration; None = raw softmax.
    Returns a list of N dicts (one per image).
    """
    temperature = temperature or {}
    th = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    crops, conds, health_names = list(maps["crop"]), list(maps["condition"]), list(maps["health"])
    member, healthy = aggregation_matrices(conds, crops)
    pc = softmax(logits["condition"], temperature.get("condition", 1.0))
    p_crop_marg = pc @ member
    p_healthy = pc[:, healthy].sum(1)
    raw = {h: softmax(logits[h], temperature.get(h, 1.0)) for h in ("crop", "health")}
    raw["cause"] = softmax(logits["cause"])
    p1, p2, viol = project_ordinal(*sigmoid(logits["severity"]).T)
    sev_probs = severity_class_probs(p1, p2)
    sev_cls = decode_ordinal(p1, p2)
    has_cause = cause_available(trained_causes)
    out = []
    for i in range(pc.shape[0]):
        ci = int(pc[i].argmax())
        cond = conds[ci]
        crop, is_healthy = CONDITION_META[cond]
        health = "Healthy" if is_healthy else "Unhealthy"
        r = {"condition": cond, "condition_conf": float(pc[i, ci]),
             "crop": crop, "crop_conf": float(p_crop_marg[i, crops.index(crop)]),
             "health": health, "health_conf": float(p_healthy[i] if is_healthy else 1.0 - p_healthy[i])}
        # ambiguity of the marginals relative to the chosen condition
        md = []
        if crops[int(p_crop_marg[i].argmax())] != crop:
            md.append("crop")
        if (p_healthy[i] > 0.5) != is_healthy and abs(p_healthy[i] - 0.5) > 1e-12:
            md.append("health")
        r["marginal_disagreement"] = ";".join(md)
        # raw independent heads, kept for diagnostics only
        rc = int(raw["crop"][i].argmax())
        rh = int(raw["health"][i].argmax())
        r.update({"raw_crop": crops[rc], "raw_crop_conf": float(raw["crop"][i, rc]),
                  "raw_health": health_names[rh], "raw_health_conf": float(raw["health"][i, rh])})
        hd = [h for h, a, b in (("crop", r["raw_crop"], crop), ("health", r["raw_health"], health)) if a != b]
        r["head_disagreement"] = ";".join(hd)
        # cause
        if is_healthy:
            r["cause"], r["cause_conf"] = "None", float("nan")
        elif not has_cause:
            r["cause"], r["cause_conf"] = CAUSE_UNAVAILABLE, float("nan")
        else:
            k = int(raw["cause"][i].argmax())
            r["cause"], r["cause_conf"] = maps["cause"][k], float(raw["cause"][i, k])
        # severity (shown only for unhealthy final health; raw values always kept)
        r["raw_p_gt_mild"], r["raw_p_gt_moderate"] = float(p1[i]), float(sigmoid(logits["severity"][i])[1])
        r["p_gt_moderate_projected"] = float(p2[i])
        r["ordinal_violation_raw"] = bool(viol[i])
        if is_healthy:
            r["severity"], r["severity_conf"] = "None", float("nan")
        else:
            r["severity"] = SEVERITY[int(sev_cls[i])]
            r["severity_conf"] = float(sev_probs[i, int(sev_cls[i])])
        # uncertainty
        reasons = []
        for h in ("condition", "crop", "health"):
            if r[h + "_conf"] < th[h]:
                reasons.append(f"low {h} confidence")
        if md:
            reasons.append("ambiguous " + "/".join(md))
        if hd:
            reasons.append("independent " + "/".join(hd) + " head disagrees")
        r["uncertain"] = bool(reasons)
        r["uncertain_reasons"] = "; ".join(reasons)
        out.append(r)
    return out


def count_inconsistent_raw(raw_crop, raw_health, condition):
    """Number of rows whose raw crop/health heads disagree with the condition (via metadata). Inputs are sequences."""
    n = 0
    for k, h, c in zip(raw_crop, raw_health, condition):
        crop, is_h = CONDITION_META[c]
        n += int(k != crop or (h == "Healthy") != is_h)
    return n
