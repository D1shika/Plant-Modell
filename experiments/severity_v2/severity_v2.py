"""severity_v2: EXPERIMENTAL revision of the heuristic damage-fraction labeller (src/severity.py, "v1").  Pilot only.

Nothing here is used by training, evaluation or predict.py. Labels from this module are heuristic, not expert-verified.

Single targeted change versus v1 (everything else is kept identical so differences are attributable):
  v1 builds the leaf mask as  GrabCut  AND  (healthy-green OR saturation >= 40)   (src/severity.py:78).
  Low-saturation pixels (white powdery mildew, grey lesion centres, pale necrosis) are therefore removed from the LEAF
  and can never be counted as damage, so white/grey symptoms are systematically under-counted.
  v2 keeps that saturated "core" only to find the leaf outline, then fills the holes inside the outline: low-saturation
  pixels are restored to the leaf ONLY where they are enclosed by leaf tissue. Low-saturation pixels connected to the
  outside of the leaf (grey backdrop, cast shadows at the margin) stay excluded, as in v1.

Unchanged from v1: image resize (max side 256), GrabCut initialisation and iterations, the 5%/97% mask-validity rules and
the HSV fallback, the "healthy green" definition (H 35-85, S >= 40, V >= 40), the erosion (5x5, 2 iterations), the
damage definition (leaf pixels that are not healthy green), the 2% minimum-area failure rule, and the category
thresholds (config.SEVERITY_THRESHOLDS: Mild < 0.10 <= Moderate <= 0.30 < Severe). Category semantics are not redefined.

Added for transparency (do not change the damage fraction): the damage is split into
  chlorotic (yellowing: S >= 40, V >= 40, H 15-34), necrotic/other coloured (S >= 40, V >= 40, other non-green hues),
  pale (white/grey: S < 40, V >= 40) and dark (V < 40) fractions of the leaf.
Added failure rule: if pale pixels exceed 50% of the leaf the result is flagged 'pale_dominant' and marked invalid,
because that pattern is as consistent with glare or a leaked backdrop as with real disease.

Known limitations: white/grey symptoms that touch the leaf margin are not enclosed and stay excluded; glare and
specular highlights inside the leaf are counted as pale damage; veins and natural pale midribs can add damage on healthy
leaves; GrabCut failures are inherited from v1; thresholds were not re-fitted.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from severity import ERODE, GREEN_H, MAX_SIDE, MIN_SV, _hsv_fallback, severity_class  # noqa: E402  (read-only reuse)

VERSION = "severity_v2-pilot-0.1"
PALE_DOMINANT = 0.50
CLOSE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def fill_holes(mask):
    """Boolean mask -> mask with every background region not connected to the image border filled in."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    flood = pad.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)  # paint all outside background with 1
    holes = flood[1:-1, 1:-1] == 0             # never reached from the border = enclosed
    return mask.astype(bool) | holes


def load(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img


def analyse_v2(img_bgr):
    """BGR uint8 image -> dict with damage_fraction, components, flag, masks. Deterministic for a given image."""
    cv2.setNumThreads(1)
    img = img_bgr
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * .05), int(h * .05), int(w * .9), int(h * .9))
    try:
        cv2.grabCut(img, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
        gc = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        gc = np.zeros((h, w), bool)
    share = gc.sum() / (h * w)
    if share < 0.05:
        gc, flag = _hsv_fallback(hsv), "hsv_fallback"
    elif share > 0.97:
        flag = "leaf_fills_frame"
    else:
        flag = "ok"
    green = (H >= GREEN_H[0]) & (H <= GREEN_H[1]) & (S >= MIN_SV) & (V >= MIN_SV)
    core = gc & (green | (S >= MIN_SV))                      # = v1 leaf before erosion
    outline = fill_holes(cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, CLOSE_K).astype(bool))
    leaf = outline & gc                                        # low-sat pixels kept only where enclosed by leaf
    leaf = cv2.erode(leaf.astype(np.uint8), ERODE, iterations=2).astype(bool)
    damage = leaf & ~green
    area = int(leaf.sum())
    res = {"img": img, "leaf": leaf, "damage": damage, "leaf_share": area / (h * w), "flag": flag,
           "restored_lowsat_px": int((leaf & ~core).sum())}
    if area < 0.02 * h * w:
        res.update(damage_fraction=float("nan"), flag="mask_failed", chlorotic=float("nan"), necrotic=float("nan"),
                   pale=float("nan"), dark=float("nan"))
        return res
    sat, bright = S >= MIN_SV, V >= MIN_SV
    chl = damage & sat & bright & (H >= 15) & (H < GREEN_H[0])
    pale = damage & ~sat & bright
    dark = damage & ~bright
    nec = damage & ~chl & ~pale & ~dark
    res.update(damage_fraction=float(damage.sum() / area), chlorotic=float(chl.sum() / area),
               necrotic=float(nec.sum() / area), pale=float(pale.sum() / area), dark=float(dark.sum() / area))
    if res["pale"] > PALE_DOMINANT and flag == "ok":
        res["flag"] = "pale_dominant"
    return res


def label(res):
    """(severity category or 'Invalid', valid flag) with v1 semantics; v1 validity = flag 'ok' and a finite fraction."""
    f = res["damage_fraction"]
    sev = severity_class(f) if f == f else "Invalid"
    return sev, bool(res["flag"] == "ok" and sev != "Invalid")


def overlay_components(res):
    """BGR panel colouring the v2 damage components: chlorotic yellow, necrotic red, pale cyan, dark blue."""
    out = res["img"].copy()
    if res["damage_fraction"] != res["damage_fraction"]:
        return out
    hsv = cv2.cvtColor(res["img"], cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    d = res["damage"]
    sat, bright = S >= MIN_SV, V >= MIN_SV
    chl = d & sat & bright & (Hh >= 15) & (Hh < GREEN_H[0])
    pale, dark = d & ~sat & bright, d & ~bright
    nec = d & ~chl & ~pale & ~dark
    for m, col in ((chl, (0, 255, 255)), (nec, (0, 0, 255)), (pale, (255, 255, 0)), (dark, (255, 0, 0))):
        out[m] = col
    return out
