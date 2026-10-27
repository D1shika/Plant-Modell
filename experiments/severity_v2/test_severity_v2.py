"""Synthetic-image tests for experiments/severity_v2/severity_v2.py (no project data, writes nothing outside a tempdir).
Run from the project root:  .venv\\Scripts\\python.exe -I experiments\\severity_v2\\test_severity_v2.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from severity_v2 import analyse_v2, fill_holes, label, load  # noqa: E402
from severity import analyse  # noqa: E402

CATS = {"Mild", "Moderate", "Severe", "Invalid"}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def leaf_img(bg=(128, 128, 128)):
    img = np.full((200, 200, 3), bg, np.uint8)
    cv2.ellipse(img, (100, 100), (75, 55), 0, 0, 360, (40, 170, 40), -1)  # healthy green, H~60, S high
    return img


def v1(img):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"
        cv2.imwrite(str(p), img)
        _, _, _, f, _, flag = analyse(p)
    return f, flag


# fill_holes
ring = np.zeros((50, 50), bool)
cv2.circle(ring.view(np.uint8), (25, 25), 15, 1, 3)
filled = fill_holes(ring)
check("fill_holes fills an enclosed hole", filled[25, 25] and filled.sum() > ring.sum())
c_shape = ring.copy()
c_shape[20:31, 35:45] = False  # open the ring
check("fill_holes leaves an open (border-connected) gap unfilled", not fill_holes(c_shape)[25, 25])
check("fill_holes on an empty mask stays empty", fill_holes(np.zeros((20, 20), bool)).sum() == 0)
check("fill_holes keeps every original pixel", (fill_holes(ring) | ring).sum() == fill_holes(ring).sum())

# healthy uniform leaf
r = analyse_v2(leaf_img())
s, ok = label(r)
check("healthy green leaf: flag ok", r["flag"] == "ok", r["flag"])
check("healthy green leaf: near-zero damage", r["damage_fraction"] < 0.02, f"{r['damage_fraction']:.4f}")
check("healthy green leaf: Mild and valid", s == "Mild" and ok)

# white symptom enclosed in the leaf
img = leaf_img()
cv2.circle(img, (100, 100), 20, (245, 245, 245), -1)
r = analyse_v2(img)
f1, _ = v1(img)
check("white enclosed patch: v1 misses it (known defect)", f1 < 0.02, f"v1={f1:.4f}")
check("white enclosed patch: v2 counts it", r["damage_fraction"] > 0.08, f"v2={r['damage_fraction']:.4f}")
check("white enclosed patch: counted as pale component", r["pale"] > 0.08 and r["restored_lowsat_px"] > 0)

# grey symptom enclosed (low saturation, mid value)
img = leaf_img(bg=(200, 160, 120))
cv2.circle(img, (80, 100), 15, (150, 150, 150), -1)
r = analyse_v2(img)
check("grey enclosed patch: v2 counts it", r["damage_fraction"] > 0.04, f"v2={r['damage_fraction']:.4f}")

# low-saturation region touching the leaf margin stays excluded (same as v1)
img = leaf_img()
cv2.circle(img, (175, 100), 18, (128, 128, 128), -1)  # bites into the leaf edge, same grey as backdrop
r = analyse_v2(img)
check("grey notch connected to background: not counted", r["damage_fraction"] < 0.03, f"v2={r['damage_fraction']:.4f}")

# low-saturation object outside the leaf does not change the result
base = analyse_v2(leaf_img())["damage_fraction"]
img = leaf_img()
cv2.rectangle(img, (2, 2), (25, 25), (250, 250, 250), -1)
check("white object in the background: no extra damage", abs(analyse_v2(img)["damage_fraction"] - base) < 0.01)

# very small damaged region
img = leaf_img()
cv2.circle(img, (110, 95), 4, (30, 60, 140), -1)  # small brown spot
r = analyse_v2(img)
f1, _ = v1(img)
check("small brown spot: small positive damage, Mild", 0 < r["damage_fraction"] < 0.05 and label(r)[0] == "Mild",
      f"v2={r['damage_fraction']:.4f}")
check("small saturated spot: v1 and v2 agree (change targets low-sat only)", abs(r["damage_fraction"] - f1) < 0.01,
      f"v1={f1:.4f} v2={r['damage_fraction']:.4f}")
check("small brown spot: necrotic component, no pale", r["necrotic"] > 0 and r["pale"] == 0)

# chlorosis (yellow, saturated)
img = leaf_img()
cv2.ellipse(img, (100, 100), (40, 30), 0, 0, 360, (40, 200, 220), -1)
r = analyse_v2(img)
check("yellow chlorosis counted as chlorotic", r["chlorotic"] > 0.15, f"chl={r['chlorotic']:.3f}")

# pale-dominant leaf -> invalid
img = leaf_img()
cv2.ellipse(img, (100, 100), (60, 42), 0, 0, 360, (235, 235, 235), -1)
r = analyse_v2(img)
s, ok = label(r)
check("pale-dominant leaf flagged and invalid", r["flag"] == "pale_dominant" and not ok, f"{r['flag']} pale={r['pale']}")
check("pale-dominant category still in valid set", s in CATS)

# empty / invalid masks
r = analyse_v2(np.full((200, 200, 3), 128, np.uint8))
s, ok = label(r)
check("uniform grey image: mask_failed, Invalid, not valid", r["flag"] == "mask_failed" and s == "Invalid" and not ok,
      r["flag"])
r = analyse_v2(np.zeros((200, 200, 3), np.uint8))
check("black image: Invalid, not valid", label(r) == ("Invalid", False), r["flag"])
r = analyse_v2(np.full((200, 200, 3), (40, 170, 40), np.uint8))
s, ok = label(r)
check("uniform green frame: category valid set, not marked valid unless flag ok",
      s in CATS and (ok == (r["flag"] == "ok")), f"{r['flag']} {s}")
with tempfile.TemporaryDirectory() as d:
    bad = Path(d) / "bad.jpg"
    bad.write_bytes(b"not an image")
    try:
        load(bad)
        check("unreadable file raises ValueError", False)
    except ValueError:
        check("unreadable file raises ValueError", True)

# determinism and output contract
img = leaf_img()
cv2.circle(img, (100, 100), 20, (245, 245, 245), -1)
before = img.copy()
a, b = analyse_v2(img), analyse_v2(img)
check("deterministic: identical masks and fraction", np.array_equal(a["damage"], b["damage"])
      and a["damage_fraction"] == b["damage_fraction"])
check("input image not modified", np.array_equal(img, before))
big = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_NEAREST)
check("large input resized to max side 256", max(analyse_v2(big)["img"].shape[:2]) == 256)
check("components sum to damage fraction",
      abs(a["chlorotic"] + a["necrotic"] + a["pale"] + a["dark"] - a["damage_fraction"]) < 1e-9)

n_ok = sum(ok for _, ok in results)
print(f"\n{n_ok}/{len(results)} passed")
sys.exit(0 if n_ok == len(results) else 1)
