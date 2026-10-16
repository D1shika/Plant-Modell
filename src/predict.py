"""Command-line predictor.

  python src/predict.py --image "path\\to\\leaf.jpg"
  python src/predict.py --image "path\\to\\leaf.jpg" --temp 28 --rainfall 10 --humidity 40 --soil_moisture 20
  python src/predict.py --folder "path\\to\\folder"

Outputs per image: crop, health, cause, condition, severity with confidences, a Grad-CAM overlay (condition head,
EfficientNet-B0 last conv block) and a CSV in outputs/predictions/.

CONFIDENCE = the model's softmax probability of the predicted class for that head (for severity: the probability of the
predicted ordinal class derived from the cumulative outputs). It is NOT a calibrated probability of being correct.
Cause: the model only saw Biotic examples (no abiotic data were available), so it can never meaningfully say Abiotic.
Severity: a heuristic visual-damage estimate, not an expert assessment. Environment inputs were SIMULATED in training;
if they are not given, the training-set means are used.

The model configuration, label maps and image size come from the checkpoint; the environment normalisation statistics
from models/env_stats.json (fitted on the training split). The checkpoint is loaded strictly: a mismatching or missing
checkpoint raises an error instead of predicting from random weights.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset import load_rgb_256, make_transform  # noqa: E402
from infer import GradCAM, get_device, load_variant, overlay  # noqa: E402
from model import SEVERITY, decode_severity  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_BYTES = 50 * 2**20
MAX_IMAGES = 5000
LOW_CONF = 0.5
ENV_ARGS = ("temp", "rainfall", "humidity", "soil_moisture")
ENV_RANGES = {"temp": (-50.0, 70.0), "rainfall": (0.0, 2000.0), "humidity": (0.0, 100.0), "soil_moisture": (0.0, 100.0)}
ENV_MSG = "Environment values not provided: using training averages"
LOW_MSG = "Low confidence - this may not be one of the 8 supported crops or may be a poor photo"
OUT_DIR = C.OUTPUTS / "predictions"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:60]


def check_image(path):
    """Return (256x256 RGB PIL image) or raise ValueError with a helpful message."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"file not found: {p}")
    if not p.is_file():
        raise ValueError(f"not a file: {p}")
    if p.suffix.lower() not in EXTS:
        raise ValueError(f"unsupported file type '{p.suffix}' (supported: {', '.join(sorted(EXTS))})")
    if p.stat().st_size > MAX_BYTES:
        raise ValueError(f"file larger than {MAX_BYTES // 2**20} MB")
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w, h = im.size
            if min(w, h) < 32:
                raise ValueError(f"image too small ({w}x{h}); minimum side is 32 px")
        return load_rgb_256(p)
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"not a readable image ({type(e).__name__}): {p}") from e


def env_values(args, stats):
    given = {k: getattr(args, k) for k in ENV_ARGS}
    means = dict(zip(ENV_ARGS, stats["mean"]))
    for k, v in given.items():
        if v is None:
            continue
        if not math.isfinite(v):
            fail(f"--{k} must be a finite number (got {v})")
        lo, hi = ENV_RANGES[k]
        if not lo <= v <= hi:
            fail(f"--{k}={v} is outside the plausible range [{lo}, {hi}]")
    missing = [k for k, v in given.items() if v is None]
    if missing:
        print(ENV_MSG if len(missing) == 4 else f"{ENV_MSG} (for: {', '.join(missing)})")
    vals = {k: (means[k] if v is None else float(v)) for k, v in given.items()}
    src = "training_average" if len(missing) == 4 else ("user" if not missing else "partly_training_average")
    return vals, src


def severity_probs(logits):
    p = torch.sigmoid(logits)
    p1, p2 = p[:, 0], torch.minimum(p[:, 1], p[:, 0])
    return torch.stack([1 - p1, (p1 - p2).clamp(min=0), p2], 1)


def main():
    ap = argparse.ArgumentParser(description="Leaf crop / health / cause / condition / severity predictor")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="path to one leaf photo")
    g.add_argument("--folder", help="folder of leaf photos (non-recursive)")
    for k in ENV_ARGS:
        ap.add_argument(f"--{k}", type=float, default=None, help=f"{k} (optional; training average if omitted)")
    ap.add_argument("--variant", default="fusion", choices=["image", "env", "fusion", "fusion_shuffled_env"])
    ap.add_argument("--no-cam", action="store_true", help="skip Grad-CAM overlays")
    ap.add_argument("--max_images", type=int, default=MAX_IMAGES)
    args = ap.parse_args()
    
    stats = json.load(open(C.MODELS / "env_stats.json"))
    dev = get_device()
    try:
        model, maps, cfg = load_variant(args.variant, "best", dev)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        fail(str(e))
    if stats["features"] != maps["env_features"]:
        fail("env_stats.json feature order does not match the checkpoint's label maps")
    # cause classes that had training examples (older checkpoints lack the key; models/label_maps.json is current)
    try:
        trained_causes = json.load(open(C.MODELS / "label_maps.json")).get("cause_trained_classes", ["Biotic"])
    except (OSError, ValueError):
        trained_causes = ["Biotic"]
    env, env_src = env_values(args, stats)
    env_norm = (np.array([env[k] for k in ENV_ARGS]) - np.array(stats["mean"])) / np.array(stats["std"])

    if args.image:
        paths = [Path(args.image)]
    else:
        d = Path(args.folder)
        if not d.is_dir():
            fail(f"folder not found: {d}")
        paths = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        if not paths:
            fail(f"no supported images in {d}")
        if len(paths) > args.max_images:
            fail(f"{len(paths)} images exceeds --max_images {args.max_images}; raise it explicitly if intended")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    tf = make_transform(False, cfg["img_size"])
    rows, valid_imgs, valid_paths = [], [], []
    for p in paths:
        try:
            valid_imgs.append(check_image(p))
            valid_paths.append(p)
        except ValueError as e:
            print(f"SKIPPED {p.name}: {e}")
            rows.append({"image": str(p), "status": f"error: {e}"})
    if not valid_imgs:
        fail("no valid images to process")

    results = []
    envt = torch.from_numpy(env_norm).float().to(dev)
    for s in range(0, len(valid_imgs), 16):
        batch = valid_imgs[s:s + 16]
        x = torch.stack([tf(i) for i in batch]).to(dev)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x, envt.expand(len(batch), -1))
        out = {k: v.float().cpu() for k, v in out.items()}
        probs = {h: torch.softmax(out[h], 1) for h in ("crop", "health", "cause", "condition")}
        sev = decode_severity(out["severity"])
        sprob = severity_probs(out["severity"])
        sp = torch.sigmoid(out["severity"])
        for i in range(len(batch)):
            r = {}
            for h, names in (("crop", maps["crop"]), ("health", maps["health"]), ("cause", maps["cause"]),
                             ("condition", maps["condition"])):
                c, k = probs[h][i].max(0)
                r[h], r[h + "_conf"] = names[int(k)], float(c)
            unhealthy = r["health"] == "Unhealthy"
            # only classes seen in training can be reported; with a single trained class the head carries no information
            ti = [maps["cause"].index(c) for c in trained_causes]
            if len(ti) == 1:
                r["cause"], r["cause_conf"] = maps["cause"][ti[0]], float("nan")
            else:
                pc = probs["cause"][i][ti]
                pc = pc / pc.sum()
                r["cause"], r["cause_conf"] = maps["cause"][ti[int(pc.argmax())]], float(pc.max())
            if not unhealthy:
                r["cause"], r["cause_conf"] = "None", float("nan")
            r["severity"] = SEVERITY[int(sev[i])] if unhealthy else "None"
            r["severity_conf"] = float(sprob[i, int(sev[i])]) if unhealthy else float("nan")
            r["p_gt_mild"], r["p_gt_moderate"] = float(sp[i, 0]), float(sp[i, 1])
            relevant = ["crop", "health", "condition"] + (["severity"] if unhealthy else [])
            if unhealthy and len(trained_causes) > 1:
                relevant.append("cause")
            r["low_confidence_heads"] = ";".join(h for h in relevant if r[h + "_conf"] < LOW_CONF)
            results.append(r)

    cams = {}
    if not args.no_cam and model.use_img:
        gc = GradCAM(model)
        for p, img in zip(valid_paths, valid_imgs):
            x = tf(img).unsqueeze(0).to(dev)
            cam, _ = gc(x, envt.unsqueeze(0), "condition")
            c224 = cv2.resize(cam[0].float().cpu().numpy(), (224, 224), interpolation=cv2.INTER_CUBIC).clip(0, 1)
            base = np.asarray(img.resize((224, 224)))
            panel = np.hstack([base, overlay(base, c224)])
            cp = run_dir / f"cam_{safe(p.stem)}_{hashlib.md5(str(p).encode('utf-8')).hexdigest()[:6]}.png"
            ok, buf = cv2.imencode(".png", panel[:, :, ::-1])
            cp.write_bytes(buf.tobytes())
            cams[str(p)] = str(cp)
    elif not args.no_cam:
        print("Grad-CAM skipped: this variant has no image branch")

    for p, r in zip(valid_paths, results):
        print(f"\nImage: {p}")
        print(f"  Crop:       {r['crop']} (confidence {r['crop_conf']:.2f})")
        print(f"  Health:     {r['health']} (confidence {r['health_conf']:.2f})")
        if r["health"] == "Unhealthy":
            if r["cause_conf"] != r["cause_conf"]:  # NaN: single trained class
                print(f"  Cause:      {r['cause']} [only class seen in training; no abiotic data, so abiotic stress "
                      f"cannot be identified - not a meaningful prediction]")
            else:
                print(f"  Cause:      {r['cause']} (confidence {r['cause_conf']:.2f})")
        else:
            print("  Cause:      None (healthy)")
        print(f"  Condition:  {r['condition']} (confidence {r['condition_conf']:.2f})")
        if r["health"] == "Unhealthy":
            print(f"  Severity:   {r['severity']} (confidence {r['severity_conf']:.2f}) [heuristic damage estimate]")
        else:
            print("  Severity:   None (healthy)")
        if str(p) in cams:
            print(f"  Grad-CAM:   {cams[str(p)]}")
        if r["low_confidence_heads"]:
            print(f"  {LOW_MSG}  (heads below {LOW_CONF}: {r['low_confidence_heads'].replace(';', ', ')})")
        rows.append({"image": str(p), "status": "ok", **{k: (round(v, 4) if isinstance(v, float) else v)
                                                        for k, v in r.items()}, **{k: round(env[k], 3) for k in ENV_ARGS},
                     "env_source": env_src, "cam_path": cams.get(str(p), ""), "variant": args.variant})
    print(f"\nConfidence = softmax probability of the predicted class (not a calibrated probability).")
    cols = ["image", "status", "crop", "crop_conf", "health", "health_conf", "cause", "cause_conf", "condition",
            "condition_conf", "severity", "severity_conf", "p_gt_mild", "p_gt_moderate", "low_confidence_heads",
            *ENV_ARGS, "env_source", "cam_path", "variant"]
    csv_path = run_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
