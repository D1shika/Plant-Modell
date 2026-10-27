"""Command-line predictor.

  python src/predict.py --image "path\\to\\leaf.jpg"
  python src/predict.py --folder "path\\to\\folder"
  python src/predict.py --image "leaf.jpg" --variant fusion --temp 28 --rainfall 10 --humidity 40 --soil_moisture 20
                                                                          (EXPERIMENTAL fusion model, simulated inputs)

Default model: image-only (models/image/best.pt). It matches the fusion model on every test metric (condition accuracy
0.967 vs 0.969, overlapping 95% CIs; outputs/tables/table3_modality_ablation.md) and needs no environment values.

Outputs per image: crop, health, cause, condition, severity, an uncertainty flag, a Grad-CAM overlay (condition head,
EfficientNet-B0 last conv block) and a CSV in outputs/predictions/<timestamp>/ (or --out_dir).

CONSISTENCY: crop and health are derived from the condition head (src/reconcile.py), so the three always agree. The
independent crop/health heads are still computed and written to the CSV as raw_* columns; 'head_disagreement' lists
heads whose own answer differs. This makes the displayed answer consistent; it does not change recognition accuracy.

CONFIDENCE: condition_conf = softmax probability of the chosen condition; crop_conf / health_conf = probability mass of
all conditions belonging to that crop / health state. These are RAW (uncalibrated) values unless
models/calibration/<variant>.json (validation-only temperature scaling) exists. They are NOT probabilities of being
correct. The uncertainty thresholds are provisional unless that calibration file supplies them.

CAUSE: the training data contain no abiotic examples, so cause is reported as unavailable for unhealthy leaves.
SEVERITY: heuristic visual-damage estimate (not expert-validated); raw cumulative outputs and the rank-consistent
projection are both written to the CSV.
ENVIRONMENT (fusion variants only, EXPERIMENTAL): the four values were SIMULATED during training, are not real sensor
measurements, and showed no measurable effect on predictions; they cannot establish any link between environment and
disease. If omitted with a fusion variant, the training-set means are used.

The model configuration, label maps and image size come from the checkpoint; it is loaded strictly, so a mismatching or
missing checkpoint raises an error instead of predicting from random weights.
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
from class_meta import validate as validate_meta  # noqa: E402
from dataset import load_rgb_256, make_transform  # noqa: E402
from infer import GradCAM, get_device, load_variant, overlay  # noqa: E402
from reconcile import CAUSE_UNAVAILABLE, CAUSE_UNAVAILABLE_MSG, UNCERTAIN_MSG, reconcile  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_BYTES = 50 * 2**20
MAX_IMAGES = 5000
VARIANTS = ["image", "env", "fusion", "fusion_shuffled_env"]
ENV_VARIANTS = {"env", "fusion", "fusion_shuffled_env"}
ENV_ARGS = ("temp", "rainfall", "humidity", "soil_moisture")
ENV_RANGES = {"temp": (-50.0, 70.0), "rainfall": (0.0, 2000.0), "humidity": (0.0, 100.0), "soil_moisture": (0.0, 100.0)}
ENV_MSG = "Environment values not provided: using training averages"
EXPERIMENTAL_MSG = ("EXPERIMENTAL variant '{v}': trained with SIMULATED environment values (not sensor data). They had no "
                    "measurable effect on predictions; use the default image-only model for normal use.")
OUT_DIR = C.OUTPUTS / "predictions"
CALIB_DIR = C.MODELS / "calibration"


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


def load_calibration(variant):
    """models/calibration/<variant>.json from validation-only calibration (src/calibrate.py), or None."""
    p = CALIB_DIR / f"{variant}.json"
    if not p.exists():
        return None
    cal = json.load(open(p))
    if cal.get("fitted_on") != "val":
        fail(f"{p} was not fitted on the validation split; refusing to use it")
    return cal


def build_parser():
    ap = argparse.ArgumentParser(description="Leaf crop / health / condition / severity predictor")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="path to one leaf photo")
    g.add_argument("--folder", help="folder of leaf photos (non-recursive)")
    for k in ENV_ARGS:
        ap.add_argument(f"--{k}", type=float, default=None,
                        help=f"{k}: EXPERIMENTAL, only with a fusion/env variant (simulated during training)")
    ap.add_argument("--variant", default="image", choices=VARIANTS,
                    help="model to use (default: image). fusion / fusion_shuffled_env / env are experimental")
    ap.add_argument("--no-cam", action="store_true", help="skip Grad-CAM overlays")
    ap.add_argument("--max_images", type=int, default=MAX_IMAGES)
    ap.add_argument("--out_dir", default=None, help="folder for results (default: outputs/predictions/<timestamp>)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    env_given = [k for k in ENV_ARGS if getattr(args, k) is not None]
    if env_given and args.variant not in ENV_VARIANTS:
        fail(f"--{', --'.join(env_given)} only apply to the experimental fusion variants (the default image-only model "
             f"does not use environment values). Remove them, or add --variant fusion.")

    stats = json.load(open(C.MODELS / "env_stats.json"))
    dev = get_device()
    try:
        model, maps, cfg = load_variant(args.variant, "best", dev)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        fail(str(e))
    problems = validate_meta(maps["condition"], maps["crop"])
    if problems:
        fail("class metadata does not match the checkpoint: " + " | ".join(problems))
    if stats["features"] != maps["env_features"]:
        fail("env_stats.json feature order does not match the checkpoint's label maps")
    # cause classes that had training examples (older checkpoints lack the key; models/label_maps.json is current)
    trained_causes = maps.get("cause_trained_classes")
    if trained_causes is None:
        try:
            trained_causes = json.load(open(C.MODELS / "label_maps.json")).get("cause_trained_classes", ["Biotic"])
        except (OSError, ValueError):
            trained_causes = ["Biotic"]
    cal = load_calibration(args.variant)
    temperature = cal.get("temperature") if cal else None
    thresholds = cal.get("thresholds") if cal else None
    conf_kind = "calibrated (validation temperature scaling)" if cal else "raw softmax (uncalibrated)"

    uses_env = args.variant in ENV_VARIANTS
    if uses_env:
        print(EXPERIMENTAL_MSG.format(v=args.variant))
        env, env_src = env_values(args, stats)
        env_norm = (np.array([env[k] for k in ENV_ARGS]) - np.array(stats["mean"])) / np.array(stats["std"])
    else:  # the image-only network has no environment branch; a placeholder tensor is passed and ignored
        env, env_src = {k: float("nan") for k in ENV_ARGS}, "not_used"
        env_norm = np.zeros(len(ENV_ARGS))

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

    run_dir = Path(args.out_dir) if args.out_dir else OUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
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
        logits = {k: v.float().cpu().numpy() for k, v in out.items()}
        results += reconcile(logits, maps, trained_causes, temperature, thresholds)

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
        if r["uncertain"]:
            print(f"  RESULT:     {UNCERTAIN_MSG}")
            print(f"              (why: {r['uncertain_reasons']}; best guess shown below)")
        print(f"  Crop:       {r['crop']} (confidence {r['crop_conf']:.2f})")
        print(f"  Health:     {r['health']} (confidence {r['health_conf']:.2f})")
        if r["health"] == "Unhealthy":
            if r["cause"] == CAUSE_UNAVAILABLE:
                print("  Cause:      unavailable (no abiotic training data; cannot tell disease from nutrient/water stress)")
            else:
                print(f"  Cause:      {r['cause']} (confidence {r['cause_conf']:.2f})")
        else:
            print("  Cause:      None (healthy)")
        print(f"  Condition:  {r['condition']} (confidence {r['condition_conf']:.2f})")
        if r["health"] == "Unhealthy":
            print(f"  Severity:   {r['severity']} (confidence {r['severity_conf']:.2f}) [heuristic damage estimate]")
        else:
            print("  Severity:   None (healthy)")
        if r["head_disagreement"]:
            print(f"  Diagnostic: the model's separate {r['head_disagreement'].replace(';', '/')} output disagreed "
                  f"(raw crop {r['raw_crop']}, raw health {r['raw_health']}); the answer above follows the condition.")
        if str(p) in cams:
            print(f"  Grad-CAM:   {cams[str(p)]}")
        rows.append({"image": str(p), "status": "ok",
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                     **{k: (round(env[k], 3) if uses_env else "") for k in ENV_ARGS},
                     "env_source": env_src, "confidence_type": conf_kind, "cam_path": cams.get(str(p), ""),
                     "variant": args.variant})
    print(f"\nConfidence = {conf_kind}; not a probability of being correct. Uncertainty thresholds are "
          f"{'from validation calibration' if thresholds else 'provisional'}.")
    if any(r["health"] == "Unhealthy" for r in results) and trained_causes is not None and len(set(trained_causes)) < 2:
        print(CAUSE_UNAVAILABLE_MSG)
    cols = ["image", "status", "uncertain", "uncertain_reasons", "crop", "crop_conf", "health", "health_conf",
            "condition", "condition_conf", "cause", "cause_conf", "severity", "severity_conf",
            "raw_crop", "raw_crop_conf", "raw_health", "raw_health_conf", "head_disagreement", "marginal_disagreement",
            "raw_p_gt_mild", "raw_p_gt_moderate", "p_gt_moderate_projected", "ordinal_violation_raw",
            *ENV_ARGS, "env_source", "confidence_type", "cam_path", "variant"]
    csv_path = run_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Results written to {csv_path}")
    return results


if __name__ == "__main__":
    main()
