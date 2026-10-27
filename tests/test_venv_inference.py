"""Integration tests that need the project venv (torch, real checkpoints). Read-only for project files: every prediction is
written to a temporary folder, and the test fails if models/label_maps.json or outputs/predictions/ change.

Run:  .venv\\Scripts\\python.exe -I tests\\test_venv_inference.py
Time: ~1-2 min on CPU (several small predictor runs on 1-6 images). No internet, no training.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
RESULTS = []
PY = sys.executable


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_predict(args, cpu=True):
    env = dict(os.environ)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([PY, "-I", str(SRC / "predict.py"), *args], cwd=ROOT, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def consistent(df):
    from class_meta import CONDITION_META
    ok = df[df.status == "ok"]
    return all(CONDITION_META[c][0] == k and CONDITION_META[c][1] == (h == "Healthy")
               for c, k, h in zip(ok.condition, ok.crop, ok.health))


label_maps_before = sha(ROOT / "models" / "label_maps.json")
pred_dir = ROOT / "outputs" / "predictions"
pred_listing_before = sorted(p.name for p in pred_dir.iterdir()) if pred_dir.exists() else []

import torch  # noqa: E402
from class_meta import validate  # noqa: E402
from infer import load_variant  # noqa: E402
import predict  # noqa: E402

# ---- checkpoint loading (strict) and metadata
for v in ("image", "fusion", "fusion_shuffled_env", "env"):
    try:
        m, maps, cfg = load_variant(v, "best", torch.device("cpu"))
        check(f"{v}: best.pt loads strictly on CPU", cfg["variant"] == v and validate(maps["condition"], maps["crop"]) == [])
    except Exception as e:  # noqa: BLE001
        check(f"{v}: best.pt loads strictly on CPU", False, repr(e))
check("default predictor variant is image-only", predict.build_parser().parse_args(["--image", "x.jpg"]).variant == "image")

demo_pv = sorted((ROOT / "outputs" / "predict_demo" / "PlantVillage").glob("*"))
demo_pd = sorted((ROOT / "outputs" / "predict_demo" / "PlantDoc").glob("*"))
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # ---- single image, default (image-only), CPU, with Grad-CAM
    rc, out = run_predict(["--image", str(demo_pd[0]), "--out_dir", str(tmp / "single")])
    df = pd.read_csv(tmp / "single" / "predictions.csv") if rc == 0 else None
    check("single image, default variant, CPU: exit 0 and CSV written", rc == 0 and df is not None and len(df) == 1, out[-300:])
    if df is not None:
        check("single image: CSV variant is 'image' and env not used", df.variant[0] == "image" and df.env_source[0] == "not_used")
        check("single image: crop/health/condition consistent", consistent(df))
        check("single image: raw head columns present", {"raw_crop", "raw_health", "head_disagreement"} <= set(df.columns))
        check("Grad-CAM overlay written to the temp folder", any((tmp / "single").glob("cam_*.png")))
        check("no simulated-environment prompt in default output", "EXPERIMENTAL" not in out and "Environment values" not in out)
    # ---- batch (folder)
    rc, out = run_predict(["--folder", str(demo_pv[0].parent), "--out_dir", str(tmp / "batch"), "--no-cam"])
    df = pd.read_csv(tmp / "batch" / "predictions.csv") if rc == 0 else None
    check("batch folder: exit 0, one row per image", rc == 0 and df is not None and len(df) == len(demo_pv), out[-300:])
    if df is not None:
        check("batch: all rows consistent", consistent(df))
        unh = df[(df.status == "ok") & (df.health == "Unhealthy")]
        check("batch: cause shown as unavailable for unhealthy leaves", len(unh) > 0 and (unh.cause == "unavailable").all())
        check("batch: no ordinal violation after projection",
              (df.p_gt_moderate_projected <= df.raw_p_gt_mild + 1e-9).all())
    # ---- environment flags
    rc, out = run_predict(["--image", str(demo_pd[0]), "--temp", "25", "--out_dir", str(tmp / "envimg"), "--no-cam"])
    check("env flag with the image-only default is rejected with an explanation", rc != 0 and "fusion" in out)
    rc, out = run_predict(["--image", str(demo_pd[0]), "--variant", "fusion", "--temp", "25", "--rainfall", "50",
                           "--humidity", "60", "--soil_moisture", "45", "--out_dir", str(tmp / "fusion"), "--no-cam"])
    check("experimental fusion still runs with env values and says EXPERIMENTAL", rc == 0 and "EXPERIMENTAL" in out, out[-300:])
    rc, out = run_predict(["--image", str(demo_pd[0]), "--variant", "fusion", "--humidity", "250",
                           "--out_dir", str(tmp / "fbad"), "--no-cam"])
    check("out-of-range env value rejected", rc != 0 and "outside the plausible range" in out)
    # ---- awkward inputs (EXIF rotation and letterboxing are NOT changed in Phase B; this checks they are handled)
    from PIL import Image
    odd = tmp / "odd"
    odd.mkdir()
    src = Image.open(demo_pv[0]).convert("RGB")
    src.resize((300, 500)).save(odd / "portrait.jpg")
    src.resize((500, 300)).save(odd / "landscape.jpg")
    rgba = src.convert("RGBA")
    rgba.putalpha(128)
    rgba.save(odd / "transparent.png")
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 CW
    src.save(odd / "exif_rotated.jpg", exif=exif)
    src.resize((20, 20)).save(odd / "tiny.jpg")
    (odd / "corrupt.jpg").write_bytes(b"not an image at all")
    src.save(odd / "image.bmp")
    rc, out = run_predict(["--folder", str(odd), "--out_dir", str(tmp / "oddout"), "--no-cam"])
    df = pd.read_csv(tmp / "oddout" / "predictions.csv") if rc == 0 else None
    check("odd-input folder: exit 0", rc == 0 and df is not None, out[-300:])
    if df is not None:
        st = dict(zip(df.image.map(lambda s: Path(s).name), df.status))
        check("corrupt file skipped with an error row", str(st.get("corrupt.jpg", "")).startswith("error"))
        check("tiny (<32 px) image skipped with an error row", "too small" in str(st.get("tiny.jpg", "")))
        check("portrait, landscape, transparent PNG, BMP and EXIF JPEG accepted",
              all(st.get(n) == "ok" for n in ("portrait.jpg", "landscape.jpg", "transparent.png", "image.bmp", "exif_rotated.jpg")))
    # ---- GPU path only when CUDA exists
    if torch.cuda.is_available():
        rc, out = run_predict(["--image", str(demo_pd[0]), "--out_dir", str(tmp / "gpu")], cpu=False)
        check("GPU inference (CUDA available)", rc == 0, out[-300:])
    else:
        print("SKIP GPU inference: CUDA not available")

check("models/label_maps.json unchanged by the tests", sha(ROOT / "models" / "label_maps.json") == label_maps_before)
check("outputs/predictions/ unchanged by the tests",
      (sorted(p.name for p in pred_dir.iterdir()) if pred_dir.exists() else []) == pred_listing_before)
n_ok = sum(ok for _, ok in RESULTS)
print(f"\n{n_ok}/{len(RESULTS)} venv inference checks passed")
sys.exit(0 if n_ok == len(RESULTS) else 1)
