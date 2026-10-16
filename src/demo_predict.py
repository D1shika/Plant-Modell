"""Phase H: run predict.py on 3 reproducibly selected held-out TEST images per available source (seed 42).

Copies the chosen images to outputs/predict_demo/<source>/ (originals untouched), runs the real command-line
predictor on each folder, and records which sources are unavailable. Usage: python src/demo_predict.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from log_util import log  # noqa: E402

SOURCES = ["PlantVillage", "PlantDoc", "OLID_I"]  # OLID I was never obtained (Kaggle credentials not used)

df = pd.read_csv(C.DATA_PROCESSED / "splits.csv")
test = df[df["split"] == "test"]
out = C.OUTPUTS / "predict_demo"
lines = ["# Predictor demo on held-out test images (3 per available source, seed 42)", ""]
for s in SOURCES:
    sub = test[test["source"] == s].sort_values("path")
    if len(sub) == 0:
        lines.append(f"- {s}: UNAVAILABLE (no images in the dataset/test split)")
        print(f"{s}: unavailable")
        continue
    pick = sub.sample(3, random_state=C.SEED)
    d = out / s
    d.mkdir(parents=True, exist_ok=True)
    for r in pick.itertuples():
        shutil.copy2(C.ROOT / r.path, d / Path(r.path).name.replace(" ", "_"))
    truth = {Path(r.path).name.replace(" ", "_"): (r.crop, r.health, r.condition) for r in pick.itertuples()}
    cmd = [sys.executable, "-I", str(Path(__file__).resolve().parent / "predict.py"), "--folder", str(d)]
    res = subprocess.run(cmd, cwd=C.ROOT, capture_output=True, text=True)
    print(f"===== {s} (exit {res.returncode})\n{res.stdout}{res.stderr}")
    lines += [f"## {s}", "", "```", res.stdout.strip(), "```", "", "Ground truth: " +
              "; ".join(f"{k}: {v[0]} / {v[1]} / {v[2]}" for k, v in truth.items()), ""]
    log(f"demo_predict {s}: exit={res.returncode}, 3 test images")
(out / "DEMO_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
