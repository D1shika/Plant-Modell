"""SIMULATED environmental features (temperature C, rainfall mm, humidity %, soil moisture %).

The photographs have no real sensor readings. Values are synthetic and clearly NOT real measurements.
Method: per-sample numpy Generator seeded with int(md5(<relative image path>)[:8], 16) (never Python hash()).
Values = crop-specific baseline mean + large Gaussian noise (std comparable to between-crop spread), clipped to
physical ranges. Condition/health/cause/source are NOT used, so the values carry only (weak) crop information.
Normalisation statistics are fitted on the TRAIN split only and saved to models/env_stats.json.

Usage: python src/env_features.py --splits data/processed/splits.csv --out data/processed/env.csv
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS, ROOT  # noqa: E402
from log_util import log  # noqa: E402

FEATURES = ["temperature", "rainfall", "humidity", "soil_moisture"]
# crop baselines (mean temp C, rainfall mm, humidity %, soil moisture %) and shared noise std
BASE = {"Tomato": (26, 60, 65, 45), "Apple": (18, 80, 62, 50), "Grape": (24, 50, 55, 38),
        "Strawberry": (20, 70, 68, 52), "Peach": (22, 65, 60, 44), "Cherry": (17, 75, 63, 48),
        "Orange": (27, 90, 70, 50), "Pepper_bell": (25, 55, 62, 42)}
STD = (6.0, 40.0, 14.0, 12.0)
LOW = (0, 0, 5, 2)
HIGH = (50, 400, 100, 100)
METHOD = ("simulated: crop baseline + Gaussian noise, seed=md5(relative path)[:8]; "
          "independent of condition/health/cause; NOT sensor data")


def simulate(path: str, crop: str) -> list:
    seed = int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    v = np.array(BASE[crop]) + rng.normal(0, 1, 4) * np.array(STD)
    return list(np.clip(v, LOW, HIGH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(ROOT / a.splits)
    env = pd.DataFrame([simulate(p, c) for p, c in zip(df["path"], df["crop"])], columns=FEATURES)
    env.insert(0, "path", df["path"])
    env.to_csv(ROOT / a.out, index=False)
    tr = env[df["split"] == "train"][FEATURES]
    stats = {"features": FEATURES, "mean": tr.mean().tolist(), "std": tr.std().replace(0, 1).tolist(),
             "fitted_on": "train split only", "n_train": int(len(tr)), "method": METHOD}
    MODELS.mkdir(exist_ok=True)
    json.dump(stats, open(MODELS / "env_stats.json", "w"), indent=2)
    print("env stats (train only):", {k: [round(x, 2) for x in stats[k]] for k in ("mean", "std")})
    log(f"env_features: {len(env)} rows simulated; env_stats.json fitted on train n={len(tr)}; {METHOD}")


if __name__ == "__main__":
    main()
