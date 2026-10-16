"""Reproducible pipeline: resume completed preparation stages, then train, evaluate and explain in sequence.

Invoked as:  python src/train.py --all [--skip-eval] [--smoke]
Each stage is skipped when its output already exists and is consistent with the stage before it; nothing is deleted or
overwritten unless a stage's output is missing/inconsistent (then that stage is recomputed). Raw data are NOT downloaded
here (downloads need explicit approval): data/raw/ must already contain PlantVillage and PlantDoc as prepared in Phase C.
Every stage runs as a separate process with the project's interpreter and is logged to logs/progress.log.
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from log_util import log  # noqa: E402

SRC = Path(__file__).resolve().parent
P = C.DATA_PROCESSED
VARIANTS = ("env", "image", "fusion", "fusion_shuffled_env")


def rows(path):
    try:
        return len(pd.read_csv(path))
    except Exception:  # noqa: BLE001
        return -1


def run(script, *args):
    cmd = [sys.executable, "-I", str(SRC / script), *map(str, args)]
    t0 = time.time()
    log(f"pipeline: RUN {script} {' '.join(map(str, args))}")
    rc = subprocess.run(cmd, cwd=C.ROOT).returncode
    log(f"pipeline: {script} exit={rc} ({time.time() - t0:.0f}s)")
    if rc != 0:
        sys.exit(f"stage {script} failed with exit code {rc}; see logs/progress.log (outputs preserved)")


def newest(paths):
    ts = [p.stat().st_mtime for p in paths if p.exists()]
    return max(ts) if ts else 0.0


def run_pipeline(skip_eval=False, smoke=False):
    pv, pd_, lm = C.DATA_RAW / "plantvillage", C.DATA_RAW / "plantdoc" / "_manifest.csv", \
        C.DATA_RAW / "plantvillage_dl" / "leaf_grouping" / "leaf-map.json"
    if not (pv.is_dir() and pd_.exists() and lm.exists()):
        sys.exit("Raw data missing (data/raw/plantvillage, data/raw/plantdoc/_manifest.csv, "
                 "data/raw/plantvillage_dl/leaf_grouping/leaf-map.json). See README 'Data' for how they were obtained.")
    n_idx = rows(P / "index.csv")
    if n_idx > 0:
        print(f"[skip] prepare_data: index.csv has {n_idx} rows")
    else:
        run("prepare_data.py")
        n_idx = rows(P / "index.csv")
    if rows(P / "splits.csv") == n_idx > 0:
        print("[skip] split_data: splits.csv consistent")
    else:
        run("split_data.py", "--index", "data/processed/index.csv", "--out", "data/processed/splits.csv")
    n_unh = int((pd.read_csv(P / "index.csv")["health"] == "Unhealthy").sum())
    if rows(P / "severity.csv") == n_unh > 0:
        print("[skip] severity: severity.csv consistent")
    else:
        run("severity.py", "--index", "data/processed/index.csv", "--out", "data/processed/severity.csv", "--force")
    if rows(P / "env.csv") == n_idx and (C.MODELS / "env_stats.json").exists():
        print("[skip] env_features: env.csv + env_stats.json present")
    else:
        run("env_features.py", "--splits", "data/processed/splits.csv", "--out", "data/processed/env.csv")
    run("build_cache.py")  # resumable; no-op when complete
    if smoke:
        run("smoke_test.py")
    for v in VARIANTS:
        run("train.py", "--variant", v)  # resumes from last.pt; exits immediately if DONE.json exists
    if skip_eval:
        return
    done = [C.MODELS / v / "DONE.json" for v in VARIANTS]
    t3 = C.OUTPUTS / "tables" / "table3_modality_ablation.csv"
    if t3.exists() and t3.stat().st_mtime > newest(done):
        print("[skip] evaluate: tables newer than all completed trainings")
    else:
        run("evaluate.py")
    if not (C.OUTPUTS / "tables" / "sanity_probe.md").exists():
        run("sanity.py")
    else:
        print("[skip] sanity: sanity_probe.md exists")
    run("explain.py", "--variant", "fusion")
    log("pipeline: complete")
