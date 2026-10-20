# Repair plan — Fruit Leaf Health AI (Phase A: verify and plan)

Created 2026-10-20. Status: **PLAN ONLY. No source file, checkpoint, dataset or output was modified in Phase A.**
The only file written is this one (`logs/repair_plan.md`, which is new).

## 1. Verified project

| Item | Finding | How verified |
|---|---|---|
| Project root | `C:\Plant model\fruit_health` (mounted in the inspection shell as `~/mnt/Plant model/fruit_health`) | `pwd` and directory listing |
| Same project as the analysis report? | **Yes.** The report analysed this exact folder on the same day. `git diff HEAD` shows no change to `src/`, README, HOW_TO_RUN, requirements, `run_predict.bat`, `.vscode` or `label_maps.json` | `git diff --stat HEAD` (empty for those paths) |
| Original path | Created at `C:\Users\dishi\OneDrive\Desktop\Plant model\fruit_health` (per `.venv/pyvenv.cfg` and `progress.log`), later moved to `C:\Plant model`. All code uses paths relative to `src/`, so the move breaks nothing in code. `.venv` launchers may still embed the old path (NOT TESTED) | `pyvenv.cfg`, `config.py:4` |
| Git | Branch `main`, 2 commits (`e364639`, `9dd747f`), remote `github.com/D1shika/plant-model`. Working tree: `logs/progress.log` modified; two untracked prediction folders (`outputs/predictions/20261016_214005`, `_214324`). Pre-existing changes will be left untouched | `git status`, `git log`, `git remote -v` |
| Tracked in git | All 22 `src/*.py`, docs, configs, histories, `label_maps.json`, `env_stats.json`, logs (439 files) | `git ls-files` |
| NOT in git (no recovery copy) | All `*.pt` checkpoints, `data/raw`, `data/processed` (index, splits, severity, env, cache), most `outputs/` images | `.gitignore` |

## 2. Python environment

* Project venv: `.venv` = Windows CPython 3.11.0 with torch 2.12.1+cu126, torchvision 0.27.1+cu126, numpy 2.4.6, pandas 3.0.6, opencv-python 5.0.0.93, pillow 12.3.0, scikit-learn 1.9.1, ImageHash 4.3.2 (from `site-packages` dist-info folders). **pytest is not installed.**
* **Limitation of this session:** my shell on your device is a Linux sandbox that can read and write the folder, but it cannot execute the Windows `.venv` (it has Python 3.10 with numpy/pandas only, no torch). So in Phase A I could **not** run `smoke_test.py`, load checkpoints into PyTorch, or run inference. Every "ran" claim below is a file-level or pandas-level check only.
* For Phase B tests to run in the real venv, either (a) I start a Remote Control session on your computer in this folder (it runs Claude Code with your `.venv`; you approve it once), or (b) you run the test commands yourself and paste the output. Recommendation: (a).

## 3. Checkpoints and artefacts

All 15 `.pt` files open as valid PyTorch zip archives with no CRC errors (`zipfile.testzip`). Loading into the model with `strict=True` is **NOT RUN** (needs the Windows venv).

| Run | best.pt | last.pt | best epoch (eval_manifest / history) |
|---|---|---|---|
| image | 16.5 MB OK | 49.2 MB OK | 7 |
| env | 16.5 kB OK | 62.7 kB OK | 6 |
| fusion | 16.5 MB OK | 49.2 MB OK | 6 |
| fusion_shuffled_env | 16.5 MB OK | 49.2 MB OK | 6 |
| image_seed1/2/3 | OK | OK | 7 / 7 / 6 |
| _smoke_test/smoke_ckpt.pt | untrained test file | — | — |

ImageNet backbone cached at `models/torch_cache/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth`, so no download is needed for inference.

## 4. Dataset counts and class mappings (recomputed from `data/processed/splits.csv`)

* 23,559 images · 27 conditions · 8 crops · split train 16,635 / val 3,460 / test 3,464 (PlantDoc 1,061 / 211 / 206).
* Cause: **every** unhealthy image in train (12,705), val (2,714) and test (2,713) is `Biotic`; healthy images have no cause. There are zero Abiotic labels anywhere, and `label_maps.json` records `cause_trained_classes: ["Biotic"]`. **Confirmed single-class.**
* Health is defined in `prepare_data.py:126` by the folder suffix `___healthy`. There is no separate per-class metadata file, so Phase B will add an explicit healthy-class list (derived once from `prepare_data.PV_MAP`/`PD_MAP`, then written as data) rather than substring matching at inference time.
* No saved **validation** predictions exist (`outputs/tables/preds_*.csv` are test only). Validation-only calibration therefore needs one inference pass over the 3,460 validation images (about 1–3 min on the GPU; needs the device venv).

## 5. Existing tests

* `src/smoke_test.py`: 18 checks. The last logged run passed 18/18 (`progress.log`, 2026-10-16 21:39). It was **not re-run** in Phase A (cannot execute the venv here).
* `smoke_test.py:47` calls `save_label_maps(maps)`, which **rewrites the production `models/label_maps.json`** on every run. Confirmed.
* There is no pytest suite.

## 6. Baseline metrics (copied from saved artefacts; spot-checks recomputed from `preds_fusion.csv`)

All metrics are on the **test** split, best checkpoint by validation loss, with 95% group-bootstrap CIs from `evaluate.py`.

| Metric (Fusion unless stated) | Value | n | Source | Recomputed? |
|---|---|---|---|---|
| Condition acc, all | 0.969 [0.961, 0.975] | 3,464 | `outputs/tables/table3_modality_ablation.md` | point value yes |
| Condition acc, image-only | 0.967 [0.960, 0.973] | 3,464 | same | copied |
| Condition acc, lab / field | 0.990 / 0.631 [0.565, 0.692] | 3,258 / 206 | `table4_lab_vs_field.md` | yes |
| Like-for-like biotic condition, field | 0.541 [0.455, 0.631] | 122 | `table4` | copied |
| Health acc, lab / field | 0.999 / 0.951 | 3,258 / 206 | `table4` | yes |
| Crop acc, field | 0.835 | 206 | `per_group_metrics.md` | copied |
| Cause acc | 1.000 (**trivial, single class**) | 2,713 | `table3` | confirmed trivial |
| Severity macro-F1 (heuristic labels, true-unhealthy) | 0.831 [0.811, 0.848] | 2,664 | `table6_severity.md` | yes (0.834 lab / 0.724 field) |
| Seed repeat, image-only condition | 0.969 ± 0.002 (4 seeds) | 3,464 | `seed_repeat_image.md` | copied |
| Env permutation importance | ≤ 0.07 pp change | 3,464 | `env_permutation_importance.md` | copied |
| Head inconsistencies (crop vs condition, health vs condition) | 19 unique (18 field) | 3,464 | `preds_fusion.csv` | recomputed |
| Ordinal violations P(>Mod) > P(>Mild) | 115 | 3,464 | `preds_fusion.csv` | recomputed |
| Wrong field predictions not flagged by the 0.5 threshold | 41 of 76 | 206 | `preds_fusion.csv` | recomputed |

## 7. Report claims: confirmed / not confirmed

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Independent heads can contradict each other; `predict.py` does not reconcile | **Confirmed** | `model.py:47-51`, `predict.py:176-199`; 19 cases in `preds_fusion.csv` |
| 2 | Cause trained on Biotic only | **Confirmed** | splits.csv crosstab; `label_maps.json` |
| 3 | Predictor defaults to fusion with simulated env | **Confirmed** | `predict.py:115`, `env_features.py` docstring |
| 4 | Env has no measurable effect | **Confirmed on saved results, single seed for fusion** | table3, permutation importance. Fusion vs shuffled were run with one seed only, so "no effect" is stated as "no detectable effect at this sample size" |
| 5 | Severity mask drops low-saturation pixels | **Confirmed in code** | `severity.py:78`. Effect on mildew (median damage 0.012) recomputed from `severity.csv` |
| 6 | Ordinal violations | **Confirmed** | 115 rows |
| 7 | pHash chaining makes mixed mega-groups | **Confirmed** | largest group 648 images, 98 mixed-condition groups |
| 8 | CPU install conflicts with `+cu126` pins | **Confirmed in files**, not executed | `requirements.txt:43-44` vs README CPU steps |
| 9 | No EXIF transpose; square resize | **Confirmed** | `dataset.py:28-31`; no `exif_transpose` in `src/` |
| 10 | Smoke test rewrites production label maps | **Confirmed** | `smoke_test.py:47` |
| 11 | Pasted URL in `run_predict.bat`; only first dropped file used | **Confirmed** | line 9; `%~1` only |
| 12 | Doc drift (env MLP layers, table names, section names) | **Confirmed** | README vs `model.py:44`, `outputs/tables` filenames |
| 13 | Calibration (lab ECE ≈ 0.095) | **Confirmed on test preds**, but must not be used for fitting; validation preds do not exist yet | — |
| 14 | Checkpoints load strictly and run | **NOT CONFIRMED in Phase A** (zip-valid only; needs the device venv) | — |
| 15 | Smoke test currently 18/18 | **NOT RE-RUN** (last logged pass 2026-10-16) | — |

## 8. Prioritised work plan

Each phase starts only after your approval. Time estimates are for the RTX 3050 laptop.

### Phase B — Safe inference fixes (no retraining, no downloads)
Files: new `src/reconcile.py`, new `src/class_meta.json`, edits to `src/predict.py`, `src/infer.py` (small), new `tests/` folder.
1. **Consistent predictions:** crop and health derived from condition probabilities (sum per crop; healthy set from explicit metadata). Raw head outputs are kept in the CSV as `raw_*` columns, with a `head_disagreement` warning column.
2. **Cause:** report "unavailable (single-class model)" instead of "Biotic".
3. **Default variant → `image`.** Fusion stays selectable as `--variant fusion` with an "experimental, simulated inputs" notice. Env flags are only accepted with fusion.
4. **Uncertainty:** documented confidence semantics, an "Uncertain prediction — please verify…" outcome, and thresholds marked **provisional** until validation calibration exists (one approved validation inference pass, ~3 min GPU, then temperature scaling fitted on validation only).
5. **Ordinal projection:** report P(>Mod) ← min(P(>Mod), P(>Mild)), keep raw values, count violations before/after. Evaluate severity with predicted-health gating.
6. **Tests:** synthetic-logit unit tests (pure numpy, runnable without torch), plus checkpoint-load, single and batch inference, and Grad-CAM tests run in the venv. pytest is not installed, so tests will be plain `python tests/run_tests.py` scripts (no install needed).
Acceptance: 0 user-facing inconsistencies on saved test predictions (stated as display consistency, **not** an accuracy gain); all tests pass in the venv; original checkpoints byte-identical (sha256 before/after).

### Phase C — Severity heuristic prototype
New `src/severity_v2.py` (the old file stays untouched). Pilot on ~60 seeded images (mildew, chlorosis, necrosis, healthy, field), with side-by-side overlays and a comparison report. Labels are written to `severity_v2_pilot.csv` only. Stop before full regeneration (~20–30 min CPU) or retraining.

### Phase D — Data and evaluation audit
Candidate split `splits_v2_candidate.csv` (new file) with non-chaining duplicate grouping, a field-weighted validation proposal, and a public field-dataset list with licences (no downloads). The existing test split and metrics are preserved.

### Phase E — Packaging and documentation
`requirements-gpu.txt` / `requirements-cpu.txt` (no install), batch runner fix (remove URL, loop over all dropped files), isolated smoke test, README / HOW_TO_RUN / LIMITATIONS corrections, EXIF handling added behind a flag with compatibility tests.

### Phase F — Retraining (separate approval per run)
Only after B–E. New run names; each image-model run is about 12–15 min GPU, ~70 MB disk, < 1 GB VRAM.

## 9. Backups and rollback

* Source files: all tracked in git with a clean tree for `src/`, which is a verified recovery path (`git checkout -- <file>`). In addition, before Phase B edits, every changed file is copied to `backups/<timestamp>/` inside the project.
* Checkpoints, data and outputs: never modified by Phases B–E. sha256 of every `best.pt`/`last.pt` is recorded before Phase B and re-checked after.
* New outputs go to new filenames; nothing existing is overwritten except `predict.py` and docs (backed up first).
* No git commit or push unless you ask.
