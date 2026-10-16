# Fruit Leaf Health AI

Given a leaf photograph (and optionally four environmental values) the system predicts, for eight crops
(Tomato, Apple, Grape, Strawberry, Peach, Cherry, Orange, Pepper_bell): the **crop**, **health** (Healthy/Unhealthy),
**cause** (Biotic/Abiotic), the specific **condition** (e.g. `Tomato___Early_blight`), a **severity** (Mild/Moderate/Severe)
and a **Grad-CAM heatmap**. It runs on CPU or CUDA GPU.

> **Read `outputs/LIMITATIONS.md` before using any number or output.** Key caveats: the environmental inputs are
> *simulated* and the models do not use them; no abiotic (stress) data were available, so the cause head cannot
> distinguish biotic from abiotic; severity labels are *heuristic*; accuracy on field photos is far below lab accuracy.

## Results at a glance (held-out test split, 3,464 images, seed 42, single run)
Full tables with 95% group-bootstrap CIs and sample counts: `outputs/tables/` (CSV + Markdown).

| Model | Crop acc. | Health acc. | Condition acc. | Severity macro-F1 (heuristic labels) |
|---|---|---|---|---|
| Image-only | 0.990 | 0.995 | 0.967 | 0.825 |
| Environment-only (simulated inputs) | 0.456 | 0.783 | 0.088 | 0.236 |
| Fusion (image + simulated env) | 0.990 | 0.997 | 0.969 | 0.831 |
| Fusion, env rows shuffled in training (control) | 0.990 | 0.997 | 0.969 | 0.830 |
| Majority-class baseline | 0.408 | 0.783 | 0.022 | 0.208 |

- The environment adds nothing: Fusion, the shuffled-env control and Image-only are statistically indistinguishable, and
  permuting any env feature changes accuracy by <= 0.07 percentage points.
- Seed repeat (image-only, 4 training seeds, same split): condition accuracy 0.969 +/- 0.002, field condition accuracy
  0.632 +/- 0.014, severity macro-F1 0.828 +/- 0.006 (`outputs/tables/seed_repeat_image.md`). Only the image-only model
  was repeated.
- Lab vs field (Fusion): condition accuracy 0.990 on PlantVillage lab photos (n=3,258) vs 0.631 on PlantDoc field photos
  (n=206); on the like-for-like biotic subset 0.978 (n=1,303) vs 0.541 (n=122).
- Cause accuracy is 1.000 for every model but is meaningless (only Biotic labels exist).
- Background probe: a classifier that sees only a blurred 16x16 image predicts the 27-way condition with 63% accuracy
  (chance 2.2%) - lab accuracy is partly a background/capture effect.

## Environment setup (Windows, Python 3.11, no admin rights, nothing installed globally)
```powershell
python -m venv .venv                                   # already present in this folder
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install pandas scikit-learn opencv-python imagehash matplotlib tqdm huggingface_hub
```
Exact tested versions: `requirements.txt`. Without an NVIDIA GPU, install the CPU build of torch/torchvision from
`https://download.pytorch.org/whl/cpu`; training/inference fall back to CPU automatically (training would be much slower).
All scripts are run with the project interpreter, e.g. `.\.venv\Scripts\python.exe -I src\<script>.py`.

## Data (obtained with explicit approval; downloads are not automated)
| Dataset | Source used | License (as stated in the source's metadata) | What was used |
|---|---|---|---|
| PlantVillage | Hugging Face `mohanty/PlantVillage` (`data.zip`, 2,083.5 MB; GitHub `spMohanty/PlantVillage-Dataset` has no license field) | CC-BY-SA-3.0 (dataset card) | color images of the 8 crops, 27 classes, up to 1,000 per condition; original-leaf ids from `leaf_grouping/leaf-map.json` |
| PlantDoc | GitHub `pratikkayal/PlantDoc-Dataset` (cropped classification version, ~0.95 GB) | CC-BY-4.0 (repo LICENSE) | 18 mappable classes, 1,478 field images; `Bell_pepper leaf spot` and non-target crops excluded |
| OLID I (abiotic tomato) | Kaggle only | - | **not obtained** (no credentials used) |

Credits: Hughes & Salathe (2015) and Mohanty et al. (2016) for PlantVillage; Singh et al., *PlantDoc: A Dataset for Visual
Plant Disease Detection* (CoDS-COMAD 2020). Licenses were taken from the repositories' metadata, not independently
reviewed; check ShareAlike/attribution terms before redistributing data or data-derived weights.
PlantDoc could not be checked out with `git` on Windows (filenames contain `?`); `src/extract_plantdoc.py` reads the files
from the git objects (sha1-verified) into `data/raw/plantdoc/` with sanitized names and writes `_manifest.csv`.
Label mapping tables are in `src/prepare_data.py`; the dataset inspection notes are `outputs/dataset_inspection.md`.

## Reproduce
```powershell
.\.venv\Scripts\python.exe -I src\train.py --all          # skips prep stages whose outputs are consistent, trains the
                                                          # 4 variants (resumes from last.pt), then evaluates/explains
.\.venv\Scripts\python.exe -I src\train.py --variant fusion      # one variant (resumable; skipped if DONE.json exists)
```
`--all` expects `data/raw/` to be populated as described above. Stages it runs (each is also a script):
`prepare_data.py` -> `split_data.py` -> `severity.py` -> `env_features.py` -> `build_cache.py` -> `train.py` (x4) ->
`evaluate.py` -> `sanity.py` -> `explain.py`. Other scripts: `smoke_test.py` (forward/backward, masked and ordinal loss,
checkpoint reload tests), `verify_cache.py`, `demo_predict.py`, `infer.py` (strict checkpoint loading, Grad-CAM).
Everything is logged with timestamps to `logs/progress.log`.

## Predict
New to this? See **`HOW_TO_RUN.md`** (step-by-step, including drag-and-drop with `run_predict.bat`).
```powershell
.\.venv\Scripts\python.exe src\predict.py --image "path\to\leaf.jpg"
.\.venv\Scripts\python.exe src\predict.py --image "path\to\leaf.jpg" --temp 28 --rainfall 10 --humidity 40 --soil_moisture 20
.\.venv\Scripts\python.exe src\predict.py --folder "path\to\folder"          # also: --variant image|fusion, --no-cam
```
Example (real output, held-out PlantVillage test image):
```
Environment values not provided: using training averages
Image: ...JR_FrgE.S_8685.JPG
  Crop:       Apple (confidence 0.87)
  Health:     Unhealthy (confidence 0.94)
  Cause:      Biotic [only class seen in training; no abiotic data, so abiotic stress cannot be identified - not a meaningful prediction]
  Condition:  Apple___Black_rot (confidence 0.84)
  Severity:   Mild (confidence 0.71) [heuristic damage estimate]
  Grad-CAM:   outputs\predictions\<timestamp>\cam_...png
```
- Confidence = softmax probability of the predicted class (severity: probability of the predicted ordinal class); it is
  **not calibrated**. If a relevant confidence is below 0.5 the tool prints
  "Low confidence - this may not be one of the 8 supported crops or may be a poor photo"; this is a heuristic, not a
  guarantee, and wrong predictions can be confident.
- Missing environment values are replaced by the training means (the tool says so). The simulated-env models ignore them anyway.
- Results (CSV + Grad-CAM overlays) go to `outputs/predictions/<timestamp>/`. Invalid files, tiny images and non-finite or
  out-of-range environment values are rejected with an error message; filenames are never executed.
- The checkpoint is loaded strictly (architecture, label maps, image size from the checkpoint; normalisation statistics
  from `models/env_stats.json`). A missing or mismatching checkpoint stops with an error. Default variant: `fusion`;
  `image` is equivalent in accuracy.
- Demo on 3 seeded test images per source: `outputs/predict_demo/DEMO_RESULTS.md` (PlantVillage 3/3 correct, PlantDoc 1/3).

## Method summary (what the code does)
- **Labels** (`prepare_data.py`): `condition = Crop___Condition`; healthy leaves are the class `Crop___healthy` in the condition
  head. Disease/pest -> cause Biotic; healthy -> cause None. Images that cannot be opened or are under 32x32 px are dropped
  (none were). Index columns: `path,crop,condition,health,cause,source,acquisition,group_id`.
- **Splits** (`split_data.py`): 70/15/15, seed 42, stratified by condition, whole groups only. Groups merge original-leaf ids
  (PlantVillage `leaf-map.json`, fallback original-image name), exact duplicates (MD5) and near-duplicates (pHash Hamming <= 6).
  Not plant-level independence. Counts: train 16,635 / val 3,460 / test 3,464.
- **Simulated environment** (`env_features.py`): crop baseline + large Gaussian noise seeded from md5 of the path; normalisation
  statistics fitted on the train split only (`models/env_stats.json`).
- **Severity** (`severity.py`): GrabCut leaf mask (HSV fallback), low-saturation pixels removed, eroded; damage = leaf pixels
  that are not healthy green; Mild < 0.10 <= Moderate <= 0.30 < Severe. Heuristic only; masks flagged invalid are excluded.
- **Model** (`model.py`): EfficientNet-B0 (ImageNet weights, 1,280-d) + env MLP 4->32->32 (ReLU); fusion = 1,312-d concat.
  Heads: crop (8), health (2), cause (2), condition (27), severity (2 cumulative-ordinal logits P(>Mild), P(>Moderate)).
  Variants: image, env, fusion, fusion_shuffled_env (training env rows permuted across samples, seed 42; evaluated on real env).
  Losses are masked (cause: unhealthy only; severity: unhealthy with a valid mask); label smoothing 0.1 on classification heads.
- **Training** (`train.py`): AdamW (heads/env 3e-4, backbone 1e-4, weight decay 1e-4), cosine decay, grad clip 1.0, mixed
  precision, seed 42, batch 16, 3 loader workers, 224 px (from a 256 px cache), <= 8 epochs (all ran 8; early stopping with
  patience 3 never triggered), checkpoint every epoch (`last.pt`) plus best by validation loss (`best.pt`), CSV history and
  `outputs/figures/curves_<variant>.png`. Augmentation (train only): random resized crop, flips, rotation <= 25 deg, colour
  jitter, light blur. Class weights were not needed (max imbalance 2.99:1 < 3:1).
- **Evaluation** (`evaluate.py`): test split only, checkpoints chosen on validation loss. 95% CIs from 1,000 bootstrap resamples
  of whole groups (seed 42); macro-F1 averages only classes present in each resample; CIs are omitted below 30 samples or
  10 groups. Sanity checks: `sanity.py` (background probe, cross-split duplicates); Grad-CAM and permutation importance:
  `explain.py` (review notes in `outputs/explain/REVIEW.md`).

## Folder structure
```
src/        all code (see above)               data/raw/        downloaded datasets (git-ignored)
data/processed/  index.csv, splits.csv, severity.csv, env.csv, cache256/ (256 px JPEG cache), *_pilot.csv (pilot runs)
models/     <variant>/{best.pt,last.pt,history.csv,config.json,DONE.json}, label_maps.json, env_stats.json, torch_cache/
outputs/    tables/ (Tables 3-6, preds_*.csv, sanity_*, permutation importance), figures/ (curves, confusion matrices),
            explain/ (Grad-CAM overlays + REVIEW.md), severity_debug/ (24 images), predictions/, predict_demo/,
            dataset_inspection.md, LIMITATIONS.md
logs/       progress.log (every stage), per-run text logs
```
`_superseded_*` and `*_flawed` folders are earlier attempts kept for transparency, not used.

## Hardware and runtime (as measured)
Windows 11, Intel i5-13420H (8 cores/12 threads), 15.6 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB, driver 592.82), Python 3.11.0,
torch 2.12.1+cu126. Peak training VRAM ~0.8 GiB at batch 16.
Approximate wall-clock: torch install 20 min (2.6 GB); PlantVillage download ~13 min; index build 9 min; split 6 min;
severity labels 50 min (6 processes); training 8 epochs: env 2 min, image 12 min, fusion 12.5 min, fusion_shuffled_env 35 min
(slowed by other programs); evaluation 5 min; sanity probe 4 min; Grad-CAM + permutation 4 min.
Disk: `.venv` 4.6 GB, raw data ~3.5 GB (+2 GB original zip), processed data 0.7 GB, models 0.2 GB.

## Appropriate use
A research/education demonstrator for 27 biotic/healthy classes of eight crops. Not for agronomic or treatment decisions, not
for diagnosing nutrient or water stress, and not validated for photos unlike PlantVillage/PlantDoc. Do not commit or upload the
datasets, model weights or credentials.
#   p l a n t - m o d e l  
 