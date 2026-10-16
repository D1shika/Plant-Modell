# Fruit Leaf Health AI

> **Multi-Task Deep Learning Framework for Crop Identification, Disease Diagnosis, Heuristic Severity Grading, and Visual Interpretability Across Eight Fruit and Vegetable Crops.**

---

## Project Overview

**Fruit Leaf Health AI** is an end-to-end multi-task deep learning system designed to diagnose plant conditions from leaf photographs. Given an image of a crop leaf (and optionally four environmental sensor values: temperature, rainfall, humidity, and soil moisture), the system concurrently predicts:

1. **Crop Identity** (8 supported agricultural crops)
2. **Health Status** (`Healthy` vs `Unhealthy`)
3. **Causal Agent** (`Biotic` vs `Abiotic` — *see limitations regarding training data availability*)
4. **Specific Condition** (27 conditions spanning fungal, bacterial, viral diseases, pests, and healthy states)
5. **Damage Severity** (`Mild`, `Moderate`, `Severe` via cumulative ordinal regression)
6. **Visual Explanation** (Class Activation Mapping via Grad-CAM on the feature backbone)

The system supports execution on both CUDA-enabled NVIDIA GPUs and CPU-only environments. It incorporates a strictly validated inference pipeline with automated out-of-distribution / low-confidence warnings, tabular logging, and visual diagnostic overlays.

> [!CAUTION]
> **Research and Educational Demonstrator Only:** This project is not calibrated for commercial agronomic decision-making, field treatment recommendations, or stress detection beyond the 27 evaluated biotic conditions. Please review [Limitations and Known Issues](#limitations-and-known-issues) and [`outputs/LIMITATIONS.md`](outputs/LIMITATIONS.md) before interpreting any output or metric.

---

## Key Features

- **Multi-Task Neural Architecture:** Employs a shared pre-trained EfficientNet-B0 visual backbone coupled with an environmental feature Multi-Layer Perceptron (MLP) and five specialized task heads.
- **Masked & Cumulative Ordinal Losses:** Uses masked cross-entropy for conditional heads (cause and severity only evaluate on unhealthy leaves with valid masks) and cumulative ordinal binary cross-entropy for monotonic damage level progression ($P(\text{Severity} > \text{Mild})$ and $P(\text{Severity} > \text{Moderate})$).
- **Comprehensive Modality Ablation:** Evaluates four distinct configurations:
  - **Image-only:** Uses leaf imagery alone.
  - **Environment-only:** Uses four simulated weather/soil indicators.
  - **Multimodal Fusion:** Concatenates visual embeddings with environmental representations.
  - **Fusion Shuffled Environment (Statistical Control):** Randomly permutes environmental features across samples during training to measure causal information leakage.
- **Group-Aware, Leak-Free Evaluation:** Enforces strict group-level partitioning using original leaf identifiers (`leaf-map.json`), exact MD5 hashes, and perceptual hashing (pHash Hamming distance $\le 6$) to prevent duplicate or same-leaf leakage between train, validation, and test splits.
- **Statistical Rigor:** All reported performance intervals reflect 95% group-bootstrap confidence intervals across 1,000 resamples, alongside four-seed repeatability verification.
- **Explainable AI (XAI):** Built-in Grad-CAM implementation generates high-resolution saliency overlays illustrating which leaf regions drove the condition prediction.
- **Flexible Operational Interfaces:** Includes a Windows drag-and-drop batch runner (`run_predict.bat`), an interactive command-line interface (`src/predict.py`), and preconfigured VS Code debugging targets (`.vscode/launch.json`).

---

## Supported Crops

The framework supports 8 agricultural crops covering 27 distinct condition categories:

| Crop | Number of Classes | Supported Diagnostic Conditions |
|---|:---:|---|
| **Apple** | 4 | `Apple___Apple_scab`<br>`Apple___Black_rot`<br>`Apple___Cedar_apple_rust`<br>`Apple___healthy` |
| **Cherry** | 2 | `Cherry___Powdery_mildew`<br>`Cherry___healthy` |
| **Grape** | 4 | `Grape___Black_rot`<br>`Grape___Esca`<br>`Grape___Leaf_blight`<br>`Grape___healthy` |
| **Orange** | 1 | `Orange___Haunglongbing` *(Citrus greening; lab only)* |
| **Peach** | 2 | `Peach___Bacterial_spot`<br>`Peach___healthy` |
| **Pepper (Bell)** | 2 | `Pepper_bell___Bacterial_spot`<br>`Pepper_bell___healthy` |
| **Strawberry** | 2 | `Strawberry___Leaf_scorch`<br>`Strawberry___healthy` |
| **Tomato** | 10 | `Tomato___Bacterial_spot`<br>`Tomato___Early_blight`<br>`Tomato___Late_blight`<br>`Tomato___Leaf_Mold`<br>`Tomato___Mosaic_virus`<br>`Tomato___Septoria_leaf_spot`<br>`Tomato___Spider_mites`<br>`Tomato___Target_Spot`<br>`Tomato___Yellow_Leaf_Curl_Virus`<br>`Tomato___healthy` |

---

## System Architecture and Workflow

```
                             [ Input Leaf Photo ]
                                      │
                         Image Preprocessing & Transform
                               (224 x 224 RGB)
                                      │
                         ┌────────────┴────────────┐
                         ▼                         │
                 EfficientNet-B0                   │
               Feature Extractor                   │
                 (1,280-d Vector)                  │
                         │                         ▼
                         │               Grad-CAM Engine
                         │           (Last Conv Activation)
                         │                         │
                         ├─────────────────────────┼──────────────┐
                         ▼                         ▼              ▼
     [ Optional Weather / Soil Readings ]      Grad-CAM       Grad-CAM
      (Temp, Rain, Humidity, Moisture)          Heatmap        Overlay
                         │
                    Env MLP
               (4 -> 32 -> 32)
                 (32-d Vector)
                         │
                         ▼
        ┌───────────────────────────────────┐
        │   Multimodal Fusion Concat Layer  │
        │          (1,312-d Vector)         │
        └─────────────────┬─────────────────┘
                          │
     ┌───────────┬────────┼───────────┬───────────┐
     ▼           ▼        ▼           ▼           ▼
 Crop Head   Health Head Cause Head Condition Head Severity Head
  (Linear)    (Linear)   (Linear)    (Linear)    (Ordinal Linear)
   8 crops    2 classes  2 classes   27 classes  2 cumulative logits
```

### End-to-End Pipeline Stages

1. **Data Ingestion (`src/prepare_data.py`):** Parses raw image directories, maps folder naming schemes to standardized target labels, filters corrupted files ($< 32 \times 32$ pixels), and extracts initial grouping metadata.
2. **Stratified Group Split (`src/split_data.py`):** Calculates MD5 and pHash fingerprints, clusters near-duplicate images, and creates stratified 70/15/15 partitions where entire image clusters are held out together.
3. **Severity Label Generation (`src/severity.py`):** Performs automated background removal using GrabCut (with HSV color fallback) and calculates the percentage of damaged leaf tissue.
4. **Synthetic Environment Generation (`src/env_features.py`):** Generates synthetic environmental profiles anchored to crop-specific baselines with MD5-seeded Gaussian variance; fits normalization statistics strictly on the training set (`models/env_stats.json`).
5. **Cache Compilation (`src/build_cache.py`):** Pre-renders $256 \times 256$ JPEG representations for ultra-fast I/O during multi-epoch training.
6. **Multi-Task Training (`src/train.py`):** Trains model variants under cosine learning rate decay and mixed-precision acceleration.
7. **Rigorous Evaluation (`src/evaluate.py`):** Computes unbiased performance metrics on held-out test splits with group-bootstrap confidence intervals.
8. **Sanity Probing (`src/sanity.py`):** Runs background leakage and border strip controls to quantify dataset context bias.
9. **Visual Interpretability (`src/explain.py`):** Generates Grad-CAM visual explanations and assesses feature permutation importance.

---

## Technologies Used

- **Programming Language:** Python 3.11
- **Deep Learning Framework:** PyTorch `2.12.1+cu126`, Torchvision `0.27.1+cu126`
- **Vision Backbone:** EfficientNet-B0 (ImageNet-1K pre-trained weights)
- **Computer Vision & Image Processing:** OpenCV (`opencv-python 5.0.0.93`), Pillow (`pillow 12.3.0`), ImageHash `4.3.2`
- **Data Science & Analytics:** NumPy `2.4.6`, Pandas `3.0.6`, SciPy `1.17.1`, Scikit-Learn `1.9.1`
- **Visualization:** Matplotlib `3.11.2`
- **Pipeline Utilities:** `tqdm 4.70.1`, `huggingface_hub 2.2.0`
- **Platform:** Windows 11 / Linux compatible

---

## Installation and Setup

### Prerequisites
- Python 3.11 (64-bit)
- Windows PowerShell or Bash terminal
- *(Optional)* NVIDIA GPU with CUDA 12.x drivers (falls back automatically to CPU if CUDA is unavailable)

### Step 1: Clone the Repository
```powershell
git clone https://github.com/D1shika/plant-model.git
cd plant-model
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install PyTorch and Dependencies
For CUDA 12.6 acceleration:
```powershell
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For CPU-only machines:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## How to Run the Project

### Method A: Drag-and-Drop (Windows Quick Start)
1. Navigate to the project root directory in Windows File Explorer.
2. Drag any leaf photograph (`.jpg`, `.png`, `.webp`, `.bmp`) or an entire folder of photos directly onto **`run_predict.bat`**.
3. A terminal window opens, executes the prediction pipeline, prints diagnostic findings, and displays the folder path where heatmaps and CSV results are stored.

### Method B: PowerShell / Terminal Command Line

**1. Predict on a single image:**
```powershell
.\.venv\Scripts\python.exe src\predict.py --image "path\to\leaf.jpg"
```

**2. Predict with custom environmental parameters:**
```powershell
.\.venv\Scripts\python.exe src\predict.py --image "path\to\leaf.jpg" --temp 28.0 --rainfall 15.0 --humidity 65.0 --soil_moisture 40.0
```
*(Note: If environmental flags are omitted, training dataset averages are automatically supplied).*

**3. Batch predict on a folder of leaves:**
```powershell
.\.venv\Scripts\python.exe src\predict.py --folder "path\to\leaf_folder"
```

**4. Additional CLI options:**
- `--variant image`: Run using the image-only model (`fusion` is default; both exhibit identical accuracy).
- `--no-cam`: Omit Grad-CAM visual heatmap generation to maximize throughput.

### Method C: Running via VS Code
The workspace includes preconfigured launch tasks in [`.vscode/launch.json`](.vscode/launch.json):
1. Open the project folder in VS Code (`File -> Open Folder...`).
2. Open the **Run & Debug** panel (`Ctrl + Shift + D`).
3. Select an automated workflow from the dropdown:
   - `1 - Predict: SAMPLE photo (no questions)`
   - `2 - Predict: the photo open in the editor`
   - `3 - Predict: type the path of a photo`
   - `7 - Run all 18 smoke tests`
4. Press `Ctrl + F5` to run without debugging.

---

## Model Architecture and Prediction Pipeline

### Network Topology
```
Backbone: EfficientNet-B0 (1,280 features) ──┐
                                            ├── Concat (1,312-d) ──> Task Heads
Env Branch: Linear(4->32) -> ReLU -> (32-d) ┘
```

- **Feature Dimensions:** Visual embedding: 1,280 dimensions; Environmental embedding: 32 dimensions; Fusion representation: 1,312 dimensions.
- **Classification Heads:** Fully connected linear layers mapping the 1,312-d representation to crop logits (8), health logits (2), cause logits (2), condition logits (27), and severity logits (2).
- **Optimization Strategy:**
  - Optimizer: AdamW (head & env learning rate: $3 \times 10^{-4}$, backbone learning rate: $1 \times 10^{-4}$, weight decay: $1 \times 10^{-4}$).
  - Scheduler: Cosine annealing learning rate decay.
  - Regularization: Label smoothing ($\epsilon = 0.1$) on classification heads, gradient norm clipping at $1.0$.
  - Precision: Automatic Mixed Precision (AMP FP16) on CUDA devices.

### Sample Prediction Output
```text
Image: data/sample/apple_frogeye.jpg
  Crop:       Apple (confidence 0.87)
  Health:     Unhealthy (confidence 0.94)
  Cause:      Biotic [only class seen in training; no abiotic data, so abiotic stress cannot be identified - not a meaningful prediction]
  Condition:  Apple___Black_rot (confidence 0.84)
  Severity:   Mild (confidence 0.71) [heuristic damage estimate]
  Grad-CAM:   outputs\predictions\20261016_111842\cam_apple_frogeye_cafa5c.png
```

- **Outputs saved:** Tabular predictions with per-head probabilities are stored in `outputs/predictions/<timestamp>/predictions.csv`, alongside side-by-side Grad-CAM visualizations.
- **Low-confidence warning:** If any active head records a softmax confidence below `0.5`, the system alerts the operator that the subject may represent an unsupported plant species, non-leaf object, or degraded photo.

---

## Dataset and Experimental Setup

### Dataset Sources
1. **PlantVillage Dataset:** 22,081 laboratory photographs across 27 conditions of target crops (source: Hugging Face `mohanty/PlantVillage`, CC-BY-SA-3.0). Standardized laboratory backdrop with leaf-level identification mapping from `leaf_grouping/leaf-map.json`.
2. **PlantDoc Dataset:** 1,478 field photographs across 18 target conditions (source: GitHub `pratikkayal/PlantDoc-Dataset`, CC-BY-4.0; Singh et al., 2020). Extracted and sanitized via low-level git tree unpacking to ensure cross-platform compatibility on Windows (`src/extract_plantdoc.py`).
3. **Abiotic Stress Data (OLID I):** **Not included.** OLID I requires proprietary Kaggle authentication; no unauthenticated mirrors were available. Consequently, no abiotic training images are present.

### Data Partitioning and Leakage Control
- **Split Ratio:** 70% Train (16,635 images), 15% Validation (3,460 images), 15% Test (3,464 images).
- **Stratification:** Stratified across all 27 condition classes while enforcing strict grouping constraints.
- **Deduplication:** Merged original leaf identifiers (`leaf-map.json`), exact MD5 hash collisions (11 instances), and perceptual hashes (pHash Hamming distance $\le 6$, 506 pairs merged).
- **Leakage Verification:** Cross-split perceptual distance auditing confirmed zero test images within a Hamming distance of $\le 6$ to any training image.

---

## Results and Performance Comparison

All results are evaluated on the held-out test split ($n = 3,464$ images across 1,557 groups) using the best checkpoint determined by validation loss. Intervals denote 95% group-bootstrap confidence intervals ($B = 1,000$ resamples).

### Table 1: Modality Ablation Study (`outputs/tables/table3_modality_ablation.md`)

| Model Variant | Crop Acc. | Health Acc. | Cause Acc.* | Condition Acc. | Severity Macro-F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Image-only** | **0.990** [0.986, 0.993] | 0.995 [0.993, 0.997] | 1.000 [1.000, 1.000] | **0.967** [0.960, 0.973] | 0.825 [0.805, 0.843] |
| **Environment-only** *(simulated)* | 0.456 [0.430, 0.485] | 0.783 [0.752, 0.811] | 1.000 [1.000, 1.000] | 0.088 [0.075, 0.101] | 0.236 [0.217, 0.254] |
| **Multimodal Fusion** | **0.990** [0.986, 0.993] | **0.997** [0.994, 0.998] | 1.000 [1.000, 1.000] | **0.969** [0.961, 0.975] | **0.831** [0.811, 0.848] |
| **Fusion (Shuffled Env Control)** | **0.990** [0.986, 0.993] | **0.997** [0.994, 0.999] | 1.000 [1.000, 1.000] | **0.969** [0.962, 0.975] | 0.830 [0.810, 0.848] |
| **Majority-Class Baseline** | 0.408 | 0.783 | 1.000 | 0.022 | 0.208 |

*\*Note: Cause accuracy of 1.000 is mathematically trivial because only Biotic labels exist in the training distribution.*

#### Modality Insights:
- **Environmental Data Contributes Zero Predictive Value:** The Multimodal Fusion model, Image-only model, and Shuffled-Environment control are statistically indistinguishable across every evaluated head.
- **Permutation Importance:** Permuting any environmental feature at test time alters condition accuracy by $\le 0.07$ percentage points, confirming the neural network ignores the synthetic environmental inputs.

---

### Table 2: Laboratory vs. Field Generalization Gap (`outputs/tables/table4_lab_vs_field.md`)

Evaluation of the Multimodal Fusion model across laboratory-acquired photos (PlantVillage) versus uncontrolled agricultural field conditions (PlantDoc):

| Evaluation Metric | Lab Images (PlantVillage)<br>*(n = 3,258)* | Field Images (PlantDoc)<br>*(n = 206)* | Performance Difference<br>*(Field vs. Lab)* |
|---|:---:|:---:|:---:|
| **Health Accuracy** (All Samples) | 0.999 [0.998, 1.000] | 0.951 [0.922, 0.980] | **-4.8 pp** [-7.7, -1.9] |
| **Condition Accuracy** (All Samples) | 0.990 [0.986, 0.995] | 0.631 [0.565, 0.692] | **-35.9 pp** [-42.5, -29.8] |
| **Like-for-Like Biotic: Health Accuracy** | 1.000 [1.000, 1.000] | 0.951 [0.910, 0.984] | **-4.9 pp** [-9.0, -1.6] |
| **Like-for-Like Biotic: Condition Accuracy** | 0.978 [0.966, 0.987] | 0.541 [0.455, 0.631] | **-43.7 pp** [-52.0, -34.5] |

*(Like-for-like biotic subset = 11 overlapping conditions present in both test sources: Apple scab, Apple cedar rust, Grape black rot, and 8 Tomato conditions).*

---

### Table 3: Severity Prediction Breakdown (`outputs/tables/table6_severity.md`)

Performance of the cumulative ordinal severity head on unhealthy leaves with valid segmentation masks ($n = 2,664$):

| Severity Level | Damage Fraction Range | Precision | Recall | Macro-F1 | Support ($n$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Mild** | $< 10\%$ surface damage | 0.885 | 0.895 | 0.890 | 1,207 |
| **Moderate** | $10\% - 30\%$ surface damage | 0.744 | 0.758 | 0.751 | 833 |
| **Severe** | $> 30\%$ surface damage | 0.872 | 0.832 | 0.852 | 624 |
| **Overall Macro-F1** | — | — | — | **0.831** [0.811, 0.848] | 2,664 |

---

### Table 4: Multi-Seed Repeatability Analysis (`outputs/tables/seed_repeat_image.md`)

Four independent training seeds (Seeds 42, 1, 2, 3) evaluated on the fixed test partition:

| Metric | Seed 42 | Seed 1 | Seed 2 | Seed 3 | Mean $\pm$ Std Dev | Range (Min - Max) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall Condition Accuracy** | 0.967 | 0.970 | 0.970 | 0.968 | **0.969 $\pm$ 0.002** | 0.967 - 0.970 |
| **Lab Condition Accuracy** | 0.989 | 0.990 | 0.991 | 0.990 | **0.990 $\pm$ 0.001** | 0.989 - 0.991 |
| **Field Condition Accuracy** | 0.617 | 0.650 | 0.636 | 0.626 | **0.632 $\pm$ 0.014** | 0.617 - 0.650 |
| **Like-for-Like Field Biotic Acc.** | 0.549 | 0.557 | 0.533 | 0.566 | **0.551 $\pm$ 0.014** | 0.533 - 0.566 |
| **Severity Macro-F1** | 0.825 | 0.833 | 0.833 | 0.820 | **0.828 $\pm$ 0.006** | 0.820 - 0.833 |

---

## Evaluation Metrics

- **Categorical Accuracy:** Fraction of correctly classified samples across all classes for a given head.
- **Balanced Accuracy:** Macro-average of recall across all classes, preventing class imbalance skew.
- **Macro-Averaged Precision, Recall, and F1 Score:** Harmonic mean computed across all classes with positive support.
- **Cumulative Ordinal Loss:** Binary cross-entropy evaluating ordinal boundaries:
  $$\mathcal{L}_{\text{sev}} = \text{BCE}(\sigma(z_1), y > \text{Mild}) + \text{BCE}(\sigma(z_2), y > \text{Moderate})$$
- **Group-Bootstrap Confidence Intervals:** 1,000 resamples clustered by image group identifiers to ensure realistic variance estimation under data dependency.
- **Background Sanity Probe:** Classifiers trained on $16 \times 16$ Gaussian-blurred thumbnails and border strips to benchmark background context bias.

---

## Limitations and Known Issues

Detailed transparent findings are documented in [`outputs/LIMITATIONS.md`](outputs/LIMITATIONS.md):

1. **Simulated Environmental Features:** Temperature, rainfall, humidity, and soil moisture are synthetic Gaussian approximations seeded from file path hashes. The neural network learns that they carry zero causal signal, rendering multimodal fusion identical to image-only inference.
2. **Absence of Abiotic Stress Data:** No abiotic dataset (such as nitrogen or water stress) was available during training. Every unhealthy leaf is labeled `Biotic`. Consequently, the cause head **cannot diagnose abiotic stress**, and cause accuracy of 1.000 is an artifact of a single-class training distribution.
3. **Heuristic Severity Ground Truth:** Severity levels (`Mild`, `Moderate`, `Severe`) are produced by an automated GrabCut/HSV computer vision heuristic based on non-green surface area ratios. They do not constitute agronomist-validated pathology ground truth.
4. **Significant Field Generalization Drop:** While laboratory test accuracy reaches $99.0\%$, accuracy on unstructured field photos drops to $63.1\%$ overall and $54.1\%$ on disease-only samples. Real-world agricultural deployment requires domain-adapted fine-tuning.
5. **Background Confounding (Context Bias):** A linear classifier trained solely on heavily blurred $16 \times 16$ thumbnail backdrops achieves $63.0\%$ condition accuracy (random chance is $2.2\%$). Part of the high lab accuracy reflects laboratory lighting and background consistency.
6. **Confidence Calibration:** Output softmax scores are uncalibrated and tend toward overconfidence. Predictions on unrepresented crop species or out-of-distribution inputs may exhibit high confidence despite being incorrect.
7. **Single-Condition Crop Bias:** The dataset contains Orange images solely for *Haunglongbing* (Citrus greening) on uniform laboratory backdrops, with no healthy citrus counterparts.

---

## Project Structure

```
plant-model/
├── .vscode/                     # VS Code debug launcher and editor settings
│   ├── launch.json              # 7 pre-configured debugging tasks
│   └── settings.json            # Interpreter path and watcher exclusions
├── logs/                        # Detailed text execution logs and progress traces
│   ├── progress.log             # Timestamped pipeline event log
│   └── *.txt                    # Per-stage execution stdout/stderr records
├── models/                      # Model architectures, training histories, metadata
│   ├── env_stats.json           # Environmental normalization parameters (fitted on train)
│   ├── label_maps.json          # 27 diagnostic condition and crop class mappings
│   ├── env/                     # Environment-only variant (history.csv, config.json)
│   ├── fusion/                  # Multimodal fusion variant (history.csv, config.json)
│   ├── fusion_shuffled_env/     # Shuffled-env control variant
│   └── image/                   # Image-only baseline variant
├── outputs/                     # Verification artifacts, tables, and figures
│   ├── dataset_inspection.md    # Pre-training dataset audit report
│   ├── LIMITATIONS.md           # Comprehensive honest-reporting documentation
│   ├── explain/                 # Grad-CAM overlays and attention analysis
│   ├── figures/                 # Loss/accuracy curves and confusion matrices
│   ├── predict_demo/            # Seeded demo test images and demonstration notes
│   ├── predictions/             # Timestamped inference outputs and overlays
│   ├── severity_debug/          # GrabCut/HSV segmentation verification images
│   └── tables/                  # Markdown & CSV result tables with 95% bootstrap CIs
├── src/                         # Complete Python source code
│   ├── build_cache.py           # 256px JPEG cache builder
│   ├── config.py                # Hyperparameters, paths, and hardware constants
│   ├── dataset.py               # PyTorch Dataset loaders and augmentations
│   ├── demo_predict.py          # Demo inference script on seeded test samples
│   ├── env_features.py          # Environmental feature synthesis and scaling
│   ├── evaluate.py              # Test split evaluation and bootstrap CIs
│   ├── explain.py               # Grad-CAM generation and permutation importance
│   ├── extract_plantdoc.py      # Git-object extractor for Windows PlantDoc data
│   ├── infer.py                 # Core inference routines and strict checkpoint loader
│   ├── log_util.py              # Logging formatting utilities
│   ├── model.py                 # FruitHealthNet architecture and masked losses
│   ├── pipeline.py              # High-level pipeline driver
│   ├── predict.py               # Production CLI predictor with confidence checks
│   ├── prepare_data.py          # Dataset discovery and index construction
│   ├── sanity.py                # Blurred thumbnail and border strip sanity probes
│   ├── seed_repeat.py           # 4-seed statistical reproducibility runner
│   ├── severity.py              # GrabCut/HSV damage-fraction heuristic extractor
│   ├── smoke_test.py            # 18-step unit and integration test suite
│   ├── split_data.py            # Group-aware stratified train/val/test splitting
│   ├── train.py                 # Multi-task training engine with cosine decay
│   └── verify_cache.py          # Integrity validator for image cache
├── .gitignore                   # Excludes virtual environments, raw data, and binary weights
├── HOW_TO_RUN.md                # Comprehensive user guide with troubleshooting
├── README.md                    # Primary project documentation
├── requirements.txt             # Exact verified Python dependency versions
└── run_predict.bat              # Windows double-click and drag-and-drop launcher
```

---

## Future Improvements

- **Real IoT Sensor Integration:** Integrate physical telemetry streams (calibrated canopy sensors, microclimate hygrometers, and soil capacitance probes) to assess genuine multi-modal synergy.
- **Abiotic Stress Data Acquisition:** Expand the dataset with verified field and greenhouse imagery of nutrient deficiencies (Nitrogen, Potassium, Phosphorus) and drought/waterlogging stress.
- **Pathologist-Annotated Severity Benchmarks:** Transition from automated GrabCut surface heuristics to expert agronomist lesion annotations and pixel-level segmentation masks.
- **Domain Adaptation for Field Robustness:** Implement adversarial domain adaptation or contrastive representation learning to bridge the $35.9\%$ accuracy gap between laboratory and field imagery.
- **Uncertainty Calibration & OOD Detection:** Integrate temperature scaling and conformal prediction to provide mathematically calibrated uncertainty bounds and reject non-crop imagery.
- **Mobile & Edge Optimization:** Quantize the EfficientNet-B0 backbone into INT8 ONNX and TorchScript / TFLite formats for real-time mobile deployment without network connectivity.

---

## Citations and References

- **PlantVillage Dataset:** Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- **PlantDoc Dataset:** Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection in the Wild.* CoDS-COMAD 2020.
- **EfficientNet Backbone:** Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.