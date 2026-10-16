# Dataset inspection (read-only; nothing downloaded) — 2026-10-14

Method: GitHub REST API, Hugging Face API, Zenodo API listings only.

## PlantVillage
- **GitHub** `spMohanty/PlantVillage-Dataset` (~2.1 GB repo): `raw/{color,grayscale,segmented}`; GitHub license field empty.
  Target-crop classes in `raw/color` (27 of the 8 crops): Apple (scab, black rot, cedar rust, healthy), Cherry (powdery mildew, healthy),
  Grape (black rot, esca, leaf blight, healthy), Orange (Haunglongbing only — no healthy), Peach (bacterial spot, healthy),
  Pepper,_bell (bacterial spot, healthy), Strawberry (leaf scorch, healthy), Tomato (9 diseases + healthy).
- **HF** `mohanty/PlantVillage`: license **CC-BY-SA-3.0** (per card), `data.zip` (~2 GB), `splits/`, and
  `leaf_grouping/leaf-map.json` + a `leaf_id` feature, i.e. a published original-leaf grouping usable for group splits.
- Notes: all classes are biotic or healthy; no abiotic. Orange has no healthy class.

## PlantDoc (cropped classification version)
- **GitHub** `pratikkayal/PlantDoc-Dataset` (~0.95 GB), license **CC-BY-4.0**, 2,598 images total, `train/` (28 class dirs) and `test/` (27).
- Mappable to target crops: Apple (Scab, rust, healthy "Apple leaf"), Bell_pepper (leaf spot, healthy), Cherry leaf (healthy only),
  Peach leaf (healthy only), Strawberry leaf (healthy only), grape (black rot, healthy), Tomato (early blight, septoria, bacterial spot,
  late blight, mosaic virus, yellow virus, mold, spider mites, healthy).
- **No Orange/citrus classes.** Ambiguous label names (e.g. "Bell_pepper leaf spot") need file-level confirmation before mapping.
- Only a README name; class-name-to-condition mapping is NOT yet built (needs the real files).

## OLID I (abiotic tomato)
- Kaggle only (`raiaone/olid-i`) — requires credentials; user prohibited credential use. No HF/GitHub copy found:
  HF "olid" hits are offensive-language NLP datasets; GitHub searches returned nothing.
- Zenodo 8252277 (tomato nitrogen levels, CC-BY-4.0) is a PDF paper only, no images.
- Search effort so far: HF (5 queries), GitHub (3), Zenodo (1). No usable abiotic image dataset found.

## Consequence
Without abiotic data, cause is always Biotic or None, so cause (Biotic/Abiotic), the abiotic severity table and the
Tomato Disease/N/NK confusion matrix cannot be produced honestly. Options: (a) user manually downloads OLID I from Kaggle
into data/raw (no credentials shared with Claude), (b) continue without abiotic and document it, (c) further search.
