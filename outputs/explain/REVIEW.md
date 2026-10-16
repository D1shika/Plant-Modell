# Grad-CAM review (fusion model, test split)

Overlays: `outputs/explain/fusion/<head>/<crop>/{correct|wrong}_<k>_true-<label>_pred-<label>.png` (health and condition
heads; up to 5 correct + 5 wrong per crop, up to 2 of the 5 slots reserved for field/PlantDoc images; 118 overlays, none
duplicated). The first, purely random selection (almost only lab images) is kept in
`_superseded_random_selection_fusion/`. A montage of 8 field overlays is `outputs/figures/gradcam_field_review.png`.

CAM = Grad-CAM on the last conv block of EfficientNet-B0 (7x7 grid) for the PREDICTED class. It is coarse and does not
prove causal reasoning.

## Quantitative attention summary (mean over the saved overlays; area = share of the 224x224 frame)
`attention_summary_fusion.csv`. Examples (condition head): lab/correct (n=26): 63% of CAM energy inside the heuristic leaf
mask that covers 34% of the frame (1.9x), 14% in the outer 10% frame that covers 35% (0.4x), 13% on the heuristic damage
mask covering 7% (1.8x). Field/correct (n=14): leaf 41% vs 25% area (1.6x), edge 21% vs 35%, damage 17% vs 12% (1.5x).
Field/wrong (n=19): leaf 42% vs 26%, damage 21% vs 14% (1.5x). Leaf and damage masks are heuristic and unreliable on
field images, so these ratios are indicative only.

## Visual review (what I actually looked at: 3 lab overlays + 8 field overlays)
- Lab images: attention mostly on the leaf body, often on or around the visible lesions; little on the backdrop.
- Field images: usually a single central blob on the leaf or on the most damaged part of the leaf (e.g. Grape black rot,
  Tomato Septoria, Tomato yellow-leaf-curl); not lesion-precise.
- Misclassified field examples: (a) an orchard photo of an apple tree, attention on the red FRUIT, predicted Peach healthy;
  (b) healthy Cherry predicted Peach healthy and healthy Grape (white stock background) predicted Strawberry healthy:
  attention on leaf tissue, error is crop confusion between visually similar healthy leaves.
- One correct healthy-Apple field image carries a visible stock-photo watermark across the leaf; web-scraped watermarks
  and plain studio backgrounds are possible shortcut cues in PlantDoc (not class-specific, but not leaf symptoms).
- Central bias: subjects are usually centred, so "attends to the lesion" cannot be separated from "attends to the middle".

## Environment permutation importance (simulated inputs)
See `outputs/tables/env_permutation_importance.md`: shuffling any single feature changes test health/condition accuracy
by at most 0.07 percentage points for both fusion and fusion_shuffled_env; cause is trivial. The fusion models do not use
the simulated environment inputs.
