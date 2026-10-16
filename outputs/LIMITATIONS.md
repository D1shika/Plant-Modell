# Limitations and honest-reporting notes

Everything below describes what was actually done and measured (test split: 3,464 images in 1,557 groups; training
2026-10-14/16, single seed 42, 8 epochs per variant). Metrics and confidence intervals: `outputs/tables/`.

## 1. What the headline numbers do and do not mean
- Test results (Fusion / Image-only): crop 0.990, health 0.995-0.997, condition 0.967-0.969, severity macro-F1 0.83
  (heuristic labels). These are dominated by PlantVillage lab photographs (3,258 of 3,464 test images).
- On field photographs (PlantDoc, n=206 test images) the same model reaches health 0.951 (95% CI 0.922-0.980) but only
  condition 0.631 (0.565-0.692). Like-for-like biotic subset (11 conditions present in both sources): condition
  0.978 on lab (n=1,303) vs 0.541 on field (n=122), a gap of -43.7 percentage points (CI -52.0 to -34.5).
  Do not expect lab-level accuracy on real farm photos.
- Field results rest on a small test set (206 images; Tomato spider mites has 2 field images in total), so field
  intervals are wide and several field conditions cannot be assessed.
- Seed repeat (image-only model, seeds 42/1/2/3, data split fixed; `outputs/tables/seed_repeat_image.md`): condition
  accuracy 0.969 +/- 0.002 (range 0.967-0.970), lab 0.990 +/- 0.001, field 0.632 +/- 0.014 (0.617-0.650), like-for-like
  biotic field 0.551 +/- 0.014 (0.533-0.566), severity macro-F1 0.828 +/- 0.006. The lab-to-field gap therefore does not
  depend on the training seed. Training randomness is much smaller than the test-sampling uncertainty (field CIs are
  about +/-0.06); with only 4 runs the standard deviations are themselves imprecise.

## 2. Simulated environmental data
- Temperature, rainfall, humidity and soil moisture are SYNTHETIC (crop baseline + large Gaussian noise, seeded from the
  md5 of each image path; `src/env_features.py`). They are not sensor readings and carry no real causal information about
  health or disease. Crop identity leaks into them only through the crop-specific baselines, by design.
- Shuffled-environment control: Fusion vs Fusion(shuffled env) are statistically indistinguishable on every metric
  (e.g. condition 0.969 [0.961-0.975] vs 0.969 [0.962-0.975]); Image-only is also indistinguishable from Fusion
  (condition 0.967 [0.960-0.973]). Environment-only reaches health 0.783 (= majority-class baseline, a constant
  "Unhealthy" predictor) and condition 0.088 (majority 0.022; crop-prior effect only).
- Permutation importance: shuffling any single environment feature changes test health/condition accuracy by at most
  0.07 percentage points for both fusion models. The fusion models do not use the simulated inputs.
- Conclusion: this project provides NO evidence that environmental data improve leaf-health prediction. The predictor
  accepts environment values but they have essentially no effect.

## 3. Abiotic stress: not supported
- No abiotic (nutrient, drought, waterlogging) image data were obtained. OLID I is Kaggle-only; the user chose not to
  use Kaggle credentials, and searches of Hugging Face, GitHub and Zenodo found no usable copy.
- Every crop has zero abiotic examples. Cause is only Biotic (diseases and pests) or None (healthy). The cause head
  cannot learn Biotic vs Abiotic: its accuracy is trivially 1.000 (n=2,713 unhealthy test samples) and is reported as
  uninformative. `predict.py` only reports cause classes seen in training, i.e. never "Abiotic".
- Not producible: the abiotic severity table, and Table 5 (Tomato Disease / N_Deficiency / NK_Deficiency). Table 5 is
  recorded as UNAVAILABLE; a Tomato healthy-vs-disease matrix and a Tomato per-condition matrix are provided as
  clearly labelled, non-equivalent alternatives.

## 4. Heuristic severity labels
- Severity (Mild < 0.10 <= Moderate <= 0.30 < Severe damage fraction) comes from GrabCut/HSV image heuristics
  (`src/severity.py`), not from experts. The severity head is evaluated against these same heuristic labels
  (macro-F1 0.831 [0.811-0.848], n=2,664); agreement with them does not establish real-world severity accuracy.
- Mask quality: 17,709 of 18,132 unhealthy images have a valid mask; 337 used an HSV fallback and 86 failed (excluded).
  Only 613 of 929 unhealthy field images have a valid mask. Known artifacts: gray-brown lesions can be dropped from the
  leaf mask (damage undercounted), stems/edges/shadows can leak in, field masks are unreliable. Severity counts (valid):
  Mild 7,904 (44.6%), Moderate 5,359 (30.3%), Severe 4,446 (25.1%).

## 5. Datasets, coverage and imbalance
- Sources: PlantVillage (HF copy `mohanty/PlantVillage`, CC-BY-SA-3.0 per the dataset card; GitHub repo has no license
  field) and PlantDoc (`pratikkayal/PlantDoc-Dataset`, CC-BY-4.0). The Kaggle PlantVillage identifier was not used.
- Index: 23,559 images (PlantVillage 22,081 lab; PlantDoc 1,478 field), 27 conditions over 8 crops, up to 1,000
  PlantVillage images per condition (seed 42). Healthy 5,427 / Unhealthy 18,132.
- Orange has a single condition (Haunglongbing, PlantVillage only, 1,000 images): no healthy citrus, no field citrus,
  so Orange scores (1.000) cannot reflect any health or condition discrimination within that crop.
- PlantDoc field coverage: Cherry, Peach, Strawberry and Pepper have only healthy field images; field disease data
  exist only for Apple, Grape and Tomato. `Bell_pepper leaf spot` was excluded as ambiguous (not verified as
  Bacterial_spot); non-target crops (blueberry, corn, potato, raspberry, soybean, squash) were excluded.
- Smallest conditions (images): Apple Cedar_apple_rust 364, Tomato Mosaic_virus 427, Peach healthy 472, Grape healthy
  492. Class weights were NOT applied: the largest-to-smallest condition ratio in the training split is 2.99:1, just
  under the 3:1 trigger (logged in each `models/<variant>/config.json`).
- PlantDoc images are web-scraped (including stock photos with visible watermarks) and their labels were not
  independently re-verified.

## 6. Split grouping and leakage
- 70/15/15 stratified group split (seed 42): train 16,635 / val 3,460 / test 3,464 (70.6/14.7/14.7%); PlantDoc
  1,061/211/206. No path or group id appears in more than one split (asserted).
- Grouping levels: (a) original leaf for 18,348 of 22,081 PlantVillage images (from `leaf-map.json`; 621 Apple
  Black_rot images matched through its "Apple_Frogeye Spot" label, treated as an alias); (b) original-image name only
  for 3,733 PlantVillage images with no leaf entry (all Tomato Target Spot and Tomato mosaic virus images, Grape healthy
  and parts of other classes), which does NOT guarantee that photos of the same leaf stay together; (c) exact
  duplicates by MD5 (11 duplicate files, all inside one split; 0 hashes cross splits); (d) near-duplicates by pHash
  (Hamming <= 6; 506 pairs merged). PlantDoc has no reliable ids, so only (c) and (d) apply to it.
- Independent cross-split check (`outputs/tables/sanity_probe.md`): 0 test images within Hamming 6 of a train image;
  297 (8.6%) within 8 and 1,165 (33.6%) within 10. Closest pairs were inspected (`outputs/figures/crosssplit_pairs_preview.png`):
  different leaves and crops on the same gray lab backdrop, not same-leaf leakage.
- No claim of physical-plant-level independence is made; PlantVillage photos share capture conditions across splits.

## 7. Background / source probe (mandatory sanity check)
Classifiers that see only a heavily blurred 16x16 image (or only border strips), trained on the train split and scored on
the group-aware test split:

| inputs | target | accuracy | balanced acc | majority baseline |
|---|---|---|---|---|
| blurred 16x16 | cause (None vs Biotic) | 0.890 | 0.788 | 0.783 |
| blurred 16x16 | source dataset | 0.989 | 0.924 | 0.941 |
| blurred 16x16 | condition (27 classes) | 0.630 | 0.621 | 0.022 |
| border strips | cause | 0.863 | 0.722 | 0.783 |
| border strips | source | 0.991 | 0.943 | 0.941 |
| border strips | condition | 0.502 | 0.490 | 0.022 |

**Cause accuracy may reflect dataset background or source differences rather than leaf symptoms.** Colour layout and
backdrop alone recover about 63% of the 27-way condition label and the source dataset almost perfectly, so part of the
very high lab accuracy (and part of the lab-to-field drop) is a background/capture-condition effect; some of it is
legitimate (leaf colour, crop identity). The probe cannot separate these.

## 8. Other suspicious-result investigations
- Perfect scores: cause 1.000 (trivial, section 3); PlantVillage crop accuracy 1.000 on n=3,258 and Orange 1.000 - see
  sections 5 and 7 (uniform lab backdrops, one-condition crop). No exact duplicates cross splits.
- Large source gap: -35.9 pp condition accuracy lab vs field (Fusion); flagged in `outputs/tables/sanity_flags.md`.
- Confidence (Fusion, condition head): mean 0.57 on wrong predictions vs 0.89 on correct ones (below the 0.8 overconfidence
  flag), but 16% of the 76 wrong field predictions still had confidence >= 0.8 (12% of the 32 wrong lab ones); e.g. the
  demo Apple scab photo predicted as Tomato with crop confidence 0.93. Confidence values are softmax probabilities, not
  calibrated, and no out-of-distribution detector exists.
- Grad-CAM review (`outputs/explain/REVIEW.md`): attention is a coarse (7x7) central blob; often on leaf tissue or the
  most damaged region, sometimes on fruit or clutter in field photos; centre bias prevents claims that the model
  "looks at lesions" in a causal sense.

## 9. Acquisition problems and engineering compromises
- PlantDoc `git clone` failed at checkout (filenames containing `?` are invalid on Windows). Files were read from the
  downloaded git objects instead (`src/extract_plantdoc.py`, sha1-verified, 183 files renamed; mapping in
  `data/raw/plantdoc/_manifest.csv`). A first extraction with case-sensitive collision handling overwrote 6 files; it was
  preserved as `data/raw/plantdoc_attempt1_flawed/` and redone correctly.
- `leaf-map.json` is incomplete (no entries for several classes), hence the fallback grouping in section 6.
- Images are served from a 256x256 JPEG (q=95) cache that does not preserve aspect ratio, used identically for training,
  evaluation and `predict.py`. Batch size 16 (instead of the 4 GB-GPU default 8) and 3 loader workers were used by
  approval; one run (fusion_shuffled_env) was slowed to ~4 min/epoch by other programs competing for CPU (results
  unaffected).
- Hyper-parameters were not tuned; each variant trained 8 epochs (early stopping never triggered). env, fusion and
  fusion_shuffled_env used a single seed (42); the image-only model was repeated with three more seeds (section 10).
- Seed-repeat engineering notes: (a) a naming bug I introduced in `train.py --seed` (loss accumulator shadowed the run
  name) affected only the seed-1 run's metadata (`DONE.json` run field, curves filename), repaired afterwards (original
  kept as `DONE.json.bugged_copy`); checkpoints and histories were never affected and seeds 2-3 ran on the fixed code;
  (b) from epoch 6 the seed-3 run slowed ~10x because the laptop (on battery) ran out of memory and thrashed the disk;
  I stopped it after its epoch-6 checkpoint and resumed the final epoch with 1 loader worker instead of 3 (seed 3 epochs
  0-6: 3 workers, epoch 7: 1 worker). This changes only data-loading randomness streams, not the method.
- Pandas reads the label string `None` (healthy cause) as missing; loaders convert it back (cause is masked for healthy
  samples, so no training or evaluation result was affected).

## 10. Experiments not run
- Seed repeats of the env, fusion and fusion_shuffled_env variants (only the image-only model was repeated, with seeds
  1-3 plus the original 42, after approval). The "fusion = image-only = shuffled control" conclusion therefore rests on one
  seed for the fusion models; the image-only seed spread (about 0.002 on overall condition accuracy) is far smaller than
  the confidence intervals, so differences of that size should not be interpreted.
- SHAP (optional; not installed).
- Any evaluation on real sensor data, expert severity ratings, external field datasets, or abiotic data.
- Calibration analysis and out-of-distribution detection.

## 11. Appropriate use
Research/education demonstrator for eight crops and 27 biotic/healthy classes. Not suitable for agronomic or
treatment decisions, for diagnosing nutrient/water stress, or for crops and conditions outside the 27 supported classes.
