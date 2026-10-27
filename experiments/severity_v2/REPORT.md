# Phase C: severity_v2 pilot report (exploratory)

**Conclusion: NOT PROMISING in its current form.** The targeted fix recovers some white powdery-mildew patches that v1 misses. It also adds a similar amount of false "damage" to healthy leaves: dull, hairy or glossy healthy tissue and some shadows. On this sample the two effects cannot be separated. Nothing was changed in production, and the existing labels are untouched.

The labels here are heuristic, not expert-verified. Agreement between v1 and v2 is **heuristic agreement**, not accuracy, and nothing in this report measures real-world severity accuracy.

## 1. What was tested

* **v1** (`src/severity.py`, unchanged). The leaf is found with GrabCut. Then `leaf & (green | S>=40)` (line 78) removes every low-saturation pixel from the leaf, which means white or grey symptoms can never count as damage.
* **v2** (`experiments/severity_v2/severity_v2.py`) makes one targeted change. Low-saturation GrabCut pixels are restored to the leaf only where they are enclosed by leaf tissue: v1's core is closed with a 7×7 kernel, its holes are filled, and the result is intersected with GrabCut. Low-saturation pixels connected to the outside stay excluded.
* Everything else is identical to v1: the resize, GrabCut, the 5%/97% validity rules, the HSV fallback, the healthy-green definition, the erosion, the damage definition, the 2% failure rule, and the thresholds 0.10/0.30. The category semantics were not redefined.
* v2 adds a per-component breakdown (chlorotic, necrotic, pale, dark). These are reporting only; the damage fraction is unchanged by them. It also adds a `pale_dominant` invalid flag when pale pixels exceed 50% of the leaf; it never fired on this sample.
* No parameter was tuned on this sample. The values were fixed before the run and the method was not revised after seeing results.

## 2. Sample (`out/sample_manifest.csv`)

* TRAIN split only, seed 42, drawn uniformly at random within strata.
* No exclusions, and no unreadable files.
* 72 images in 6 strata of 12. This is small, so the results are exploratory.

| stratum | source | conditions |
|---|---|---|
| white_mildew | lab | Cherry___Powdery_mildew |
| chlorosis | lab | Orange___Haunglongbing 6, Tomato___Yellow_Leaf_Curl_Virus 6 |
| necrosis | lab | Tomato___Early_blight 6, Grape___Black_rot 6 |
| grey_lesions | lab | Tomato___Septoria_leaf_spot 6, Apple___Apple_scab 6 |
| healthy | lab | pooled *___healthy (Tomato 3, Pepper 3, Peach 2, Cherry 2, Apple 1, Strawberry 1) |
| field | field (PlantDoc) | pooled unhealthy (Septoria 5, Black rot 2, Cedar rust 2, Early blight, Leaf mold, YLCV) |

The plan said about 60 images; 72 were used so that every stratum has 12.

The expectations were stated before the run. If the fix works, damage should rise mainly in white_mildew and grey_lesions, change little in chlorosis and necrosis, and **stay low on healthy leaves**.

## 3. Results (Linux sandbox run; full table in `out/severity_v2_pilot.csv`, summary in `out/pilot_summary.md`)

| stratum | n | v1 median | v2 median | median paired change | v2 up >0.01 | v2 down >0.01 | category changed |
|---|---|---|---|---|---|---|---|
| white_mildew | 12 | 0.013 | 0.122 | +0.027 | 6 | 1 | 6/12 (all Mild→Moderate) |
| chlorosis | 12 | 0.101 | 0.101 | +0.000 | 4 | 2 | 2/12 |
| necrosis | 12 | 0.125 | 0.173 | +0.024 | 8 | 1 | 4/12 |
| grey_lesions | 12 | 0.047 | 0.102 | +0.048 | 9 | 0 | 3/12 |
| **healthy (false damage)** | 12 | 0.049 | 0.149 | +0.002 | **5** | 2 | 3/12 (Mild→Moderate, Moderate→Severe, Severe→Moderate) |
| field | 12 | 0.190 | 0.260 | +0.041 | 9 | 2 | 1/7 valid in both |

**Category distributions:**

| stratum | v1 (Mild / Moderate / Severe) | v2 (Mild / Moderate / Severe) |
|---|---|---|
| white_mildew | 10 / 1 / 1 | 4 / 7 / 1 |
| chlorosis | 6 / 3 / 3 | 5 / 5 / 2 |
| necrosis | 5 / 4 / 3 | 1 / 8 / 3 |
| grey_lesions | 8 / 1 / 3 | 6 / 2 / 4 |
| healthy | 7 / 3 / 2 | 6 / 4 / 2 |
| field | 4 / 5 / 3 | 4 / 2 / 6 |

No image was Invalid under either method.

**Heuristic agreement, unhealthy strata where both labels are valid (v1 rows × v2 columns):**

| v1 \ v2 | Mild | Moderate | Severe |
|---|---|---|---|
| Mild | 19 | 13 | 0 |
| Moderate | 0 | 8 | 2 |
| Severe | 0 | 1 | 12 |

* Lab: 33/48 the same; field: 6/7 the same.
* v2 almost only moves images upward, as expected from a change that can only add leaf pixels.

**Flags:**
* v1 ok 67, hsv_fallback 5.
* v2 ok 68, hsv_fallback 4. One field image moved from fallback to ok: when the leaf area changes, the share rule can flip.

## 4. Visual inspection (8 panels: image | v1 damage in red | v2 components)

* **Mildew, 000 and 010:** v2 marks visible white mildew patches (cyan) that v1 ignores. This is the intended effect, but diffuse mildew is still largely missed.
* **Healthy tomato 051 and early blight 025:** v2 marks large areas of hairy, dull but healthy tissue as "pale". Those pixels are desaturated green (S<40), and the green definition needs S≥40. This is the main failure: restoring low-saturation pixels also restores healthy tissue that the green test calls non-green.
* **Healthy peach 052:** specular glare is counted as pale damage, and v2 rises from 0.093 to 0.279.
* **Apple scab 044 and early blight 025:** shadow strips along the leaf margin are now kept. When the enclosed gaps are filled, neighbouring pixels survive the erosion that used to strip them. This is an interaction with the erosion that was not anticipated.
* **Field 063:** both methods segment a tomato fruit instead of the leaf. GrabCut failures on field photos are inherited and dominate the field results.

## 5. Extra finding about v1 itself

* **Healthy leaves already get substantial false damage under v1.** 5 of 12 healthy leaves are Moderate or Severe, with a maximum of 0.781. This comes mainly from shadows and dull tissue.
* **v1 is not fully reproducible across OpenCV builds.** v1 recomputed here (Linux, cv2 5.0.0) matches the stored `severity.csv` to a median of 0.0001. But 8/59 images differ by more than 0.005, 5/59 change category, and 3 field images differ greatly: 0.001 vs 0.573, 0.929 vs 0.141, and mask_failed vs 0.285. GrabCut results on hard images depend on the build.
* So any comparison of severity methods should be run in the same environment that produced the labels. The Windows commands are below.

## 6. Limitations and bias

* There is no ground truth. Only 72 images (12 per stratum) were used, so medians are unstable and there are no confidence intervals worth quoting.
* The strata were chosen by condition name. The labels are heuristic and field masks are often wrong.
* This run was on Linux with a different OpenCV build from the one that produced the stored labels.
* The visual check covered 8 panels and was done by Claude, not by a plant pathologist.

## 7. Recommendation

* Do **not** adopt v2 or regenerate labels with it.
* If severity is revisited, a v3 should be pre-specified and tested on a **fresh** seeded sample (not this one, which has now been seen):
  * Count a low-saturation pixel as damage only when it is clearly whiter than the surrounding leaf (local contrast), instead of whenever S<40.
  * Restore pixels *after* erosion so that margin shadows are not re-admitted.
  * Most valuable of all, collect a small expert-annotated set (even 50–100 masks or severity grades); without it, no method can be shown to be better.
* Separately, v1's false damage on healthy leaves and its dependence on the OpenCV build are worth noting in the model documentation (Phase E).
