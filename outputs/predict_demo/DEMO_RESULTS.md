# Predictor demo on held-out test images (3 per available source, seed 42)

## PlantVillage

```
Environment values not provided: using training averages

Image: C:\Plant model\fruit_health\outputs\predict_demo\PlantVillage\5984db18-75d7-4c3a-be4b-74c3c0351887___JR_FrgE.S_8685.JPG
  Crop:       Apple (confidence 0.87)
  Health:     Unhealthy (confidence 0.94)
  Cause:      Biotic [only class seen in training; no abiotic data, so abiotic stress cannot be identified - not a meaningful prediction]
  Condition:  Apple___Black_rot (confidence 0.84)
  Severity:   Mild (confidence 0.71) [heuristic damage estimate]
  Grad-CAM:   C:\Plant model\fruit_health\outputs\predictions\20261016_111842\cam_5984db18-75d7-4c3a-be4b-74c3c0351887___JR_FrgE.S_8685_cafa5c.png

Image: C:\Plant model\fruit_health\outputs\predict_demo\PlantVillage\cfe02ced-6be9-4119-933f-75f95b88a940___JR_B.Spot_8838.JPG
  Crop:       Pepper_bell (confidence 0.93)
  Health:     Unhealthy (confidence 0.94)
  Cause:      Biotic [only class seen in training; no abiotic data, so abiotic stress cannot be identified - not a meaningful prediction]
  Condition:  Pepper_bell___Bacterial_spot (confidence 0.90)
  Severity:   Moderate (confidence 0.93) [heuristic damage estimate]
  Grad-CAM:   C:\Plant model\fruit_health\outputs\predictions\20261016_111842\cam_cfe02ced-6be9-4119-933f-75f95b88a940___JR_B.Spot_8838_8d9fbc.png

Image: C:\Plant model\fruit_health\outputs\predict_demo\PlantVillage\e40604ba-8f26-4960-b893-d4498f4fbf2e___FAM_B.Msls_1250.JPG
  Crop:       Grape (confidence 0.91)
  Health:     Unhealthy (confidence 0.96)
  Cause:      Biotic [only class seen in training; no abiotic data, so abiotic stress cannot be identified - not a meaningful prediction]
  Condition:  Grape___Esca (confidence 0.92)
  Severity:   Moderate (confidence 0.96) [heuristic damage estimate]
  Grad-CAM:   C:\Plant model\fruit_health\outputs\predictions\20261016_111842\cam_e40604ba-8f26-4960-b893-d4498f4fbf2e___FAM_B.Msls_1250_84d6f3.png

Confidence = softmax probability of the predicted class (not a calibrated probability).
Results written to C:\Plant model\fruit_health\outputs\predictions\20261016_111842\predictions.csv
```

Ground truth: cfe02ced-6be9-4119-933f-75f95b88a940___JR_B.Spot_8838.JPG: Pepper_bell / Unhealthy / Pepper_bell___Bacterial_spot; 5984db18-75d7-4c3a-be4b-74c3c0351887___JR_FrgE.S_8685.JPG: Apple / Unhealthy / Apple___Black_rot; e40604ba-8f26-4960-b893-d4498f4fbf2e___FAM_B.Msls_1250.JPG: Grape / Unhealthy / Grape___Esca

## PlantDoc

```
Environment values not provided: using training averages

Image: C:\Plant model\fruit_health\outputs\predict_demo\PlantDoc\Apple-Scab_1_1.jpg
  Crop:       Tomato (confidence 0.93)
  Health:     Unhealthy (confidence 0.93)
  Cause:      Biotic [only class seen in training; no abiotic data, so abiotic stress cannot be identified - not a meaningful prediction]
  Condition:  Tomato___Septoria_leaf_spot (confidence 0.73)
  Severity:   Severe (confidence 0.74) [heuristic damage estimate]
  Grad-CAM:   C:\Plant model\fruit_health\outputs\predictions\20261016_111922\cam_Apple-Scab_1_1_618f67.png

Image: C:\Plant model\fruit_health\outputs\predict_demo\PlantDoc\bact-spot-fig-1.jpg
  Crop:       Apple (confidence 0.87)
  Health:     Unhealthy (confidence 0.98)
  Cause:      Biotic [only class seen in training; no abiotic data, so abiotic stress cannot be identified - not a meaningful prediction]
  Condition:  Apple___Cedar_apple_rust (confidence 0.77)
  Severity:   Mild (confidence 0.52) [heuristic damage estimate]
  Grad-CAM:   C:\Plant model\fruit_health\outputs\predictions\20261016_111922\cam_bact-spot-fig-1_914de4.png

Image: C:\Plant model\fruit_health\outputs\predict_demo\PlantDoc\WildGrapeLeaf_cutout.jpg
  Crop:       Grape (confidence 0.94)
  Health:     Healthy (confidence 0.95)
  Cause:      None (healthy)
  Condition:  Grape___healthy (confidence 0.94)
  Severity:   None (healthy)
  Grad-CAM:   C:\Plant model\fruit_health\outputs\predictions\20261016_111922\cam_WildGrapeLeaf_cutout_4f8ba1.png

Confidence = softmax probability of the predicted class (not a calibrated probability).
Results written to C:\Plant model\fruit_health\outputs\predictions\20261016_111922\predictions.csv
```

Ground truth: Apple-Scab_1_1.jpg: Apple / Unhealthy / Apple___Apple_scab; bact-spot-fig-1.jpg: Tomato / Unhealthy / Tomato___Bacterial_spot; WildGrapeLeaf_cutout.jpg: Grape / Healthy / Grape___healthy

- OLID_I: UNAVAILABLE (no images in the dataset/test split)

## Outcome summary (checked against ground truth)
- PlantVillage (lab): 3/3 correct on crop, health and condition (Apple Black_rot, Pepper_bell Bacterial_spot, Grape Esca).
- PlantDoc (field): 1/3 fully correct. Grape healthy: correct. `Apple-Scab_1_1.jpg` (truth Apple scab) -> predicted
  Tomato / Septoria_leaf_spot with crop confidence 0.93 (confident wrong crop; health correct). `bact-spot-fig-1.jpg`
  (truth Tomato bacterial spot) -> predicted Apple / Cedar_apple_rust (wrong crop and condition; health correct).
- OLID I: unavailable (never obtained; Kaggle-only, no credentials used).
Three images per source is an illustration, not an accuracy estimate; see outputs/tables for measured test metrics.
Note the confidence values are softmax probabilities, not calibrated: wrong field predictions can be confident.
