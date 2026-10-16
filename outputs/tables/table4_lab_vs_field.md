### Table 4 - Lab vs field (Fusion model, test split)

| Comparison | Lab (PlantVillage) | Field (PlantDoc) | Field - Lab |
|---|---|---|---|
| Health accuracy (all samples) | 0.999 [0.998, 1.000] n=3258 | 0.951 [0.922, 0.980] n=206 | -4.8 pp [-7.7, -1.9] |
| Cause accuracy (unhealthy; trivial) | 1.000 [1.000, 1.000] n=2591 | 1.000 [1.000, 1.000] n=122 | +0.0 pp [+0.0, +0.0] |
| Condition accuracy (all samples) | 0.990 [0.986, 0.995] n=3258 | 0.631 [0.565, 0.692] n=206 | -35.9 pp [-42.5, -29.8] |
| Like-for-like biotic: health accuracy | 1.000 [1.000, 1.000] n=1303 | 0.951 [0.910, 0.984] n=122 | -4.9 pp [-9.0, -1.6] |
| Like-for-like biotic: cause accuracy | 1.000 [1.000, 1.000] n=1303 | 1.000 [1.000, 1.000] n=122 | +0.0 pp [+0.0, +0.0] |
| Like-for-like biotic: condition accuracy | 0.978 [0.966, 0.987] n=1303 | 0.541 [0.455, 0.631] n=122 | -43.7 pp [-52.0, -34.5] |

Like-for-like biotic subset = unhealthy samples of conditions present in both sources' test sets (11): Apple___Apple_scab, Apple___Cedar_apple_rust, Grape___Black_rot, Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold, Tomato___Mosaic_virus, Tomato___Septoria_leaf_spot, Tomato___Spider_mites, Tomato___Yellow_Leaf_Curl_Virus. Differences are Field minus Lab in percentage points with independent group-bootstrap intervals. '-' = unavailable.
