# severity_v2 pilot summary (severity_v2-pilot-0.1)

Heuristic comparison only. Agreement between v1 and v2 is heuristic agreement, not accuracy.

## v1 reproducibility (recomputed v1 vs stored data/processed/severity.csv)

n = 59; |diff| median 0.0001, max 0.7877; same category 54/59; same flag 58/59

## Per stratum

| stratum | n | v1 median frac | v2 median frac | median v2-v1 | images with v2>v1+0.01 | v2<v1-0.01 | median restored px | v1 ok | v2 ok | v2 pale_dominant | category changed (both valid) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| white_mildew | 12 | 0.013 | 0.122 | +0.027 | 6 | 1 | 1022 | 12 | 12 | 0 | 6/12 |
| chlorosis | 12 | 0.101 | 0.101 | +0.000 | 4 | 2 | 0 | 12 | 12 | 0 | 2/12 |
| necrosis | 12 | 0.125 | 0.173 | +0.024 | 8 | 1 | 179 | 12 | 12 | 0 | 4/12 |
| grey_lesions | 12 | 0.047 | 0.102 | +0.048 | 9 | 0 | 790 | 12 | 12 | 0 | 3/12 |
| healthy | 12 | 0.049 | 0.149 | +0.002 | 5 | 2 | 722 | 12 | 12 | 0 | 3/12 |
| field | 12 | 0.190 | 0.260 | +0.041 | 9 | 2 | 569 | 7 | 8 | 0 | 1/7 |

## Category distributions (all rows; Invalid = mask failed)

**white_mildew**: v1 {'Mild': np.int64(10), 'Severe': np.int64(1), 'Moderate': np.int64(1)}; v2 {'Moderate': np.int64(7), 'Mild': np.int64(4), 'Severe': np.int64(1)}

**chlorosis**: v1 {'Mild': np.int64(6), 'Moderate': np.int64(3), 'Severe': np.int64(3)}; v2 {'Mild': np.int64(5), 'Moderate': np.int64(5), 'Severe': np.int64(2)}

**necrosis**: v1 {'Mild': np.int64(5), 'Moderate': np.int64(4), 'Severe': np.int64(3)}; v2 {'Moderate': np.int64(8), 'Severe': np.int64(3), 'Mild': np.int64(1)}

**grey_lesions**: v1 {'Mild': np.int64(8), 'Severe': np.int64(3), 'Moderate': np.int64(1)}; v2 {'Mild': np.int64(6), 'Severe': np.int64(4), 'Moderate': np.int64(2)}

**healthy**: v1 {'Mild': np.int64(7), 'Moderate': np.int64(3), 'Severe': np.int64(2)}; v2 {'Mild': np.int64(6), 'Moderate': np.int64(4), 'Severe': np.int64(2)}

**field**: v1 {'Moderate': np.int64(5), 'Mild': np.int64(4), 'Severe': np.int64(3)}; v2 {'Severe': np.int64(6), 'Mild': np.int64(4), 'Moderate': np.int64(2)}

## Heuristic agreement v1 (rows) x v2 (columns), unhealthy strata, both valid

| v1_severity | Mild | Moderate | Severe |
|---|---|---|---|
| Mild | 19 | 13 | 0 |
| Moderate | 0 | 8 | 2 |
| Severe | 0 | 1 | 12 |

lab: same category 33/48
field: same category 6/7

## Flags

| v1_flag | hsv_fallback | ok |
|---|---|---|
| hsv_fallback | 4 | 1 |
| ok | 0 | 67 |
