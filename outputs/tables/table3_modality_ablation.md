### Table 3 - Modality ablation (test split; 95% group-bootstrap CI)

| Model | Crop accuracy | Health accuracy | Cause accuracy | Condition accuracy | Severity macro-F1 |
|---|---|---|---|---|---|
| Image-only | 0.990 [0.986, 0.993] n=3464 | 0.995 [0.993, 0.997] n=3464 | 1.000 [1.000, 1.000] n=2713 | 0.967 [0.960, 0.973] n=3464 | 0.825 [0.805, 0.843] n=2664 |
| Environment-only | 0.456 [0.430, 0.485] n=3464 | 0.783 [0.752, 0.811] n=3464 | 1.000 [1.000, 1.000] n=2713 | 0.088 [0.075, 0.101] n=3464 | 0.236 [0.217, 0.254] n=2664 |
| Fusion | 0.990 [0.986, 0.993] n=3464 | 0.997 [0.994, 0.998] n=3464 | 1.000 [1.000, 1.000] n=2713 | 0.969 [0.961, 0.975] n=3464 | 0.831 [0.811, 0.848] n=2664 |
| Fusion (shuffled env) | 0.990 [0.986, 0.993] n=3464 | 0.997 [0.994, 0.999] n=3464 | 1.000 [1.000, 1.000] n=2713 | 0.969 [0.962, 0.975] n=3464 | 0.830 [0.810, 0.848] n=2664 |
| Majority-class baseline | 0.408 n=3464 | 0.783 n=3464 | 1.000 n=2713 | 0.022 n=3464 | 0.208 n=2664 |

Cause accuracy is trivial: only Biotic labels exist (no abiotic data). Environment values are SIMULATED. Severity labels are heuristic. The shuffled-env row is the control: compare it with Fusion.
