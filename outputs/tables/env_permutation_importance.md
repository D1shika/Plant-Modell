# Environment permutation importance (SIMULATED env inputs)

Drop in accuracy when ONE env feature is shuffled across the test set (mean/sd of 10 repeats). Cause accuracy is trivial (only Biotic labels exist), so its drop is uninformative.

## fusion
Baseline test accuracy: health 0.9965 (n=3464), cause 1.0000 (n=2713), condition 0.9688 (n=3464)

| feature | d_health_acc_mean | d_cause_acc_mean | d_condition_acc_mean | d_health_acc_sd | d_cause_acc_sd | d_condition_acc_sd |
|---|---|---|---|---|---|---|
| temperature | -0.0002 | 0.0 | 0.0006 | 0.0001 | 0.0 | 0.0002 |
| rainfall | 0.0 | 0.0 | 0.0001 | 0.0 | 0.0 | 0.0002 |
| humidity | 0.0 | 0.0 | -0.0002 | 0.0001 | 0.0 | 0.0001 |
| soil_moisture | 0.0 | 0.0 | 0.0002 | 0.0001 | 0.0 | 0.0004 |

## fusion_shuffled_env
Baseline test accuracy: health 0.9968 (n=3464), cause 1.0000 (n=2713), condition 0.9688 (n=3464)

| feature | d_health_acc_mean | d_cause_acc_mean | d_condition_acc_mean | d_health_acc_sd | d_cause_acc_sd | d_condition_acc_sd |
|---|---|---|---|---|---|---|
| temperature | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | 0.0002 |
| rainfall | 0.0 | 0.0 | 0.0007 | 0.0 | 0.0 | 0.0003 |
| humidity | 0.0 | 0.0 | 0.0005 | 0.0 | 0.0 | 0.0002 |
| soil_moisture | 0.0 | 0.0 | 0.0003 | 0.0 | 0.0 | 0.0002 |

