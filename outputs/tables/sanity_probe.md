# Background/source probe (Part 12.1)

Inputs are ONLY blurred 16x16 images or border strips; train/test are the leakage-resistant group splits. 'cause' = None vs Biotic (no abiotic data exist).

| features | target | classifier | classes | n_train | n_test | accuracy | balanced acc | majority baseline |
|---|---|---|---|---|---|---|---|---|
| blurred 16x16 | cause | HistGB | 2 | 16635 | 3464 | 0.89 | 0.788 | 0.783 |
| blurred 16x16 | source | HistGB | 2 | 16635 | 3464 | 0.989 | 0.924 | 0.941 |
| blurred 16x16 | condition | LogReg | 27 | 16635 | 3464 | 0.63 | 0.621 | 0.022 |
| border strips | cause | HistGB | 2 | 16635 | 3464 | 0.863 | 0.722 | 0.783 |
| border strips | source | HistGB | 2 | 16635 | 3464 | 0.991 | 0.943 | 0.941 |
| border strips | condition | LogReg | 27 | 16635 | 3464 | 0.502 | 0.49 | 0.022 |

**WARNING: Cause accuracy may reflect dataset background or source differences rather than leaf symptoms.**

Source dataset is highly identifiable from background/border pixels alone (accuracy >= 0.90): lab vs field differences are partly a background effect.

Condition (27 classes) from background-only inputs: blurred 16x16: acc 0.63 (balanced 0.621, majority 0.022), border strips: acc 0.502 (balanced 0.49, majority 0.022)

# Cross-split near-duplicate check (test images vs nearest TRAIN image, pHash of the 256 px cache)

| Hamming <= | test images | share |
|---|---|---|
| 4 | 0 | 0.0000 |
| 6 | 0 | 0.0000 |
| 8 | 297 | 0.0857 |
| 10 | 1165 | 0.3363 |
| 12 | 2287 | 0.6602 |

The 25 closest cross-split pairs are in crosssplit_nearest_pairs.csv for visual inspection.
Exact duplicates (MD5) across splits: 0 hash(es) appear in more than one split; 11 duplicate files in total.
