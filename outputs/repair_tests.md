# Repair tests: Phase B (safe inference fixes)

Environment used for the runs below: the Linux file sandbox attached to `C:\Plant model` (Python 3.10, numpy 2.2.6, pandas 2.3.3, **no torch**). This is **not** the project `.venv`. Results that need torch are marked **NOT RUN (needs the Windows .venv)**.

| Test script | What it covers | Result (2026-10-20, sandbox) |
|---|---|---|
| `tests/check_hashes.py` | SHA-256 of all 15 checkpoints, the backbone cache, `label_maps.json`, `env_stats.json`, and the processed CSVs against `logs/repair/hashes_before_phaseB.txt` | **PASSED**: 26/26 identical after the Phase B changes |
| `tests/test_reconcile.py` | Synthetic-logit unit tests: metadata, crop/health aggregation, report examples, healthy handling, cause unavailable, unknown or missing classes, ordinal projection, batch vs single, temperature hook | **PASSED** 32/32 |
| `tests/test_saved_predictions.py` | Saved test predictions (read-only): inconsistencies before/after, raw vs reconciled accuracy with group-bootstrap CIs, ordinal violations before/after, severity under predicted-health gating | **PASSED** 7/7 |
| `tests/test_venv_inference.py` | Strict loading of all 4 checkpoints on CPU; default variant = image; single and batch prediction; Grad-CAM; cause unavailable; env flags rejected for image-only and accepted for fusion; corrupt, tiny, portrait, landscape, transparent, BMP and EXIF inputs; GPU if available; label maps and `outputs/predictions` untouched | **NOT RUN (needs the Windows .venv)** |
| `src/smoke_test.py` (edited to be non-destructive) | Original 18 checks | **NOT RUN (needs the Windows .venv)** |

## Measured on saved test predictions (copied inputs: `outputs/tables/preds_{image,fusion}.csv`; recomputed here)

These are reporting numbers on the held-out test split. Nothing was tuned on them.

| | image-only | fusion |
|---|---|---|
| Crop/health/condition inconsistencies, before → after | 21 (19 field) → **0** | 19 (18 field) → **0** |
| Condition accuracy (unchanged by design) | 0.967 [0.960, 0.973] | 0.969 [0.961, 0.975] |
| Crop accuracy, all: raw head → reconciled | 0.990 → 0.989 | 0.990 → 0.989 |
| Crop accuracy, field (n = 206): raw → reconciled | 0.835 [0.782, 0.883] → 0.816 [0.765, 0.865] | 0.835 → 0.816 |
| Health accuracy, field: raw → reconciled | 0.942 → 0.942 | 0.951 [0.922, 0.980] → 0.937 [0.903, 0.971] |
| Ordinal violations P(>Moderate) > P(>Mild), before → after | 105 → **0** | 115 → **0** |
| Severity class changed by the projection | 0 (decoding identical) | 0 |
| Severity macro-F1 vs heuristic labels, true-health gating → user pathway (predicted-health gating) | 0.825 → 0.825 | 0.831 → 0.830 |
| Same, field only (n = 81) | 0.685 → 0.663 | 0.724 → 0.705 |

**Plain reading:** the displayed answer is now always internally consistent, but recognition did not improve.

* Condition accuracy is identical, because the final condition is still the condition head's top answer.
* On field photos, deriving crop from the condition is slightly worse than the separate crop head (4 of 206 images). The confidence intervals overlap.
* In those cases the separate crop head was right while the condition was wrong. Every such case is now flagged "Uncertain" because the heads disagree, so the user is warned instead of being shown a contradiction.

## Commands to run the remaining tests in the project venv (from `C:\Plant model\fruit_health`)

```powershell
.\.venv\Scripts\python.exe -I tests\run_tests.py      # all four scripts (about 1-2 min)
.\.venv\Scripts\python.exe -I src\smoke_test.py       # original 18 checks, now non-destructive
.\.venv\Scripts\python.exe -I tests\check_hashes.py   # re-confirm checkpoints untouched after the runs
```
