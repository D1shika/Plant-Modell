### Per-source and per-crop metrics with sample counts (test split)

| model | group_type | group | n | crop_acc | crop_n | health_acc | health_n | condition_acc | condition_n | cause_acc | cause_n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Image-only | source | PlantDoc | 206 | 0.835 | 206 | 0.942 | 206 | 0.617 | 206 | 1.000 | 122 |
| Image-only | source | PlantVillage | 3258 | 1.000 | 3258 | 0.999 | 3258 | 0.989 | 3258 | 1.000 | 2591 |
| Image-only | crop | Apple | 411 | 0.985 | 411 | 0.995 | 411 | 0.978 | 411 | 1.000 | 261 |
| Image-only | crop | Cherry | 214 | 0.972 | 214 | 1.000 | 214 | 0.972 | 214 | 1.000 | 137 |
| Image-only | crop | Grape | 525 | 0.992 | 525 | 1.000 | 525 | 0.990 | 525 | 1.000 | 451 |
| Image-only | crop | Orange | 150 | 1.000 | 150 | 1.000 | 150 | 1.000 | 150 | 1.000 | 150 |
| Image-only | crop | Peach | 219 | 0.986 | 219 | 0.995 | 219 | 0.986 | 219 | 1.000 | 150 |
| Image-only | crop | Pepper_bell | 300 | 0.980 | 300 | 0.987 | 300 | 0.973 | 300 | 1.000 | 151 |
| Image-only | crop | Strawberry | 232 | 0.996 | 232 | 0.996 | 232 | 0.996 | 232 | 1.000 | 150 |
| Image-only | crop | Tomato | 1413 | 0.994 | 1413 | 0.994 | 1413 | 0.942 | 1413 | 1.000 | 1263 |
| Environment-only | source | PlantDoc | 206 | 0.500 | 206 | 0.592 | 206 | 0.049 | 206 | 1.000 | 122 |
| Environment-only | source | PlantVillage | 3258 | 0.453 | 3258 | 0.795 | 3258 | 0.091 | 3258 | 1.000 | 2591 |
| Environment-only | crop | Apple | 411 | 0.404 | 411 | 0.635 | 411 | 0.010 | 411 | 1.000 | 261 |
| Environment-only | crop | Cherry | 214 | 0.070 | 214 | 0.640 | 214 | 0.252 | 214 | 1.000 | 137 |
| Environment-only | crop | Grape | 525 | 0.297 | 525 | 0.859 | 525 | 0.154 | 525 | 1.000 | 451 |
| Environment-only | crop | Orange | 150 | 0.000 | 150 | 1.000 | 150 | 0.573 | 150 | 1.000 | 150 |
| Environment-only | crop | Peach | 219 | 0.000 | 219 | 0.685 | 219 | 0.032 | 219 | 1.000 | 150 |
| Environment-only | crop | Pepper_bell | 300 | 0.000 | 300 | 0.503 | 300 | 0.030 | 300 | 1.000 | 151 |
| Environment-only | crop | Strawberry | 232 | 0.000 | 232 | 0.647 | 232 | 0.125 | 232 | 1.000 | 150 |
| Environment-only | crop | Tomato | 1413 | 0.878 | 1413 | 0.894 | 1413 | 0.025 | 1413 | 1.000 | 1263 |
| Fusion | source | PlantDoc | 206 | 0.835 | 206 | 0.951 | 206 | 0.631 | 206 | 1.000 | 122 |
| Fusion | source | PlantVillage | 3258 | 1.000 | 3258 | 0.999 | 3258 | 0.990 | 3258 | 1.000 | 2591 |
| Fusion | crop | Apple | 411 | 0.981 | 411 | 0.998 | 411 | 0.976 | 411 | 1.000 | 261 |
| Fusion | crop | Cherry | 214 | 0.977 | 214 | 1.000 | 214 | 0.972 | 214 | 1.000 | 137 |
| Fusion | crop | Grape | 525 | 0.994 | 525 | 1.000 | 525 | 0.989 | 525 | 1.000 | 451 |
| Fusion | crop | Orange | 150 | 1.000 | 150 | 1.000 | 150 | 1.000 | 150 | 1.000 | 150 |
| Fusion | crop | Peach | 219 | 0.995 | 219 | 0.995 | 219 | 0.995 | 219 | 1.000 | 150 |
| Fusion | crop | Pepper_bell | 300 | 0.983 | 300 | 0.990 | 300 | 0.977 | 300 | 1.000 | 151 |
| Fusion | crop | Strawberry | 232 | 0.996 | 232 | 0.996 | 232 | 0.996 | 232 | 1.000 | 150 |
| Fusion | crop | Tomato | 1413 | 0.992 | 1413 | 0.996 | 1413 | 0.946 | 1413 | 1.000 | 1263 |
| Fusion (shuffled env) | source | PlantDoc | 206 | 0.830 | 206 | 0.956 | 206 | 0.636 | 206 | 1.000 | 122 |
| Fusion (shuffled env) | source | PlantVillage | 3258 | 1.000 | 3258 | 0.999 | 3258 | 0.990 | 3258 | 1.000 | 2591 |
| Fusion (shuffled env) | crop | Apple | 411 | 0.983 | 411 | 0.998 | 411 | 0.973 | 411 | 1.000 | 261 |
| Fusion (shuffled env) | crop | Cherry | 214 | 0.977 | 214 | 1.000 | 214 | 0.977 | 214 | 1.000 | 137 |
| Fusion (shuffled env) | crop | Grape | 525 | 0.992 | 525 | 1.000 | 525 | 0.989 | 525 | 1.000 | 451 |
| Fusion (shuffled env) | crop | Orange | 150 | 1.000 | 150 | 1.000 | 150 | 1.000 | 150 | 1.000 | 150 |
| Fusion (shuffled env) | crop | Peach | 219 | 0.995 | 219 | 0.995 | 219 | 0.995 | 219 | 1.000 | 150 |
| Fusion (shuffled env) | crop | Pepper_bell | 300 | 0.980 | 300 | 0.990 | 300 | 0.977 | 300 | 1.000 | 151 |
| Fusion (shuffled env) | crop | Strawberry | 232 | 0.996 | 232 | 0.996 | 232 | 0.996 | 232 | 1.000 | 150 |
| Fusion (shuffled env) | crop | Tomato | 1413 | 0.992 | 1413 | 0.996 | 1413 | 0.946 | 1413 | 1.000 | 1263 |
