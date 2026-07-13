**3D matched procedural rooms (10 held-out, split seed 42): geometry vs pixels-only sensing. Depth-strip MAE 0.12 m**

| Policy | Episodes | Success rate [95% CI] | SPL [95% CI] | SoftSPL | DTS (m) | Steps (succ.) | Collisions | Coverage |
|---|---|---|---|---|---|---|---|---|
| fbe-geometry | 10 | 0.900 [0.700, 1.000] | 0.666 [0.472, 0.826] | 0.642 | 0.33 | 44 | 0.1 | 16.1 |
| fbe-vision (pixels only) | 10 | 0.600 [0.300, 0.900] | 0.524 [0.254, 0.782] | 0.566 | 1.51 | 30 | 2.3 | 11.5 |
| random | 10 | 0.100 [0.000, 0.300] | 0.077 [0.000, 0.232] | 0.127 | 3.39 | 74 | 46.8 | 28.7 |
| oracle (privileged) | 10 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.955 | 0.00 | 26 | 0.0 | 10.1 |
