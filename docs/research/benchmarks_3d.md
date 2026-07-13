**3D matched procedural rooms (10 held-out, split seed 42): geometry vs pixels-only sensing. Depth-strip MAE 0.14 m**

| Policy | Episodes | Success rate [95% CI] | SPL [95% CI] | SoftSPL | DTS (m) | Steps (succ.) | Collisions | Coverage |
|---|---|---|---|---|---|---|---|---|
| fbe-geometry | 10 | 0.900 [0.700, 1.000] | 0.666 [0.472, 0.826] | 0.642 | 0.33 | 44 | 0.1 | 16.1 |
| fbe-vision (pixels only) | 10 | 0.700 [0.400, 0.900] | 0.598 [0.331, 0.837] | 0.645 | 1.06 | 31 | 1.8 | 12.1 |
| random | 10 | 0.100 [0.000, 0.300] | 0.077 [0.000, 0.232] | 0.127 | 3.39 | 74 | 46.8 | 28.7 |
| oracle (privileged) | 10 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.955 | 0.00 | 26 | 0.0 | 10.1 |
