**3D matched procedural rooms (20 held-out, split seed 42): geometry vs pixels-only sensing. Depth-strip MAE 0.16 m**

| Policy | Episodes | Success rate [95% CI] | SPL [95% CI] | SoftSPL | DTS (m) | Steps (succ.) | Collisions | Coverage |
|---|---|---|---|---|---|---|---|---|
| fbe-geometry | 20 | 0.900 [0.750, 1.000] | 0.598 [0.464, 0.734] | 0.575 | 0.23 | 52 | 0.1 | 16.6 |
| fbe-vision (pixels only) | 20 | 0.700 [0.500, 0.900] | 0.543 [0.362, 0.719] | 0.546 | 1.16 | 40 | 2.5 | 12.9 |
| random | 20 | 0.050 [0.000, 0.150] | 0.039 [0.000, 0.116] | 0.079 | 3.91 | 74 | 49.2 | 30.4 |
| oracle (privileged) | 20 | 1.000 [1.000, 1.000] | 0.981 [0.944, 1.000] | 0.931 | 0.00 | 26 | 0.0 | 10.0 |
