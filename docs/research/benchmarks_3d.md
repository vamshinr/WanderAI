**3D matched procedural rooms (10 held-out, split seed 42): geometry vs pixels-only sensing. Depth-strip MAE 0.14 m**

| Policy | Episodes | Success rate [95% CI] | SPL [95% CI] | SoftSPL | DTS (m) | Steps (succ.) | Collisions | Coverage |
|---|---|---|---|---|---|---|---|---|
| fbe-geometry | 10 | 0.900 [0.700, 1.000] | 0.701 [0.513, 0.870] | 0.659 | 0.72 | 36 | 0.1 | 14.2 |
| fbe-vision (pixels only) | 10 | 0.800 [0.500, 1.000] | 0.639 [0.398, 0.862] | 0.674 | 0.94 | 32 | 1.6 | 12.0 |
| random | 10 | 0.100 [0.000, 0.300] | 0.077 [0.000, 0.232] | 0.127 | 3.39 | 74 | 46.8 | 28.7 |
| oracle (privileged) | 10 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.955 | 0.00 | 26 | 0.0 | 10.1 |
