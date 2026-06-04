aij/pij/vis_1024_dns.bin are tiny sample datasets sliced from DNS, used as
fixtures by the test suite (test/test_*.py). They hold the first N=16
trajectories of a longer run, stored as raw float64.

To read, execute:
```
import numpy as np

N = 16
T = 100
aij = np.fromfile('aij_1024_dns.bin').reshape([N, T, 3, 3])
pij = np.fromfile('pij_1024_dns.bin').reshape([N, T, 3, 3])
vis = np.fromfile('vis_1024_dns.bin').reshape([N, T, 3, 3])
```
