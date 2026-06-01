small_(aij/pij/vis) are small sample datasets from DNS.
To read, execute:
```
import numpy as np

N = 1000
T = 100
aij = np.fromfile('small_aij.bin').reshape([N,T,3,3])
pij = np.fromfile('small_pij.bin').reshape([N,T,3,3])
vis = np.fromfile('small_vis.bin').reshape([N,T,3,3])
```
