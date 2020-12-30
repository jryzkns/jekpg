import jekpglib


import numpy as np

M = 10
N = 16

v = np.array([i for i in range(M*N)]).reshape((M,N))

print("original", v.shape)
print(v)

v = jekpglib.subsample2x(v)

print("subsampled", v.shape)
print(v)

v = jekpglib.upsample2x(v).astype(np.int16)

print("reupsampled", v.shape)
print(v)

