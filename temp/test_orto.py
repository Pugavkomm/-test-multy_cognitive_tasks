import numpy as np
from scipy.linalg import det

A = np.eye(4)
for i in range(4):
    A[i, i] = 1

print(det(A))
