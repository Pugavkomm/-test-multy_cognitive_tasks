from scipy.linalg import orth, det
import numpy as np
from scipy.stats import ortho_group
A = np.eye(4)
for i in range(4):
    A[i, i] = 1
    
print(det(A))