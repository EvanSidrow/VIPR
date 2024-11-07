import autograd.numpy as np

# utility functions
def logdotexp(A, B):
    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    C = np.log(C)
    C = C + max_A + max_B
    return C
