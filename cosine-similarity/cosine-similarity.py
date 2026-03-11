import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.array(a)
    b = np.array(b)
    dotProd = a @ b
    normMul = np.linalg.norm(a) * np.linalg.norm(b)

    if (normMul == 0.0):
        return 0.0;

    return dotProd / normMul
