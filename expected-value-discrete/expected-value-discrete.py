import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)

    if (x.shape != p.shape):
        raise ValueError("Shape mismatch")
    if p.sum() != 1:
        raise ValueError("Prob must sum up to 1")

    return x @ p
