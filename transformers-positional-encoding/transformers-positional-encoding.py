import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here

    pos_indices = np.arange(seq_length).reshape(-1, 1)
    dim_indicies = np.arange(0, d_model, 2)
    
    dim_freq = pos_indices * np.exp(dim_indicies * -np.log(10_000) / d_model).reshape(1, -1)

    
    pe = np.zeros((seq_length, d_model))
    
    pe[:, dim_indicies] = np.sin(dim_freq)
    pe[:, dim_indicies + 1] = np.cos(dim_freq)

    return pe
