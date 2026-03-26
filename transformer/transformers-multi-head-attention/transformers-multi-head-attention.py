import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    B, seq_len, d_model = Q.shape
    
    queries = Q @ W_q
    keys = K @ W_k
    values = V @ W_v

    queries, keys, values = (
        x.reshape(B, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
        for x in (queries, keys, values)
    )

    attn_scores = queries @ keys.transpose(0, 1, 3, 2)
    attn_weights = softmax(attn_scores / d_model ** 0.5) # (B, h, L, L)
    context_vec = (attn_weights @ values).transpose(0, 2, 1, 3).reshape(B, seq_len, d_model)
    return context_vec @ W_o