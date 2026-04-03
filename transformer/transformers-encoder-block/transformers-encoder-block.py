import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    var = x.var(axis=-1, keepdims=True)
    mean = x.mean(axis=-1, keepdims=True)
    
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, L, d_model = Q.shape
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    Q, K, V = (
        arr.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        for arr in (Q, K, V)
    )
    attn_scores = softmax(Q @ K.transpose(0, 1, 3, 2) / d_model ** 2)
    attn_weights = (attn_scores @ V).transpose(0, 2, 1, 3).reshape(B, L, d_model)
    return attn_weights @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x1 = x + attn
    x = layer_norm(x1, gamma1, beta1)

    ff = feed_forward(x, W1, b1, W2, b2)
    x2 = x + ff
    output = layer_norm(x2, gamma2, beta2)

    return output