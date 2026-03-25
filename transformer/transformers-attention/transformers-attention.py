import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.size(-1)
    return F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(d_k), dim=-1) @ V