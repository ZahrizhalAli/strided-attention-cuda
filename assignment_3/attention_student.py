import torch
import torch.nn as nn
import math
from cuda.binding import strided_attention_forward # This will be our compiled CUDA kernel

def naive_strided_attention(q, k, v, stride):
    """
    Naive, pure-PyTorch implementation of strided attention.
    This is your reference for correctness.

    Args:
        q, k, v: Tensors of shape (batch_size, num_heads, seq_len, head_dim)
        stride: The stride for sparse attention.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create an empty output tensor
    output = torch.zeros_like(q)
    
    # Scale factor for dot products
    scale = 1.0 / math.sqrt(head_dim)

    # Iterate over each item in the batch and each head
    for b in range(batch_size):
        for h in range(num_heads):
            # (seq_len, head_dim)
            q_bh = q[b, h]  # shape [seq_len, head_dim]
            k_bh = k[b, h]  # shape [seq_len, head_dim]
            v_bh = v[b, h]  # shape [seq_len, head_dim]

            # --- Attention Score Calculation ---
            # Compute full attention scores and scale
            scores = torch.matmul(q_bh, k_bh.transpose(-2, -1)) * scale  # shape [seq_len, seq_len]

            # Mask out non-strided keys before softmax
            mask = torch.full_like(scores, float('-inf'))
            mask[:, ::stride] = 0
            scores = scores + mask

            # --- Softmax ---
            # Shift for numerical stability
            max_scores = scores.max(dim=-1, keepdim=True).values
            shifted = scores - max_scores
            exps = torch.exp(shifted)
            sums = exps.sum(dim=-1, keepdim=True) + 1e-6
            weights = exps / sums
            
            # --- Weighted Sum of Values ---
            out_bh = torch.matmul(weights, v_bh)  # shape [seq_len, head_dim]
            output[b, h] = out_bh

    return output


class CustomStridedAttention(nn.Module):
    """
    This module will use your custom CUDA kernel for strided attention.
    """
    def __init__(self, embed_dim, num_heads, stride):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.stride = stride
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Project and reshape Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # TODO: Replace the call to the naive implementation below with a call
        # to your custom CUDA kernel `strided_attention_forward`.
 
        # Using the naive implementation as a placeholder.
        # output = naive_strided_attention(q, k, v, self.stride)
        output = strided_attention_forward(q,k,v, self.stride)

        # Reshape and project the output back to the original shape
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output
