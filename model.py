#
# ============== File: model.py ==============
#
import torch
import torch.nn as nn
from attention_student import CustomStridedAttention # You will implement this

class SimpleTransformerLayer(nn.Module):
    """A single, simplified transformer layer."""
    def __init__(self, embed_dim, num_heads, stride):
        super().__init__()
        self.attention = CustomStridedAttention(embed_dim, num_heads, stride)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class SimpleModel(nn.Module):
    """A simple model with one transformer layer."""
    def __init__(self, embed_dim, num_heads, stride):
        super().__init__()
        self.layer = SimpleTransformerLayer(embed_dim, num_heads, stride)

    def forward(self, x):
        return self.layer(x)
