from __future__ import annotations
import math
import torch
import torch.nn as nn


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0, 'out_dim must be divisible by num_heads'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        # x: [B, T, N, D], adj: [N, N]
        B, T, N, D = x.shape
        q = self.q_proj(x).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(x).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = (adj > 0).to(x.device)[None, None, None, :, :]
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, self.out_dim)
        return self.out_proj(out)
