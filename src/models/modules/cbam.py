"""CBAM adapted to token sequences (Supplementary Methods S1.3, Eqs. S4-S5).

Channel attention: avg/max pooling over the sequence dimension through a shared
MLP, then sigmoid. Position (spatial) attention: avg/max pooling over channels,
concatenated and fused by a 1-D convolution, then sigmoid.
"""
import torch
import torch.nn as nn


class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=5):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(          # Conv1d(k=1) == per-token Linear, shared across paths
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )
        self.spatial = nn.Conv1d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2)

    def forward(self, h):                  # h: (B, T, C)
        avg = h.mean(dim=1)                # (B, C)
        mx = h.max(dim=1).values           # (B, C)
        att = torch.sigmoid(self.mlp(avg.unsqueeze(-1)) + self.mlp(mx.unsqueeze(-1))).squeeze(-1)
        h = h * att.unsqueeze(1)
        avg_p = h.mean(dim=-1, keepdim=True)            # (B, T, 1)
        mx_p = h.max(dim=-1, keepdim=True).values
        spatial = torch.sigmoid(self.spatial(torch.cat([avg_p, mx_p], dim=-1).transpose(1, 2)))
        return h * spatial.transpose(1, 2)
