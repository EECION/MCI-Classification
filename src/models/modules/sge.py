"""Spatial group enhance for sequences (Supplementary Methods S4.2, Eqs. S16-S17).

Channels are split into G groups; per group, the cosine similarity between each
position's vector and the group's global descriptor is standardized across
positions and mapped through a learnable sigmoid gate (gamma, beta).
"""
import torch
import torch.nn as nn


class SGE(nn.Module):
    def __init__(self, channels, groups=8, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "channels must be divisible by groups"
        self.groups = groups
        self.gamma = nn.Parameter(torch.zeros(1, groups, 1))
        self.beta = nn.Parameter(torch.zeros(1, groups, 1))
        self.eps = eps

    def forward(self, x):                  # x: (B, T, C)
        b, t, c = x.shape
        g = self.groups
        xg = x.view(b, t, g, c // g).permute(0, 2, 1, 3)      # (B, G, T, Cg)
        glob = xg.mean(dim=2, keepdim=True)                   # (B, G, 1, Cg)
        norm = (xg.norm(dim=-1) * glob.norm(dim=-1)).clamp_min(self.eps)
        sim = (xg * glob).sum(dim=-1) / norm                  # (B, G, T)
        z = (sim - sim.mean(dim=-1, keepdim=True)) / (sim.std(dim=-1, keepdim=True) + self.eps)
        gate = torch.sigmoid(self.gamma * z + self.beta)      # (B, G, T)
        out = xg * gate.unsqueeze(-1)
        return out.permute(0, 2, 1, 3).reshape(b, t, c)
