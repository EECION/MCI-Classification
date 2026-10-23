"""Squeeze-and-excitation channel recalibration (Supplementary Methods S3.2, Eqs. S12-S13)."""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):                  # x: (B, C, H, W)
        b, c = x.shape[:2]
        s = x.mean(dim=(2, 3))             # GAP -> (B, C)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w
