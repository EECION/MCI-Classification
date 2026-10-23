"""Atrous spatial pyramid pooling (Supplementary Methods S2.2, Eqs. S8-S9).

Parallel 1x1 convolution, 3x3 atrous convolutions with dilation rates
{6, 12, 18, 24} and a global-context branch; outputs are spatially aligned
(padding = dilation), concatenated, then fused by 1x1 conv + BN + activation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, mid_channels=64, dilations=(6, 12, 18, 24)):
        super().__init__()
        self.branch_1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.dilated = nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=r, dilation=r)
            for r in dilations
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        n_branches = 2 + len(dilations)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * n_branches, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=size, mode="bilinear", align_corners=False)
        outs = [self.branch_1x1(x)] + [branch(x) for branch in self.dilated] + [g]
        return self.fuse(torch.cat(outs, dim=1))
