"""Fully convolutional backbone (Supplementary Methods S3.1, Eqs. S10-S11).

Stacked 3x3 convolution layers (conv + BN + activation) followed by global
average pooling. Optional SE channel recalibration on the final feature map
(S3.2, Eqs. S12-S13). Input: (B, 90, 90) -> logits (B, 3).
"""
import torch
import torch.nn as nn

from src.models.modules.se import SEBlock


class FCNBackbone(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, channels=(32, 64, 128),
                 dropout=0.0, use_se=False, se_reduction=16):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, kernel_size=3, padding=1),
                       nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout)]
            prev = ch
        self.backbone = nn.Sequential(*layers)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels[-1], reduction=se_reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.backbone(x)
        if self.use_se:
            h = self.se(h)
        return self.head(self.gap(h).flatten(1))
