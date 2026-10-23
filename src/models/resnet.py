"""ResNet backbone on the 90x90 connectivity image (Supplementary Methods S2.1, Eqs. S6-S7).

Residual blocks with 3x3 convolutions; a 1x1 projection shortcut is used when
dimensions change. Optional ASPP enhancement after the residual stages (S2.2).
Input: (B, 90, 90) -> logits (B, 3).
"""
import torch
import torch.nn as nn

from src.models.modules.aspp import ASPP


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.res = nn.Sequential(
            conv3x3(in_ch, out_ch, stride), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch), nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.res(x) + self.shortcut(x))


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, channels=(32, 64, 128),
                 use_aspp=False, aspp_mid=64, aspp_dilations=(6, 12, 18, 24)):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Sequential(conv3x3(in_channels, c1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.stage1 = ResBlock(c1, c1, stride=1)
        self.stage2 = ResBlock(c1, c2, stride=2)
        self.stage3 = ResBlock(c2, c3, stride=2)
        self.use_aspp = use_aspp
        if use_aspp:
            self.aspp = ASPP(c3, mid_channels=aspp_mid, dilations=tuple(aspp_dilations))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.stage3(self.stage2(self.stage1(self.stem(x))))
        if self.use_aspp:
            h = self.aspp(h)
        return self.head(self.gap(h).flatten(1))
