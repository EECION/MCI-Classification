"""Transformer backbone (Supplementary Methods S1.1-S1.2, Eqs. S1-S3).

Each connectivity-matrix row is one ROI token; tokens are linearly projected,
augmented with learnable positional encoding and processed by standard encoder
layers (MHA + FFN + LayerNorm). Optional CBAM enhancement (S1.3).
Input: (B, 90, 90) -> logits (B, 3).
"""
import torch
import torch.nn as nn

from src.models.modules.cbam import CBAM1D


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim=90, num_classes=3, d_model=128, n_heads=8,
                 n_layers=2, d_ff=256, dropout=0.1,
                 use_cbam=False, cbam_reduction=16, cbam_kernel=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM1D(d_model, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embedding[:, :x.size(1)]
        h = self.encoder(h)
        if self.use_cbam:
            h = self.cbam(h)
        pooled = self.norm(h).mean(dim=1)
        return self.head(pooled)
