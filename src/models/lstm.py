"""LSTM backbone (Supplementary Methods S4.1, Eqs. S14-S15).

The connectivity matrix is read as an ordered sequence of 90 ROI connectivity
vectors; the final hidden state feeds the classifier. Optional SGE group-wise
enhancement of the output sequence (S4.2, Eqs. S16-S17).
Input: (B, 90, 90) -> logits (B, 3).
"""
import torch
import torch.nn as nn

from src.models.modules.sge import SGE


class LSTMBackbone(nn.Module):
    def __init__(self, input_size=90, num_classes=3, hidden_size=128, num_layers=1,
                 dropout=0.0, use_sge=False, sge_groups=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.use_sge = use_sge
        if use_sge:
            self.sge = SGE(hidden_size, groups=sge_groups)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)              # (B, T=90, H)
        if self.use_sge:
            out = self.sge(out)
        return self.head(out[:, -1])
