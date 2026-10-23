"""Model registry: variant name -> constructor wiring (paper Section 2.3)."""
import torch.nn as nn

from src.models.fcn import FCNBackbone
from src.models.lstm import LSTMBackbone
from src.models.resnet import ResNetBackbone
from src.models.transformer import TransformerBackbone

VARIANTS = [
    "transformer", "transformer_cbam",
    "resnet", "resnet_aspp",
    "fcn", "fcn_se",
    "lstm", "lstm_sge",
]

# enhanced variant -> its baseline, for paired hold-out comparisons (Table 2)
BASELINE_OF = {
    "transformer_cbam": "transformer",
    "resnet_aspp": "resnet",
    "fcn_se": "fcn",
    "lstm_sge": "lstm",
}

PAIRS = [("transformer", "transformer_cbam"),
         ("resnet", "resnet_aspp"),
         ("fcn", "fcn_se"),
         ("lstm", "lstm_sge")]


def build_model(model_cfg: dict) -> nn.Module:
    variant = model_cfg.get("variant", "transformer")
    dropout = model_cfg.get("dropout", 0.1)

    if variant == "transformer":
        return TransformerBackbone(
            input_dim=90, d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 2),
            d_ff=model_cfg.get("d_ff", 256), dropout=dropout)
    if variant == "transformer_cbam":
        cbam = model_cfg.get("cbam", {})
        return TransformerBackbone(
            input_dim=90, d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 2),
            d_ff=model_cfg.get("d_ff", 256), dropout=dropout,
            use_cbam=True, cbam_reduction=cbam.get("reduction", 16),
            cbam_kernel=cbam.get("kernel_size", 5))
    if variant in ("resnet", "resnet_aspp"):
        aspp = model_cfg.get("aspp", {})
        return ResNetBackbone(
            channels=tuple(model_cfg.get("channels", (32, 64, 128))),
            use_aspp=(variant == "resnet_aspp"),
            aspp_mid=aspp.get("mid_channels", 64),
            aspp_dilations=aspp.get("dilations", (6, 12, 18, 24)))
    if variant in ("fcn", "fcn_se"):
        se = model_cfg.get("se", {})
        return FCNBackbone(
            channels=tuple(model_cfg.get("channels", (32, 64, 128))),
            dropout=dropout, use_se=(variant == "fcn_se"),
            se_reduction=se.get("reduction", 16))
    if variant in ("lstm", "lstm_sge"):
        sge = model_cfg.get("sge", {})
        return LSTMBackbone(
            input_size=90, hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 1), dropout=dropout,
            use_sge=(variant == "lstm_sge"),
            sge_groups=sge.get("groups", 8))
    raise ValueError(f"unknown variant: {variant}")
