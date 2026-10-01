from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .smt import smt_t


class SMTTinyBackbone(nn.Module):
    """SMT-Tiny backbone wrapper that returns four multi-scale feature maps."""

    out_channels: Sequence[int] = (64, 128, 256, 512)
    out_strides: Sequence[int] = (4, 8, 16, 32)

    def __init__(self, weight_path: Optional[str] = None, strict: bool = False):
        super().__init__()
        self.backbone = smt_t(pretrained=False)
        if weight_path:
            self.load_pretrained(weight_path, strict=strict)

    def load_pretrained(self, weight_path: str, strict: bool = False):
        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
        else:
            state = ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        return missing, unexpected

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


def build_smt_tiny(weight_path: Optional[str] = None, strict: bool = False) -> SMTTinyBackbone:
    return SMTTinyBackbone(weight_path=weight_path, strict=strict)


def load_smt_tiny(strict: bool = False) -> SMTTinyBackbone:
    weight_path = Path(__file__).resolve().parent / "weights" / "smt_tiny.pth"
    return build_smt_tiny(str(weight_path), strict=strict)
