import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvtv2 import pvt_v2_b2
from .rafbnet_blocks import (
    BasicConv2d,
    DirectionalConvUnit,
    KTM,
    PDecoderWithFeature,
    SWSAM,
)


class RadioReliabilityGate(nn.Module):
    """Residual reliability gate driven by frozen RADIO dense features."""

    def __init__(self, channels=32, radio_channels=768, gate_alpha=1.0):
        super().__init__()
        self.gate_alpha = gate_alpha
        self.radio_proj = nn.Sequential(
            nn.Conv2d(radio_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, radio_feat):
        radio_feat = F.interpolate(radio_feat, size=feat.shape[2:], mode="bilinear", align_corners=True)
        radio_feat = self.radio_proj(radio_feat)
        gate = self.gate(torch.cat([feat, radio_feat], dim=1))
        return feat * (1.0 + self.gate_alpha * gate)


class ForegroundBackgroundRefinement(nn.Module):
    """Foreground/background complementary refinement for decoder features."""

    def __init__(self, channels=96):
        super().__init__()
        self.fg_branch = nn.Sequential(
            BasicConv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.bg_branch = nn.Sequential(
            BasicConv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(BasicConv2d(channels * 2, channels, 3, padding=1), nn.Conv2d(channels, 1, 1))

    def forward(self, decoder_feat, coarse_pred):
        sal = torch.sigmoid(coarse_pred)
        fg_feat = self.fg_branch(decoder_feat * sal)
        bg_feat = self.bg_branch(decoder_feat * (1.0 - sal))
        residual = self.fuse(torch.cat([fg_feat, bg_feat], dim=1))
        return coarse_pred + residual


class RaFBNet(nn.Module):
    """Reliability-aware Foreground-Background Network for ORSI-SOD."""

    def __init__(
        self,
        channel=32,
        pvt_checkpoint="weights/pvt_v2_b2.pth",
        radio_repo=None,
        radio_checkpoint=None,
        use_radio=True,
        radio_gate_alpha=1.0,
    ):
        super().__init__()

        self.backbone = pvt_v2_b2()
        if pvt_checkpoint:
            save_model = torch.load(pvt_checkpoint, map_location="cpu")
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)
        self.ChannelNormalization_2 = BasicConv2d(128, channel, 3, 2, 1)
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1)
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1)
        self.SWSAM_4 = SWSAM(channel)
        self.dirConv = DirectionalConvUnit(channel)
        self.DSWSAM_1 = SWSAM(channel)
        self.KTM_23 = KTM(channel)
        self.PDecoder = PDecoderWithFeature(channel)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.fgbg_refine = ForegroundBackgroundRefinement(channels=channel * 3)

        self.use_radio = use_radio
        self.radio = None
        if use_radio:
            if not radio_repo or not radio_checkpoint:
                raise ValueError("radio_repo and radio_checkpoint are required when use_radio=True")
            self.radio = self._load_radio(radio_repo, radio_checkpoint)
            for param in self.radio.parameters():
                param.requires_grad = False
            self.radio.eval()
            self.radio_gate_2 = RadioReliabilityGate(channel, gate_alpha=radio_gate_alpha)
            self.radio_gate_3 = RadioReliabilityGate(channel, gate_alpha=radio_gate_alpha)
            self.radio_gate_4 = RadioReliabilityGate(channel, gate_alpha=radio_gate_alpha)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    def _load_radio(self, radio_repo, radio_checkpoint):
        radio_repo = str(Path(radio_repo).resolve())
        if radio_repo not in sys.path:
            sys.path.insert(0, radio_repo)
        from hubconf import radio_model

        if not os.path.isfile(radio_checkpoint):
            raise FileNotFoundError(f"RADIO checkpoint not found: {radio_checkpoint}")
        return radio_model(version=radio_checkpoint, progress=False)

    def _radio_features(self, x):
        radio_x = (x * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
        with torch.no_grad():
            self.radio.eval()
            _, radio_feat = self.radio(radio_x, feature_fmt="NCHW")
        return radio_feat.float()

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x1_nor = self.ChannelNormalization_1(x1)
        x2_nor = self.ChannelNormalization_2(x2)
        x3_nor = self.ChannelNormalization_3(x3)
        x4_nor = self.ChannelNormalization_4(x4)

        if self.use_radio and self.radio is not None:
            radio_feat = self._radio_features(x)
            x2_nor = self.radio_gate_2(x2_nor, radio_feat)
            x3_nor = self.radio_gate_3(x3_nor, radio_feat)
            x4_nor = self.radio_gate_4(x4_nor, radio_feat)

        x4_swsam = self.SWSAM_4(x4_nor)
        x1_ori = self.dirConv(x1_nor)
        x1_swsam = self.DSWSAM_1(x1_ori)
        x23_ktm = self.KTM_23(x2_nor, x3_nor)
        coarse_88, decoder_feat = self.PDecoder(x4_swsam, x23_ktm, x1_swsam)
        refined_88 = self.fgbg_refine(decoder_feat, coarse_88)
        prediction = self.upsample_4(refined_88)
        return prediction, self.sigmoid(prediction)
