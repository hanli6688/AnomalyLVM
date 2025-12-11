import torch
from typing import Union, List, Optional
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility Blocks
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiHeadChannelAttention(nn.Module):
    """
    Lightweight channel-wise attention (MHCA)
    for enhancing fusion stability between SAM2 and DINO-X features.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(x_flat).chunk(3, dim=-1)

        q = q.reshape(B, -1, self.num_heads, C // self.num_heads)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, -1, C)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return out


# ============================================================
# Vision Encoders: SAM2 & DINO-X (Frozen)
# ============================================================

class FrozenSAM2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SAM2.backbone  # Replace with real SAM2 visual encoder

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)     # Feature map: [B, C_sam, H, W]


class FrozenDINOX(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DINO-X.backbone  # Replace with real DINO-X encoder

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)     # Feature map: [B, C_dino, H, W]


# ============================================================
# Text Encoder + Prompt Learning (CLIP-based, Frozen)
# ============================================================

class PromptLearner(nn.Module):
    """
    Learnable text prompts, replacing handcrafted templates.
    """
    def __init__(self, embed_dim, num_prompt_tokens=8):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(num_prompt_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self):
        # Final prompt = [Prefix Tokens | CLS Token]
        return torch.cat([self.prefix, self.cls_token], dim=0)


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.text_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, prompts):
        return self.encoder(prompts)


# ============================================================
# Multi-level Fusion + Decoder (U-like Hybrid Module)
# ============================================================

class FusionDecoder(nn.Module):
    """
    Hybrid decoder combining SAM2 & DINO-X features using:
    - channel attention
    - multi-stage fusion
    - UNet-like progressive upsampling
    """
    def __init__(self, in_sam, in_dino, base_ch=256):
        super().__init__()

        self.att_sam = MultiHeadChannelAttention(in_sam)
        self.att_dino = MultiHeadChannelAttention(in_dino)

        self.proj_sam = ConvBNReLU(in_sam, base_ch)
        self.proj_dino = ConvBNReLU(in_dino, base_ch)

        self.fuse = ConvBNReLU(base_ch * 2, base_ch)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_ch, base_ch // 2, 2, stride=2)
        self.out_conv = nn.Conv2d(base_ch // 2, 1, 1)

    def forward(self, f_sam, f_dino):
        f_sam = self.att_sam(f_sam)
        f_dino = self.att_dino(f_dino)

        fs = self.proj_sam(f_sam)
        fd = self.proj_dino(f_dino)

        fused = self.fuse(torch.cat([fs, fd], dim=1))
        x = self.up1(fused)
        x = self.up2(x)
        loc = self.out_conv(x)

        return loc, fused


# ============================================================
# Feature Enhancement Module (FEM)
# ============================================================

class FeatureEnhancementModule(nn.Module):
    """
    FEM: combines static LayerCAM maps with:
    - visual attentions
    - decoder affinity
    - cross-attention fusion
    """
    def __init__(self, in_ch=3):
        super().__init__()

        self.cross_att = nn.MultiheadAttention(embed_dim=128, num_heads=4)

        self.spatial_proj = nn.Sequential(
            ConvBNReLU(in_ch, 64),
            ConvBNReLU(64, 32),
            nn.Conv2d(32, 1, 1)
        )

        self.adaptive_weight = nn.Parameter(torch.ones(3))

    def forward(self, layercam, visual_attn, decoder_attn):
        B, _, H, W = layercam.shape

        x = torch.cat([layercam, visual_attn, decoder_attn], dim=1)

        flat = x.flatten(2).transpose(1, 2)
        q = k = v = flat

        out, _ = self.cross_att(q, k, v)
        out = out.transpose(1, 2).reshape(B, -1, H, W)

        weights = F.softmax(self.adaptive_weight, dim=0)
        combined = weights[0] * layercam + weights[1] * visual_attn + weights[2] * decoder_attn

        refined = self.spatial_proj(combined + out.mean(dim=1, keepdim=True))
        refined = torch.sigmoid(refined)

        return refined


# ============================================================
# Full Model: AnomalyLVM
# ============================================================

class AnomalyLVM(nn.Module):
    def __init__(self, clip_model, sam_ch=1024, dino_ch=1024):
        super().__init__()

        self.sam2 = FrozenSAM2()
        self.dinox = FrozenDINOX()

        embed_dim = clip_model.text_projection.shape[1]
        self.prompt_learner = PromptLearner(embed_dim=embed_dim)
        self.text_encoder = FrozenCLIPTextEncoder(clip_model)

        self.decoder = FusionDecoder(in_sam=sam_ch, in_dino=dino_ch)
        self.fem = FeatureEnhancementModule()

    def forward(self, img):

        # visual features
        f_sam = self.sam2(img)
        f_dino = self.dinox(img)

        # pixel-level localization via decoder
        loc_map, fused_feat = self.decoder(f_sam, f_dino)

        visual_attn = fused_feat.mean(dim=1, keepdim=True)
        decoder_attn = loc_map.mean(dim=1, keepdim=True)

        layercam = torch.rand_like(visual_attn)  # placeholder for real LayerCAM

        pseudo = self.fem(layercam, visual_attn, decoder_attn)

        prompts = self.prompt_learner()
        text_feat = self.text_encoder(prompts)

        img_feat = fused_feat.mean(dim=[2, 3])
        logits = img_feat @ text_feat.T

        return {
            "loc_map": loc_map,
            "pseudo_label": pseudo,
            "logits": logits,
            "fusion_feat": fused_feat
        }
