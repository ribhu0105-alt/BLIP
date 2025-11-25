# BLIP/models/pvt.py
# Adapted PVT implementation (inspired by "Understanding Pyramid Vision Transformer" by Övül Arslan)
# Rewritten and simplified for integration with BLIP.
#
# Attribution: implementation inspired by the referenced Medium article (Ovul Arslan)
# (https://medium.com/@ovularslan/understanding-pyramid-vision-transformer-a-review-and-pytorch-implementation-guide-a2ef2bea8ebc)

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities
# -------------------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape: (batch, 1, 1) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# -------------------------
# Patch embedding
# -------------------------
class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding with a Conv2d projection followed by LayerNorm on tokens.
    Returns tokens (B, N, C) and (H, W) spatial size.
    """
    def __init__(self, in_chans=3, embed_dim=64, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, (H, W)


# -------------------------
# Spatial Reduction Attention (SRA)
# -------------------------
class SpatialReductionAttention(nn.Module):
    """
    Multi-head Self-Attention with optional spatial reduction (SRA) before computing K and V.
    This follows PVT design that reduces token count for K,V using a lightweight conv-based reduction.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # use a conv-based reduction to produce fewer tokens for K and V
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C) where N = H*W
        returns: (B, N, C)
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

        if self.sr is not None:
            # reduce spatial resolution for K and V
            x_ = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
            x_ = self.sr(x_)  # (B, C, H//sr, W//sr)
            H_s, W_s = x_.shape[2], x_.shape[3]
            x_ = x_.flatten(2).transpose(1, 2)  # (B, N_s, C)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # shapes: k (B, heads, N_s, head_dim)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N_s or N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------
# MLP (Feed-forward)
# -------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------
# Transformer Block (with SRA)
# -------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0., drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -------------------------
# Compact Pyramid Vision Transformer (PVT) backbone
# -------------------------
class PyramidVisionTransformer(nn.Module):
    """
    Compact, practical PVT backbone returning multi-scale features.
    Configuration parameters follow (embed_dims, depths, num_heads, sr_ratios) patterns.
    """
    def __init__(self,
                 in_chans: int = 3,
                 embed_dims: Tuple[int, int, int, int] = (64, 128, 320, 512),
                 num_heads: Tuple[int, int, int, int] = (1, 2, 5, 8),
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
                 sr_ratios: Tuple[int, int, int, int] = (8, 4, 2, 1)):
        super().__init__()

        assert len(embed_dims) == 4 and len(depths) == 4 and len(num_heads) == 4 and len(sr_ratios) == 4

        # patch embeddings (overlapping conv)
        self.patch_embed1 = OverlapPatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
                                              kernel_size=7, stride=4, padding=3)
        self.patch_embed2 = OverlapPatchEmbed(in_chans=embed_dims[0], embed_dim=embed_dims[1],
                                              kernel_size=3, stride=2, padding=1)
        self.patch_embed3 = OverlapPatchEmbed(in_chans=embed_dims[1], embed_dim=embed_dims[2],
                                              kernel_size=3, stride=2, padding=1)
        self.patch_embed4 = OverlapPatchEmbed(in_chans=embed_dims[2], embed_dim=embed_dims[3],
                                              kernel_size=3, stride=2, padding=1)

        # build stages: each stage is a sequence of Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])

        # final layer norms to transform token streams back to conv-like feature maps
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def _forward_stage(self, x, patch_embed: OverlapPatchEmbed, blocks: nn.ModuleList, norm: nn.LayerNorm):
        x, (H, W) = patch_embed(x)
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (B, 3, H, W)
        returns: list of feature maps [c1, c2, c3, c4], where each is (B, C_i, H_i, W_i)
        """
        # Stage 1
        c1 = self._forward_stage(x, self.patch_embed1, self.block1, self.norm1)
        # Stage 2
        c2 = self._forward_stage(c1, self.patch_embed2, self.block2, self.norm2)
        # Stage 3
        c3 = self._forward_stage(c2, self.patch_embed3, self.block3, self.norm3)
        # Stage 4
        c4 = self._forward_stage(c3, self.patch_embed4, self.block4, self.norm4)

        return [c1, c2, c3, c4]


# convenience factory (common configs)
def pvt_tiny(**kwargs):
    return PyramidVisionTransformer(
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        **kwargs
    )


def pvt_small(**kwargs):
    return PyramidVisionTransformer(
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        **kwargs
    )


def pvt_medium(**kwargs):
    return PyramidVisionTransformer(
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        **kwargs
    )


# quick sanity test when running the file directly
if __name__ == "__main__":
    model = pvt_small()
    model.eval()
    inp = torch.randn(2, 3, 224, 224)
    feats = model(inp)
    for i, f in enumerate(feats):
        print(f"stage {i+1} shape: {f.shape}")
