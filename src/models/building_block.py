import torch
import einops

from torch import nn
from torch.nn import functional as F
from src.models.haar_dwt import HaarForward, HaarInverse


class TotalBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        b = x.size(0)
        c = x.size(1)
        h = x.size(2)
        x = einops.rearrange(x, "b c h w -> (b c h w) 1 1 1")
        x = super().forward(x)
        x = einops.rearrange(x, "(b c h w) 1 1 1 -> b c h w", b=b, c=c, h=h)
        return x

class GlobalBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        b = x.size(0)
        x = einops.rearrange(x, "b c h w -> (b c) 1 h w")
        x = super().forward(x)
        x = einops.rearrange(x, "(b c) 1 h w -> b c h w", b=b)
        return x

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DWTPatcher(nn.Module):
    def __init__(self, levels=2):
        super().__init__()
        self.levels = levels
        self.dwt = HaarForward(beta=1)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        all_chunks = []
        for i in range(self.levels):
            x = self.dwt(x)
            if i < self.levels-1:
                chunks = x.chunk(4, dim=1)
                x = chunks[0]
                all_chunks += chunks[1:]
                all_chunks = [self.unshuffle(c) for c in all_chunks]
        return torch.cat([x, *all_chunks], dim=1)


class DWTUnpatcher(nn.Module):
    def __init__(self, levels=2):
        super().__init__()
        self.levels = levels
        self.idwt = HaarInverse(beta=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        all_chunks = x.chunk(4**self.levels, dim=1)
        for i in range(self.levels):
            x = torch.cat(all_chunks[:4], dim=1)
            x = self.idwt(x)
            if i < self.levels-1:
                all_chunks = [torch.cat(all_chunks[c:c+4], dim=1) for c in range(4, len(all_chunks), 4)]
                all_chunks = [x] + [self.shuffle(c) for c in all_chunks]
        return x


class SimpleBlock(nn.Module):
    def __init__(self, c, attention_heads=None, dropout=0.0, reflection_pad=False):
        super().__init__()
        # ResBlock
        self.reflection_pad = reflection_pad
        if self.reflection_pad is True:
            self.pad = nn.ReflectionPad2d(1)
            self.depthwise = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, groups=c, bias=False),
                LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
            )
        else:
            self.depthwise = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
            )
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c)
        )

        # Attn
        self.mha = None
        if attention_heads is not None:
            self.mha_norm = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
            self.mha = nn.MultiheadAttention(c, attention_heads, bias=True, batch_first=True, dropout=dropout)

    def forward(self, x):
        # Residual Block
        x_res = x
        if self.reflection_pad is True:
            x = self.depthwise(self.pad(x))
        else:
            x = self.depthwise(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.channelwise(x)
        x = einops.rearrange(x, "b h w c -> b c h w")
        x = x + x_res
        # Attn
        if self.mha is not None:
            x_res = x
            x = einops.rearrange(x, "b c h w -> b (h w) c")
            x = self.mha_norm(x)
            x = self.mha(x, x, x, need_weights=False)[0]
            x = einops.rearrange(x, "b (h w) c -> b c h w", h=x_res.size(-2), w=x_res.size(-1))
            x = x + x_res
        return x
