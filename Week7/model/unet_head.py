"""
model/unet_head.py
------------------
UNet-like temporal decoder.

Reduces the number of temporal embeddings L -> L' (bottleneck) and recovers
L via upsampling + skip connections

Input:  (B, L, D) frame embeddings coming from the video encoder.
Output: (B, L, D) per-frame embeddings ready for FC + Softmax.

Design notes
============
- The "temporal decoder" in the slide lives ABOVE the bottleneck.
  Here we implement a symmetric encoder-decoder so the bottleneck L' is
  explicit (required by W7: "intermediate reduction of temporal dimension").
- Down-sampling is done with stride-2 1D convolutions (learnable pooling,
  gives better gradient flow than average pooling).
- Up-sampling is done with nearest interpolation + a 1D conv (cheaper and
  more stable than ConvTranspose1d which tends to create checkerboard
  artefacts on temporal signals).
- At the bottleneck we plug a sequence model (BiGRU or Transformer) so the
  network still sees long-range dependencies at the coarsest resolution.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Small building blocks
# ──────────────────────────────────────────────────────────────────────────
class _TempConvBlock(nn.Module):
    """1D conv + LayerNorm + GELU, applied on the time dimension."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=pad)
        self.norm = nn.GroupNorm(1, out_ch)  # equivalent to LayerNorm over channels
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.act(self.norm(self.conv(x)))


class _BottleneckBiGRU(nn.Module):
    """BiGRU acting at L' resolution, projected back to D."""

    def __init__(self, dim: int, hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T', D)
        out, _ = self.gru(x)
        return self.norm(x + self.proj(out))  # residual for stable training


# ──────────────────────────────────────────────────────────────────────────
# Main module
# ──────────────────────────────────────────────────────────────────────────
class UNetTemporalHead(nn.Module):
    """
    Temporal UNet on a sequence of frame embeddings.

    Args:
        input_dim:      D, dimensionality of the frame embeddings.
        depth:          number of down-sampling stages (1 or 2 tested).
                         depth=1 -> L -> L/2 -> L
                         depth=2 -> L -> L/2 -> L/4 -> L/2 -> L
        hidden_dim:     channel dim kept through the U (defaults to input_dim).
        bottleneck:     'bigru' | 'conv' | 'none'.
        gru_hidden:     hidden size of the bottleneck BiGRU.
        gru_layers:     layers of the bottleneck BiGRU.
        dropout:        dropout in the bottleneck.

    Output dim = input_dim (we keep embeddings compatible with the FC head).
    """

    def __init__(self,
                 input_dim: int,
                 depth: int = 2,
                 hidden_dim: Optional[int] = None,
                 bottleneck: str = 'bigru',
                 gru_hidden: int = 256,
                 gru_layers: int = 2,
                 dropout: float = 0.1,
                 kernel_size: int = 3):
        super().__init__()
        assert depth in (1, 2), "depth must be 1 or 2"
        assert bottleneck in ('bigru', 'conv', 'none')

        self.depth      = depth
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim or input_dim

        # Input projection (D -> hidden_dim) if needed
        if self.hidden_dim != input_dim:
            self.in_proj = nn.Linear(input_dim, self.hidden_dim)
        else:
            self.in_proj = nn.Identity()

        # ── Encoder (down) ────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for _ in range(depth):
            # conv at current resolution (keeps T)
            self.enc_blocks.append(_TempConvBlock(
                self.hidden_dim, self.hidden_dim, kernel_size, stride=1))
            # strided conv -> T/2
            self.down_blocks.append(_TempConvBlock(
                self.hidden_dim, self.hidden_dim, kernel_size, stride=2))

        # ── Bottleneck ────────────────────────────────────────────────────
        if bottleneck == 'bigru':
            self.bottleneck = _BottleneckBiGRU(
                self.hidden_dim, hidden=gru_hidden,
                num_layers=gru_layers, dropout=dropout)
        elif bottleneck == 'conv':
            self.bottleneck = nn.Sequential(
                _TempConvBlock(self.hidden_dim, self.hidden_dim, kernel_size, 1),
                _TempConvBlock(self.hidden_dim, self.hidden_dim, kernel_size, 1),
            )
        else:
            self.bottleneck = nn.Identity()
        self._bottleneck_type = bottleneck

        # ── Decoder (up) ──────────────────────────────────────────────────
        self.up_blocks  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            # merge skip + upsampled: 2*hidden_dim -> hidden_dim
            self.up_blocks.append(_TempConvBlock(
                2 * self.hidden_dim, self.hidden_dim, kernel_size, stride=1))
            self.dec_blocks.append(_TempConvBlock(
                self.hidden_dim, self.hidden_dim, kernel_size, stride=1))

        # Output projection back to input_dim
        if self.hidden_dim != input_dim:
            self.out_proj = nn.Linear(self.hidden_dim, input_dim)
        else:
            self.out_proj = nn.Identity()

        self.out_dim = input_dim

    # -----------------------------------------------------------------
    def _upsample_to(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: (B, C, T)
        return F.interpolate(x, size=target_len, mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)  ->  (B, L, D)
        """
        x = self.in_proj(x)                      # (B, L, H)
        x = x.transpose(1, 2).contiguous()       # (B, H, L)

        # Encoder: keep skips at each level
        skips = []
        for enc, down in zip(self.enc_blocks, self.down_blocks):
            x = enc(x)
            skips.append(x)                       # skip at current resolution
            x = down(x)                           # stride 2

        # Bottleneck (operates on (B, T', H))
        if self._bottleneck_type == 'bigru':
            x = x.transpose(1, 2)                 # (B, T', H)
            x = self.bottleneck(x)
            x = x.transpose(1, 2)                 # (B, H, T')
        else:
            x = self.bottleneck(x)

        # Decoder: interpolate to skip length, concat, conv
        for up_block, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            x = self._upsample_to(x, skip.shape[-1])
            x = torch.cat([x, skip], dim=1)       # (B, 2H, T)
            x = up_block(x)                       # (B, H, T)
            x = dec(x)

        x = x.transpose(1, 2).contiguous()        # (B, L, H)
        x = self.out_proj(x)                      # (B, L, D)
        return x
