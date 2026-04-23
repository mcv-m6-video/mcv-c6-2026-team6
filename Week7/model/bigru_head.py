"""
model/bigru_head.py
--------------------
BiGRU temporal head to plug into model_spotting.py.

Add this file to your model/ directory, then in model_spotting.py
add "bigru" as a case in the temporal_arch if/elif block:

    elif self._temporal_arch == 'bigru':
        from model.bigru_head import BiGRUHead
        hidden = getattr(args, 'gru_hidden_dim', 512)
        layers = getattr(args, 'gru_num_layers', 1)
        drop   = getattr(args, 'gru_dropout', 0.1)
        self._temporal = BiGRUHead(self._d, hidden_dim=hidden,
                                    num_layers=layers, dropout=drop)
        head_dim = self._temporal.out_dim
"""

import torch
import torch.nn as nn


class BiGRUHead(nn.Module):
    """
    Bidirectional GRU applied over the time dimension.

    Input:  (B, T, D)   — batch of clips, T frame embeddings of dim D
    Output: (B, T, H)   — H = hidden_dim (both directions concatenated → hidden_dim)

    The output dim equals hidden_dim (NOT 2*hidden_dim) because we project
    the concatenation back to hidden_dim with a linear layer.
    This keeps the FC head size predictable regardless of bidirectionality.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            # dropout only between layers, not after last layer
        )

        # Project 2*hidden_dim → hidden_dim so FC head is always the same size
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.gru(x)          # (B, T, 2*hidden_dim)
        out = self.proj(out)           # (B, T, hidden_dim)
        return out
