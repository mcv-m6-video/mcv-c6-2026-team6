import torch
import torch.nn as nn


class LSTMHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        out, _ = self.lstm(x)   # (B, T, hidden_dim)
        return out


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = x.permute(0, 2, 1)           # (B, C, T)
        out = self.conv(out)
        out = out[:, :, :x.shape[1]]       # causal trim
        out = out.permute(0, 2, 1)         # (B, T, C)
        return self.relu(self.norm(out + residual))


class TCNHead(nn.Module):
    def __init__(self, input_dim, num_layers=3, kernel_size=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.blocks = nn.ModuleList([
            TCNBlock(input_dim, kernel_size=kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        self.out_dim = input_dim

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x
