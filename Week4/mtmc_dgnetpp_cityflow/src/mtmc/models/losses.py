from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(feats, feats, p=2)
        n = dist.size(0)
        loss = []
        for i in range(n):
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]
            pos_mask[i] = False
            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue
            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            loss.append(F.relu(hardest_pos - hardest_neg + self.margin))
        if not loss:
            return torch.tensor(0.0, device=feats.device)
        return torch.stack(loss).mean()
