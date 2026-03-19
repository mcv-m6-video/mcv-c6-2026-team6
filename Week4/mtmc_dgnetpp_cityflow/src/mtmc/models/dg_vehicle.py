from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


def _build_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    name = backbone_name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    if name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        return resnet34(weights=weights)
    raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose from: resnet18, resnet34")


class DGVehicleNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        id_dim: int = 256,
        style_dim: int = 64,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        backbone = _build_backbone(backbone_name=backbone_name, pretrained=pretrained_backbone)
        self.stem = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = 512

        self.backbone_name = backbone_name.lower()
        self.pretrained_backbone = pretrained_backbone

        self.id_head = nn.Sequential(
            nn.Linear(feat_dim, id_dim),
            nn.BatchNorm1d(id_dim),
            nn.ReLU(inplace=True),
        )
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, style_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(id_dim, num_classes)

        # Lightweight decoder to encourage identity/style disentanglement.
        self.decoder = nn.Sequential(
            nn.Linear(id_dim + style_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3 * 64 * 64),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        f = self.stem(x).flatten(1)
        id_feat = self.id_head(f)
        style_feat = self.style_head(f)
        logits = self.classifier(id_feat)

        fused = torch.cat([id_feat, style_feat], dim=1)
        recon = self.decoder(fused).view(-1, 3, 64, 64)

        return {
            "id_feat": F.normalize(id_feat, dim=1),
            "style_feat": style_feat,
            "logits": logits,
            "recon": recon,
        }
