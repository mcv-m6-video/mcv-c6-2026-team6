"""
model/model_spotting.py
-----------------------------
Extended from W6 with:
  - X3D-M backbone via PyTorchVideo.
  - UNet-like temporal head.
  - Soft-label (KL-divergence) loss for Temporal Gaussian Label Smoothing.
"""

import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from model.modules import BaseRGBModel, FCLayers, step
from model.bigru_head import BiGRUHead
from model.temporal_heads import LSTMHead, TCNHead
from model.unet_head import UNetTemporalHead


# ── Loss functions ───────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=-1)
        prob     = torch.exp(log_prob)
        ce       = F.nll_loss(log_prob, targets, weight=self.weight, reduction='none')
        p_t      = prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


class SoftCrossEntropy(nn.Module):
    """
    KL-divergence style loss for soft targets (needed by TGLS, W7-3A).

    logits:       (N, C)  raw outputs of the FC head.
    soft_targets: (N, C)  probability distribution per frame (rows sum to 1).

    Class weights (if provided) are broadcast per class and applied as a
    per-sample weight equal to sum_c w_c * p_c, so positive classes still
    get extra weight — same idea as wCE adapted to soft labels.
    """

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight  # (C,)

    def forward(self, logits, soft_targets):
        log_prob = F.log_softmax(logits, dim=-1)                 # (N, C)
        if self.weight is not None:
            sample_w = (soft_targets * self.weight).sum(dim=-1)  # (N,)
            loss = -(soft_targets * log_prob).sum(dim=-1)        # (N,)
            return (loss * sample_w).mean()
        return -(soft_targets * log_prob).sum(dim=-1).mean()


# ── X3D-M backbone wrapper ───────────────────────────────────────────────────
class _X3DBackbone(nn.Module):
    """
    Wrapper around PyTorchVideo's X3D (M / S / XS) returning per-frame
    embeddings of shape (B, T, D).

    We drop the classification head and use spatial-only adaptive pooling
    so the temporal dimension is preserved (the UNet head will reduce it).
    """

    def __init__(self, variant: str = 'x3d_m', pretrained: bool = True):
        super().__init__()
        try:
            from pytorchvideo.models.hub import x3d_m, x3d_s, x3d_xs
        except ImportError as e:
            raise ImportError(
                'pytorchvideo is required for X3D backbones. '
                'Install with: pip install pytorchvideo'
            ) from e

        factory = {'x3d_m': x3d_m, 'x3d_s': x3d_s, 'x3d_xs': x3d_xs}[variant]
        net = factory(pretrained=pretrained)

        # Drop the classification/pooling head (last block)
        self.blocks = nn.ModuleList(list(net.blocks[:-1]))

        # Infer output channels from the last Conv3d layer
        out_ch = None
        for m in reversed(list(self.blocks[-1].modules())):
            if isinstance(m, nn.Conv3d):
                out_ch = m.out_channels
                break
        assert out_ch is not None, 'Could not infer X3D output channels'
        self.out_dim = out_ch

        # Pool over H, W but keep T intact
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        for blk in self.blocks:
            x = blk(x)
        x = self.spatial_pool(x)                  # (B, C', T', 1, 1)
        x = x.squeeze(-1).squeeze(-1)             # (B, C', T')
        x = x.permute(0, 2, 1).contiguous()       # (B, T', C')

        # X3D has temporal stride in the stem -> T' < T. Interpolate back.
        if x.shape[1] != T:
            x = x.transpose(1, 2)
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)
            x = x.transpose(1, 2)
        return x


# ── Main Model ───────────────────────────────────────────────────────────────
class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch  = args.feature_arch
            self._temporal_arch = getattr(args, 'temporal_arch', 'none')
            self._is_3d        = self._feature_arch.startswith('x3d')

            # ── Backbone ──────────────────────────────────────────────────
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim
                self._features = features
            elif self._feature_arch.startswith('x3d'):
                self._features = _X3DBackbone(variant=self._feature_arch, pretrained=True)
                self._d = self._features.out_dim
            else:
                raise NotImplementedError(args.feature_arch)

            # ── Temporal head ─────────────────────────────────────────────
            ta = self._temporal_arch
            if ta == 'none':
                self._temporal = None
                head_dim = self._d
            elif ta == 'bigru':
                hidden = getattr(args, 'gru_hidden_dim', 512)
                layers = getattr(args, 'gru_num_layers', 2)
                drop   = getattr(args, 'gru_dropout', 0.1)
                self._temporal = BiGRUHead(
                    self._d, hidden_dim=hidden,
                    num_layers=layers, dropout=drop)
                head_dim = self._temporal.out_dim
            elif ta == 'lstm':
                hidden = getattr(args, 'lstm_hidden_dim', 512)
                layers = getattr(args, 'lstm_num_layers', 1)
                self._temporal = LSTMHead(self._d, hidden_dim=hidden, num_layers=layers)
                head_dim = self._temporal.out_dim
            elif ta == 'tcn':
                n_layers = getattr(args, 'tcn_num_layers', 3)
                k_size   = getattr(args, 'tcn_kernel_size', 3)
                self._temporal = TCNHead(self._d, num_layers=n_layers, kernel_size=k_size)
                head_dim = self._temporal.out_dim
            elif ta == 'unet':
                depth     = getattr(args, 'unet_depth', 2)
                hidden    = getattr(args, 'unet_hidden_dim', None)
                btl       = getattr(args, 'unet_bottleneck', 'bigru')
                gru_h     = getattr(args, 'unet_gru_hidden', 256)
                gru_l     = getattr(args, 'unet_gru_layers', 2)
                drop      = getattr(args, 'unet_dropout', 0.1)
                self._temporal = UNetTemporalHead(
                    self._d, depth=depth, hidden_dim=hidden,
                    bottleneck=btl, gru_hidden=gru_h, gru_layers=gru_l,
                    dropout=drop)
                head_dim = self._temporal.out_dim
            else:
                raise NotImplementedError(f'Unknown temporal_arch: {ta}')

            # ── Classification head ───────────────────────────────────────
            self._fc = FCLayers(head_dim, args.num_classes + 1)

            # ── Augmentations ─────────────────────────────────────────────
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])
            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)
            B, Tt, C, H, W = x.shape
            if self.training:
                x = self.augment(x)
            x = self.standarize(x)

            if self._is_3d:
                feat = self._features(x)                       # (B, T, D)
            else:
                feat = self._features(
                    x.view(-1, C, H, W)
                ).reshape(B, Tt, self._d)

            if self._temporal is not None:
                feat = self._temporal(feat)
            return self._fc(feat)

        def normalize(self, x):  return x / 255.
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x
        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x
        def print_stats(self):
            total = sum(p.numel() for p in self.parameters())
            print(f'Model params: {total:,}')

    # --------------------------------------------------------------------
    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and getattr(args, 'device', 'cpu') == 'cuda':
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._num_classes = args.num_classes

        weights = torch.tensor(
            [1.0] + [5.0] * self._num_classes, dtype=torch.float32
        ).to(self.device)

        loss_type = getattr(args, 'loss', 'ce')
        self._soft_criterion = None
        if loss_type == 'focal':
            gamma = getattr(args, 'focal_gamma', 2.0)
            self._criterion = FocalLoss(gamma=gamma, weight=weights)
            print(f'Loss: Focal (gamma={gamma})')
        elif loss_type == 'soft_ce':
            self._criterion = None
            self._soft_criterion = SoftCrossEntropy(weight=weights)
            print('Loss: Soft Cross-Entropy (TGLS soft labels)')
        else:
            self._criterion = None
            print('Loss: Cross-Entropy')

        self._weights   = weights
        self._loss_type = loss_type
        self._model.to(self.device)

    # --------------------------------------------------------------------
    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame).view(-1, self._num_classes + 1)

                    if self._loss_type == 'soft_ce':
                        soft = batch['soft_label'].to(self.device).float()
                        soft = soft.view(-1, self._num_classes + 1)
                        loss = self._soft_criterion(pred, soft)
                    else:
                        label = batch['label'].to(self.device).long().view(-1)
                        if self._loss_type == 'focal':
                            loss = self._criterion(pred, label)
                        else:
                            loss = F.cross_entropy(pred, label, reduction='mean',
                                                   weight=self._weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()
        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)
            return torch.softmax(pred, dim=-1).cpu().numpy()
