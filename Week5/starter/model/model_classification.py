"""
model_classification.py  —  upgraded for MCV C6 Project 2 / Task 1 BAC

New vs baseline
───────────────
feature_arch options added
  rny002 / rny004 / rny008          original backbone family (unchanged)
  rny002_gsf / rny004_gsf / rny008_gsf  original aliases (unchanged)
  hiera_base                         Meta Hiera-Base (768-d) ← NEW

temporal_arch options  (set in config JSON)
  maxpool   original behaviour (no temporal module)
  gru       bidirectional GRU  (2 layers)     ← NEW  ★ highest-impact
  transformer  2-layer TransformerEncoder      ← NEW
  attention    learned attention pooling       ← NEW

loss options  (loss_type in config JSON)
  bce         plain BCE              (original)
  focal       Focal Loss γ=2 α=0.25           ← NEW  ★ high-impact
  weighted    frequency-weighted BCE           ← NEW

Other additions
  - CenterCrop(224) / RandomResizedCrop(224) in train  ← NEW
  - Temporal jitter + frame dropout augmentation        ← NEW
  - Label smoothing (smooth=0.05, only with bce/weighted)  ← NEW
  - Differential LR in get_optimizer                    ← NEW
  - Test-Time Augmentation (TTA) in predict             ← NEW
  - Gradient checkpointing flag                         ← NEW
"""

# Standard imports
import random
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Local imports
from model.modules import BaseRGBModel, FCLayers, step


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary Focal Loss for multi-label imbalanced classification."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Temporal modules
# ──────────────────────────────────────────────────────────────────────────────

class GRUTemporalEncoder(nn.Module):
    """Bidirectional GRU over frame features — replaces max-pool."""
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            d_model, d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = d_model * 2  # bidirectional doubles dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.gru(x)        # (B, T, 2D)
        return out.mean(dim=1)       # (B, 2D)


class TransformerTemporalEncoder(nn.Module):
    """Lightweight Transformer encoder over frame features."""
    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        # Ensure d_model is divisible by nhead
        while d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out = self.encoder(x)        # (B, T, D)
        return out.mean(dim=1)       # (B, D)


class AttentionPooling(nn.Module):
    """Single-head attention pooling — the model learns which frames matter."""
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        w = torch.softmax(self.score(x), dim=1)   # (B, T, 1)
        return (w * x).sum(dim=1)                  # (B, D)


class MaxPool(nn.Module):
    """Original baseline: max-pool across time."""
    def __init__(self, d_model: int):
        super().__init__()
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=1)[0]   # (B, D)


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch  = args.feature_arch
            self._temporal_arch = getattr(args, 'temporal_arch', 'maxpool')
            self._loss_type     = getattr(args, 'loss_type',     'bce')
            self._use_tta       = getattr(args, 'use_tta',       False)
            self._label_smooth  = getattr(args, 'label_smooth',  0.0)
            self._use_grad_ckpt = getattr(args, 'grad_checkpoint', False)

            # ── Backbone ──────────────────────────────────────────────────────
            arch_key = self._feature_arch.rsplit('_', 1)[0] \
                if '_gsf' in self._feature_arch else self._feature_arch

            if arch_key in ('rny002', 'rny004', 'rny008'):
                timm_name = {
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[arch_key]
                features = timm.create_model(timm_name, pretrained=True)
                self._d = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._features = features

            elif self._feature_arch == 'hiera_base':
                # Meta Hiera-Base (768-d) integrated into BAC pipeline
                from model.hiera_vision_encoder import (
                    HieraEncoderConfig, HieraVisionEncoder)
                cfg = HieraEncoderConfig(
                    hiera_hub_id='facebook/hiera_base_224.mae_in1k_ft_in1k')
                self._features = HieraVisionEncoder(cfg)
                self._d = self._features.config.hidden_size   # 768

            else:
                raise NotImplementedError(
                    f'Unknown feature_arch: {self._feature_arch}')

            # ── Temporal encoder ──────────────────────────────────────────────
            if self._temporal_arch == 'gru':
                self._temporal = GRUTemporalEncoder(self._d)
            elif self._temporal_arch == 'transformer':
                self._temporal = TransformerTemporalEncoder(self._d)
            elif self._temporal_arch == 'attention':
                self._temporal = AttentionPooling(self._d)
            elif self._temporal_arch == 'maxpool':
                self._temporal = MaxPool(self._d)
            else:
                raise NotImplementedError(
                    f'Unknown temporal_arch: {self._temporal_arch}')

            temporal_out_dim = self._temporal.out_dim

            # ── Classification head ───────────────────────────────────────────
            self._fc = FCLayers(temporal_out_dim, args.num_classes)

            # ── Loss ──────────────────────────────────────────────────────────
            if self._loss_type == 'focal':
                gamma = getattr(args, 'focal_gamma', 2.0)
                alpha = getattr(args, 'focal_alpha', 0.25)
                self._loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
            else:
                self._loss_fn = None   # BCE computed in epoch()

            # ── Augmentations ─────────────────────────────────────────────────
            # Training: RandomResizedCrop gives spatial augmentation AND fixes
            # the 398×224 → 224×224 resolution mismatch for Hiera/ViT backbones
            self.augmentation = T.Compose([
                T.RandomResizedCrop(224, scale=(0.6, 1.0),
                                    interpolation=T.InterpolationMode.BILINEAR),
                T.RandomApply([T.ColorJitter(hue=0.2)],              p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))],   p=0.25),
                T.RandomApply([T.GaussianBlur(5)],                    p=0.25),
                T.RandomHorizontalFlip(),
            ])
            # Validation / test: deterministic center crop only
            self.val_crop = T.CenterCrop(224)

            self.standarization = T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))

        # ── Forward ──────────────────────────────────────────────────────────

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.normalize(x)
            B, T, C, H, W = x.shape

            if self.training:
                x = self._augment_train(x)
            else:
                x = self._augment_val(x)

            x = self._standarize(x)

            # Extract per-frame features
            x_flat = x.view(-1, C, 224, 224)   # always 224 after crop

            if self._feature_arch == 'hiera_base':
                if self._use_grad_ckpt and self.training:
                    feat = grad_checkpoint(
                        lambda v: self._features(pixel_values=v).last_hidden_state,
                        x_flat)
                else:
                    feat = self._features(pixel_values=x_flat).last_hidden_state
                feat = feat.mean(dim=1)          # spatial pool: (B*T, D)
            else:
                if self._use_grad_ckpt and self.training:
                    feat = grad_checkpoint(self._features, x_flat)
                else:
                    feat = self._features(x_flat)

            im_feat = feat.reshape(B, T, self._d)   # (B, T, D)

            # Temporal encoding → (B, D_out)
            im_feat = self._temporal(im_feat)

            # Classification head
            return self._fc(im_feat)

        # ── Augmentation helpers ──────────────────────────────────────────────

        def _augment_train(self, x: torch.Tensor) -> torch.Tensor:
            """Spatial + temporal augmentations (training only)."""
            B, T, C, H, W = x.shape
            # Spatial augment per clip — build new tensor to handle resolution
            # change (e.g. 398×224 → 224×224 from RandomResizedCrop)
            clips = []
            for i in range(B):
                clips.append(self.augmentation(x[i]))   # (T, C, 224, 224)
            x = torch.stack(clips, dim=0)               # (B, T, C, 224, 224)

            # Temporal jitter: roll clip along time axis
            if random.random() < 0.5:
                shift = random.randint(-3, 3)
                x = torch.roll(x, shift, dims=1)

            # Frame dropout: zero ~10 % of frames
            if random.random() < 0.3:
                drop_mask = torch.rand(B, T, device=x.device) > 0.9
                x[drop_mask] = 0.0

            return x

        def _augment_val(self, x: torch.Tensor) -> torch.Tensor:
            """Deterministic center crop for val / test."""
            B, T, C, H, W = x.shape
            clips = []
            for i in range(B):
                clips.append(self.val_crop(x[i]))
            return torch.stack(clips, dim=0)

        def normalize(self, x: torch.Tensor) -> torch.Tensor:
            return x / 255.

        def _standarize(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C, H, W = x.shape
            for i in range(B):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                  sum(p.numel() for p in self.parameters()))

    # ── Outer wrapper ─────────────────────────────────────────────────────────

    def __init__(self, args=None):
        self.device = 'cpu'
        if torch.cuda.is_available() \
                and hasattr(args, 'device') and args.device == 'cuda':
            self.device = 'cuda'

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args  = args
        self._model.to(self.device)
        self._num_classes = args.num_classes

    # ── Differential learning rates ───────────────────────────────────────────

    def get_optimizer(self, opt_args):
        """
        Differential LR: backbone at 0.1× LR, temporal module + head at full LR.
        Falls back to single LR if no temporal module beyond maxpool.
        """
        base_lr = opt_args['lr']
        backbone_lr = base_lr * 0.1

        param_groups = [
            {'params': self._model._features.parameters(),
             'lr': backbone_lr, 'name': 'backbone'},
            {'params': self._model._temporal.parameters(),
             'lr': base_lr, 'name': 'temporal'},
            {'params': self._model._fc.parameters(),
             'lr': base_lr, 'name': 'head'},
        ]
        # Include loss params if they exist (e.g. learned focal params)
        if self._model._loss_fn is not None and \
                list(self._model._loss_fn.parameters()):
            param_groups.append({
                'params': self._model._loss_fn.parameters(),
                'lr': base_lr, 'name': 'loss'})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        return optimizer, scaler

    # ── Training / eval loop ──────────────────────────────────────────────────

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
              pos_weight=None):
        inference = optimizer is None
        if inference:
            self._model.eval()
        else:
            self._model.train()
            optimizer.zero_grad()

        smooth      = getattr(self._args, 'label_smooth', 0.0)
        num_classes = self._num_classes
        loss_type   = getattr(self._args, 'loss_type', 'bce')

        epoch_loss = 0.0
        ctx = torch.no_grad() if inference else nullcontext()

        with ctx:
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).float()

                # Label smoothing (only meaningful with BCE-family losses)
                if smooth > 0 and not inference:
                    label = label * (1 - smooth) + smooth / num_classes

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)

                    if loss_type == 'focal':
                        loss = self._model._loss_fn(pred, label)
                    elif loss_type == 'weighted' and pos_weight is not None:
                        loss = F.binary_cross_entropy_with_logits(
                            pred, label,
                            pos_weight=pos_weight.to(self.device))
                    else:
                        loss = F.binary_cross_entropy_with_logits(pred, label)

                if not inference:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    # ── Inference (with optional TTA) ─────────────────────────────────────────

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        seq = seq.to(self.device).float()

        self._model.eval()
        use_tta = getattr(self._args, 'use_tta', False)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                p1 = torch.sigmoid(self._model(seq))
                if use_tta:
                    p2 = torch.sigmoid(self._model(seq.flip(-1)))  # H-flip
                    pred = (p1 + p2) * 0.5
                else:
                    pred = p1

        return pred.cpu().numpy()