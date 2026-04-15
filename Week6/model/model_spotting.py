"""
model/model_spotting.py
Extended to support BiGRU temporal head + Focal Loss.
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


# ── Focal Loss ────────────────────────────────────────────────────────────────
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


# ── Main Model ────────────────────────────────────────────────────────────────
class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch  = args.feature_arch
            self._temporal_arch = getattr(args, 'temporal_arch', 'none')

            # Backbone
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim
            else:
                raise NotImplementedError(args.feature_arch)
            self._features = features

            # Temporal head
            if self._temporal_arch == 'none':
                self._temporal = None
                head_dim = self._d

            elif self._temporal_arch == 'bigru':
                hidden = getattr(args, 'gru_hidden_dim', 512)
                layers = getattr(args, 'gru_num_layers', 1)
                drop   = getattr(args, 'gru_dropout', 0.1)
                self._temporal = BiGRUHead(
                    self._d, hidden_dim=hidden,
                    num_layers=layers, dropout=drop)
                head_dim = self._temporal.out_dim
                print(f'BiGRU: hidden={hidden}, layers={layers}, dropout={drop}')

            else:
                raise NotImplementedError(f'Unknown temporal_arch: {self._temporal_arch}')

            # Classification head
            self._fc = FCLayers(head_dim, args.num_classes + 1)

            # Augmentations
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
            B, T, C, H, W = x.shape
            if self.training:
                x = self.augment(x)
            x = self.standarize(x)

            feat = self._features(
                x.view(-1, C, H, W)
            ).reshape(B, T, self._d)

            if self._temporal is not None:
                feat = self._temporal(feat)

            return self._fc(feat)

        def normalize(self, x):  return x / 255.
        def augment(self, x):
            for i in range(x.shape[0]): x[i] = self.augmentation(x[i])
            return x
        def standarize(self, x):
            for i in range(x.shape[0]): x[i] = self.standarization(x[i])
            return x
        def print_stats(self):
            total = sum(p.numel() for p in self.parameters())
            print(f'Model params: {total:,}')

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
        if loss_type == 'focal':
            gamma = getattr(args, 'focal_gamma', 2.0)
            self._criterion = FocalLoss(gamma=gamma, weight=weights)
            print(f'Loss: Focal (gamma={gamma})')
        else:
            self._criterion = None
            print('Loss: Cross-Entropy')

        self._weights   = weights
        self._loss_type = loss_type
        self._model.to(self.device)

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
                label = batch['label'].to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred  = self._model(frame).view(-1, self._num_classes + 1)
                    label = label.view(-1)
                    loss  = (self._criterion(pred, label) if self._loss_type == 'focal'
                             else F.cross_entropy(pred, label, reduction='mean',
                                                  weight=self._weights))

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
