"""
File containing the different modules related to the model: T-DEED.
"""

#Standard imports
import abc
import torch
import torch.nn as nn
import math

#Local imports

class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()

class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), \
            torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)    

class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if len(x.shape) == 3:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            return self._fc_out(self.dropout(x)).reshape(b, t, -1)
        elif len(x.shape) == 2:
            return self._fc_out(self.dropout(x))

def step(optimizer, scaler, loss, lr_scheduler=None):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if lr_scheduler is not None:
        lr_scheduler.step()
    optimizer.zero_grad()