
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, features, labels):
        features = nn.functional.normalize(features, p=2, dim=1)
        labels   = labels[:, None]
        mask     = torch.eq(labels, labels.t()).bool().to(self.device)
        eye      = torch.eye(mask.shape[0], device=self.device).bool()
        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())
        pos_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)
        return (1.0 - pos_mean) + 0.7 * neg_mean, pos_mean, neg_mean

class MultiLabelOPL(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, features, labels):
        features = nn.functional.normalize(features, p=2, dim=1)
        dev      = features.device
        eye      = torch.eye(len(features), device=dev).bool()
        shared   = torch.matmul(labels.float().to(dev), labels.float().t().to(dev))
        mask_pos = (shared > 0) & ~eye
        mask_neg = (shared == 0) & ~eye
        dot_prod = torch.matmul(features, features.t())
        pos_mean = (mask_pos.float() * dot_prod).sum() / (mask_pos.float().sum() + 1e-6)
        neg_mean = (mask_neg.float() * dot_prod.abs()).sum() / (mask_neg.float().sum() + 1e-6)
        return (1.0 - pos_mean) + 0.7 * neg_mean, pos_mean, neg_mean

class DatasetConfig(ABC):

    short_name:         str
    num_classes:        int
    multilabel:         bool
    default_use_aux:    bool
    default_aux_weight: float

    @abstractmethod
    def load_train(self, seed: int): ...

    @abstractmethod
    def load_val(self): ...

    @abstractmethod
    def load_test(self): ...

    @abstractmethod
    def task_loss(self, lbl_train: np.ndarray, device) -> nn.Module: ...

    @abstractmethod
    def opl(self, device) -> nn.Module: ...

    @property
    @abstractmethod
    def label_dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def metric_label(self) -> str: ...

    @abstractmethod
    def val_score(self, model, model_args, img_v, txt_v, lbl_v) -> float: ...

    @abstractmethod
    def benchmark(self, model, model_args,
                  img_v, txt_v, lbl_v,
                  img_t, txt_t, lbl_t,
                  seed: int, run_tag: str = '') -> pd.DataFrame: ...
