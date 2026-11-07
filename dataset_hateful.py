
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.autograd import Variable

from dataset_base import DatasetConfig, OrthogonalProjectionLoss

FEAT_DIR    = './features_CLIP_HatefulMemes'
NUM_CLASSES = 2

def _load_split(split: str, feat_dir: str):
    img = np.load(f'{feat_dir}/image_feats_{split}.npy').astype(np.float32)
    txt = np.load(f'{feat_dir}/text_feats_{split}.npy').astype(np.float32)
    lbl = np.load(f'{feat_dir}/labels_{split}.npy').astype(np.int64)
    return img, txt, lbl

def load_train(feat_dir: str = FEAT_DIR, seed: int = 1):
    img, txt, lbl = _load_split('train', feat_dir)
    perm  = np.random.default_rng(seed).permutation(len(lbl))
    n_hat = int(lbl.sum())
    print(f'[hateful] Train: {len(lbl)}  hateful={n_hat}  not={len(lbl)-n_hat}')
    return img[perm], txt[perm], lbl[perm]

def load_val(feat_dir: str = FEAT_DIR):
    """Returns the dev split for model selection (best checkpoint).
    The competition test labels were never released, so dev.jsonl is the
    only labelled evaluation set.  This matches the standard post-competition
    protocol used by all published baselines (Chameleon, ViLT, Ma et al., etc.)."""
    img, txt, lbl = _load_split('dev', feat_dir)
    n_hat = int(lbl.sum())
    print(f'[hateful] Val/dev: {len(lbl)}  hateful={n_hat}  not={len(lbl)-n_hat}')
    return torch.from_numpy(img).float(), torch.from_numpy(txt).float(), lbl

def load_test(feat_dir: str = FEAT_DIR):
    """Returns the dev split for final evaluation (same as load_val).
    See load_val docstring for protocol justification."""
    img, txt, lbl = _load_split('dev', feat_dir)
    n_hat = int(lbl.sum())
    print(f'[hateful] Test/dev: {len(lbl)}  hateful={n_hat}  not={len(lbl)-n_hat}')
    return torch.from_numpy(img).float(), torch.from_numpy(txt).float(), lbl

def apply_missing(feats: torch.Tensor, missing_pct: float, seed: int) -> torch.Tensor:
    if missing_pct <= 0:
        return feats.clone()
    feats = feats.clone()
    N     = feats.shape[0]
    n_missing = int(round(N * missing_pct / 100.0))
    rng   = np.random.default_rng(seed)
    feats[rng.choice(N, size=n_missing, replace=False)] = 0.0
    return feats

def run_inference(args, model,
                  img_feats: torch.Tensor, text_feats: torch.Tensor,
                  img_missing_pct: float = 0.0, text_missing_pct: float = 0.0,
                  seed: int = 42) -> np.ndarray:
    model.eval()
    img_m  = apply_missing(img_feats,  img_missing_pct,  seed)
    text_m = apply_missing(text_feats, text_missing_pct, seed + 1)
    all_logits = []
    with torch.no_grad():
        for s in range(0, img_m.shape[0], 512):
            imgs = img_m[s:s + 512]
            txts = text_m[s:s + 512]
            if args.cuda:
                imgs, txts = imgs.cuda(), txts.cuda()
            (_, logits), _, _ = model(Variable(imgs), Variable(txts))
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)

def evaluate(args, model,
             img_feats: torch.Tensor, text_feats: torch.Tensor,
             labels: np.ndarray,
             img_missing_pct: float = 0.0, text_missing_pct: float = 0.0,
             seed: int = 42, verbose: bool = True) -> tuple[float, float]:
    logits = run_inference(args, model, img_feats, text_feats,
                           img_missing_pct, text_missing_pct, seed)
    probs  = F.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds  = (probs >= 0.5).astype(int)
    auroc  = roc_auc_score(labels, probs)
    acc    = accuracy_score(labels, preds)
    if verbose:
        print(f'  [img_miss={img_missing_pct:3.0f}%  text_miss={text_missing_pct:3.0f}%]'
              f'  AUROC: {auroc*100:.2f}%   Acc: {acc*100:.2f}%')
    return auroc, acc

MISSING_SCENARIOS = [
    (0, 0), (0, 10), (0, 30), (0, 50), (0, 70), (0, 90),
            (10, 0), (30, 0), (50, 0), (70, 0), (90, 0),
            (25, 25),(35, 35),(45, 45),
]

def run_missing_benchmark(args, model,
                           img_feats: torch.Tensor, text_feats: torch.Tensor,
                           labels: np.ndarray, seed: int = 42) -> pd.DataFrame:
    print('\n' + '='*60)
    print('  MISSING MODALITY BENCHMARK  (zero-out at test time)')
    print('='*60)
    print(f"  {'img_miss':>10}  {'text_miss':>10}  {'AUROC (%)':>10}  {'Acc (%)':>10}")
    print('-'*60)
    rows = []
    for img_m, text_m in MISSING_SCENARIOS:
        auroc, acc = evaluate(args, model, img_feats, text_feats, labels,
                              img_missing_pct=img_m, text_missing_pct=text_m,
                              seed=seed, verbose=False)
        print(f'  {img_m:>10}  {text_m:>10}  {auroc*100:>10.2f}  {acc*100:>10.2f}')
        rows.append({'img_missing': img_m, 'text_missing': text_m,
                     'auroc': round(auroc*100, 2), 'accuracy': round(acc*100, 2)})
    print('='*60)
    return pd.DataFrame(rows)

class HatefulMemesConfig(DatasetConfig):
    short_name         = 'hateful'
    num_classes        = NUM_CLASSES
    multilabel         = False
    default_use_aux    = True
    default_aux_weight = 0.5

    def __init__(self, feat_dir: str = FEAT_DIR):
        self.feat_dir = feat_dir

    def load_train(self, seed: int):
        return load_train(self.feat_dir, seed)

    def load_val(self):
        return load_val(self.feat_dir)

    def load_test(self):
        return load_test(self.feat_dir)

    def task_loss(self, lbl_train: np.ndarray, device) -> nn.Module:
        n_pos = float(lbl_train.sum())
        n_neg = float(len(lbl_train) - n_pos)
        w     = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=w).to(device)

    def opl(self, device) -> nn.Module:
        return OrthogonalProjectionLoss(device).to(device)

    @property
    def label_dtype(self) -> torch.dtype:
        return torch.long

    @property
    def metric_label(self) -> str:
        return 'AUROC'

    def val_score(self, model, model_args, img_v, txt_v, lbl_v) -> float:
        auroc, _ = evaluate(model_args, model, img_v, txt_v, lbl_v,
                            img_missing_pct=0, text_missing_pct=0, verbose=False)
        print(f'  AUROC: {auroc*100:.2f}%')
        return float(auroc)

    def benchmark(self, model, model_args,
                  img_v, txt_v, lbl_v,
                  img_t, txt_t, lbl_t,
                  seed: int, run_tag: str = '') -> pd.DataFrame:
        return run_missing_benchmark(model_args, model, img_t, txt_t, lbl_t, seed=seed)
