
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.autograd import Variable

from dataset_base import DatasetConfig, MultiLabelOPL

FEAT_DIR    = './features_CLIP_IMDb'
NUM_CLASSES = 23

def _load_split(split: str, feat_dir: str):
    img = np.load(f'{feat_dir}/image_feats_{split}.npy').astype(np.float32)
    txt = np.load(f'{feat_dir}/text_feats_{split}.npy').astype(np.float32)
    lbl = np.load(f'{feat_dir}/labels_{split}.npy').astype(np.float32)
    return img, txt, lbl

def load_train(feat_dir: str = FEAT_DIR, seed: int = 1):
    img, txt, lbl = _load_split('train', feat_dir)
    perm = np.random.default_rng(seed).permutation(len(lbl))
    print(f'[imdb] Train: {len(lbl)}  avg genres/sample: {lbl.sum(axis=1).mean():.2f}')
    return img[perm], txt[perm], lbl[perm]

def load_val(feat_dir: str = FEAT_DIR):
    img, txt, lbl = _load_split('val', feat_dir)
    print(f'[imdb] Val  : {len(lbl)}')
    return torch.from_numpy(img).float(), torch.from_numpy(txt).float(), lbl

def load_test(feat_dir: str = FEAT_DIR):
    img, txt, lbl = _load_split('test', feat_dir)
    print(f'[imdb] Test : {len(lbl)}')
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

def find_thresholds(logits: np.ndarray, labels: np.ndarray,
                    n_steps: int = 101) -> np.ndarray:
    probs      = 1.0 / (1.0 + np.exp(-logits))
    C          = labels.shape[1]
    thresholds = np.full(C, 0.5)
    for c in range(C):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, n_steps):
            preds = (probs[:, c] >= t).astype(int)
            f1    = f1_score(labels[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds

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
             labels: np.ndarray, thresholds: np.ndarray | None = None,
             img_missing_pct: float = 0.0, text_missing_pct: float = 0.0,
             seed: int = 42, verbose: bool = True) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.full(labels.shape[1], 0.5)
    logits = run_inference(args, model, img_feats, text_feats,
                           img_missing_pct, text_missing_pct, seed)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    preds  = (probs >= thresholds).astype(int)
    f1_mac = f1_score(labels, preds, average='macro',    zero_division=0)
    f1_wt  = f1_score(labels, preds, average='weighted', zero_division=0)
    if verbose:
        print(f'  [img_miss={img_missing_pct:3.0f}%  text_miss={text_missing_pct:3.0f}%]'
              f'  F1-macro: {f1_mac*100:.2f}%   F1-weighted: {f1_wt*100:.2f}%')
    return f1_mac, f1_wt

MISSING_SCENARIOS = [
    (0, 0), (0, 10), (0, 30), (0, 50), (0, 70), (0, 90),
            (10, 0), (30, 0), (50, 0), (70, 0), (90, 0),
            (25, 25),(35, 35),(45, 45),
]

def run_missing_benchmark(args, model,
                           img_feats: torch.Tensor, text_feats: torch.Tensor,
                           labels: np.ndarray, thresholds: np.ndarray,
                           seed: int = 42) -> pd.DataFrame:
    print('\n' + '='*65)
    print('  MISSING MODALITY BENCHMARK  (zero-out at test time)')
    print('='*65)
    print(f"  {'img_miss':>10}  {'text_miss':>10}  {'F1-macro (%)':>13}  {'F1-weighted (%)':>16}")
    print('-'*65)
    rows = []
    for img_m, text_m in MISSING_SCENARIOS:
        f1_mac, f1_wt = evaluate(args, model, img_feats, text_feats, labels,
                                  thresholds=thresholds,
                                  img_missing_pct=img_m, text_missing_pct=text_m,
                                  seed=seed, verbose=False)
        print(f'  {img_m:>10}  {text_m:>10}  {f1_mac*100:>13.2f}  {f1_wt*100:>16.2f}')
        rows.append({'img_missing': img_m, 'text_missing': text_m,
                     'f1_macro': round(f1_mac*100, 2), 'f1_weighted': round(f1_wt*100, 2)})
    print('='*65)
    return pd.DataFrame(rows)

class MMIMDbConfig(DatasetConfig):
    short_name         = 'imdb'
    num_classes        = NUM_CLASSES
    multilabel         = True
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
        N      = len(lbl_train)
        counts = lbl_train.sum(axis=0).astype(np.float32)
        pos_w  = torch.tensor((N - counts) / np.maximum(counts, 1.0),
                              dtype=torch.float32).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_w).to(device)

    def opl(self, device) -> nn.Module:
        return MultiLabelOPL(device).to(device)

    @property
    def label_dtype(self) -> torch.dtype:
        return torch.float

    @property
    def metric_label(self) -> str:
        return 'Val-F1-macro'

    def val_score(self, model, model_args, img_v, txt_v, lbl_v) -> float:
        logits = run_inference(model_args, model, img_v, txt_v,
                               img_missing_pct=0, text_missing_pct=0,
                               seed=model_args.seed if hasattr(model_args, 'seed') else 1)
        probs  = 1.0 / (1.0 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)
        f1     = float(f1_score(lbl_v, preds, average='macro', zero_division=0))
        print(f'  Val F1-macro (thr=0.5): {f1*100:.2f}%')
        return f1

    def benchmark(self, model, model_args,
                  img_v, txt_v, lbl_v,
                  img_t, txt_t, lbl_t,
                  seed: int, run_tag: str = '') -> pd.DataFrame:
        print('\nOptimising per-class thresholds on validation set ...')
        val_logits = run_inference(model_args, model, img_v, txt_v,
                                   img_missing_pct=0, text_missing_pct=0, seed=seed)
        thresholds = find_thresholds(val_logits, lbl_v, n_steps=101)
        print(f'  Range: [{thresholds.min():.3f}, {thresholds.max():.3f}]  '
              f'mean: {thresholds.mean():.3f}')
        if run_tag:
            import os; os.makedirs('output', exist_ok=True)
            np.save(f'output/thresholds_{run_tag}.npy', thresholds)
            print(f'  Thresholds saved -> output/thresholds_{run_tag}.npy')
        return run_missing_benchmark(model_args, model, img_t, txt_t, lbl_t,
                                     thresholds=thresholds, seed=seed)
