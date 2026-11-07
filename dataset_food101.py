
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.autograd import Variable

from dataset_base import DatasetConfig, OrthogonalProjectionLoss

FEAT_DIR    = './features_CLIP_Food101'
NUM_CLASSES = 101

def load_train(feat_dir: str = FEAT_DIR, seed: int = 1):
    img_df  = pd.read_csv(f'{feat_dir}/image_feats_train.csv', header=None)
    txt_df  = pd.read_csv(f'{feat_dir}/text_feats_train.csv',  header=None)

    lbl = preprocessing.LabelEncoder().fit_transform(img_df.iloc[:, -1])
    img = np.asarray(img_df.iloc[:, :-1], dtype=np.float64)
    txt = np.asarray(txt_df.iloc[:, :-1], dtype=np.float64)

    print(f'[food101] Train: {len(lbl)}')
    perm = np.random.permutation(len(lbl))
    return img[perm].astype(np.float64), txt[perm].astype(np.float64), lbl[perm]

def load_test(feat_dir: str = FEAT_DIR):
    img_df  = pd.read_csv(f'{feat_dir}/image_feats_test.csv', header=None)
    txt_df  = pd.read_csv(f'{feat_dir}/text_feats_test.csv',  header=None)

    lbl = torch.from_numpy(np.asarray(img_df.iloc[:, -1],  dtype=np.int64))
    img = torch.from_numpy(np.asarray(img_df.iloc[:, :-1], dtype=np.float32))
    txt = torch.from_numpy(np.asarray(txt_df.iloc[:, :-1], dtype=np.float32))

    print(f'[food101] Test : {len(lbl)}')
    return img, txt, lbl

def apply_missing(feats: torch.Tensor, missing_pct: float, seed: int) -> torch.Tensor:
    if missing_pct <= 0:
        return feats.clone()
    feats     = feats.clone()
    N         = feats.shape[0]
    n_missing = int(round(N * missing_pct / 100.0))
    rng       = np.random.default_rng(seed)
    feats[rng.choice(N, size=n_missing, replace=False)] = 0.0
    return feats

def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk  = np.argsort(logits, axis=1)[:, -k:]
    return float(np.any(topk == labels[:, None], axis=1).mean())

def test(args, model,
         img_feats: torch.Tensor, text_feats: torch.Tensor, labels: torch.Tensor,
         img_missing_pct: float = 0.0, text_missing_pct: float = 0.0,
         seed: int = 42, verbose: bool = True) -> tuple[float, float]:
    model.eval()
    if args.cuda:
        model.cuda()

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

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = labels.numpy()
    top1 = topk_accuracy(logits_np, labels_np, k=1)
    top5 = topk_accuracy(logits_np, labels_np, k=5)

    if verbose:
        print(f'  [img_miss={img_missing_pct:3.0f}%  text_miss={text_missing_pct:3.0f}%]'
              f'  Top-1: {top1*100:.2f}%   Top-5: {top5*100:.2f}%')
    return top1, top5

MISSING_SCENARIOS = [
    (0, 0), (0, 10), (0, 30), (0, 50), (0, 70), (0, 90),
            (10, 0), (30, 0), (50, 0), (70, 0), (90, 0),
            (25, 25),(35, 35),(45, 45),
]

def run_missing_benchmark(args, model,
                           img_feats: torch.Tensor, text_feats: torch.Tensor,
                           labels: torch.Tensor, seed: int = 42) -> pd.DataFrame:
    print('\n' + '='*60)
    print('  MISSING MODALITY BENCHMARK  (zero-out at test time only)')
    print('='*60)
    print(f"  {'img_miss':>10}  {'text_miss':>10}  {'Top-1 (%)':>10}  {'Top-5 (%)':>10}")
    print('-'*60)
    rows = []
    for img_m, text_m in MISSING_SCENARIOS:
        top1, top5 = test(args, model, img_feats, text_feats, labels,
                          img_missing_pct=img_m, text_missing_pct=text_m,
                          seed=seed, verbose=False)
        print(f'  {img_m:>10}  {text_m:>10}  {top1*100:>10.2f}  {top5*100:>10.2f}')
        rows.append({'img_missing': img_m, 'text_missing': text_m,
                     'top1': round(top1*100, 2), 'top5': round(top5*100, 2)})
    print('='*60)
    return pd.DataFrame(rows)

class Food101Config(DatasetConfig):
    short_name         = 'food101'
    num_classes        = NUM_CLASSES
    multilabel         = False
    default_use_aux    = True
    default_aux_weight = 0.5

    def __init__(self, feat_dir: str = FEAT_DIR):
        self.feat_dir = feat_dir

    def load_train(self, seed: int):
        return load_train(self.feat_dir, seed)

    def load_val(self):
        return load_test(self.feat_dir)

    def load_test(self):
        return load_test(self.feat_dir)

    def task_loss(self, lbl_train: np.ndarray, device) -> nn.Module:
        return nn.CrossEntropyLoss().to(device)

    def opl(self, device) -> nn.Module:
        return OrthogonalProjectionLoss(device).to(device)

    @property
    def label_dtype(self) -> torch.dtype:
        return torch.long

    @property
    def metric_label(self) -> str:
        return 'Top-1'

    def val_score(self, model, model_args, img_v, txt_v, lbl_v) -> float:
        top1, top5 = test(model_args, model, img_v, txt_v, lbl_v,
                          img_missing_pct=0, text_missing_pct=0, verbose=True)
        return float(top1)

    def benchmark(self, model, model_args,
                  img_v, txt_v, lbl_v,
                  img_t, txt_t, lbl_t,
                  seed: int, run_tag: str = '') -> pd.DataFrame:
        return run_missing_benchmark(model_args, model, img_t, txt_t, lbl_t, seed=seed)
