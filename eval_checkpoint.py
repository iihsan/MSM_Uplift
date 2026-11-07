
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import dataset_food101  as food101
import dataset_imdb     as imdb
import dataset_hateful  as hateful
from retrieval_model import FOP

FEAT_DIRS = {
    'food101':       './features_CLIP_Food101',
    'mm_imdb':       './features_CLIP_IMDb',
    'hateful_memes': './features_CLIP_HatefulMemes'
}

def _detect_config(state_dict: dict, cuda: bool):
    args = argparse.Namespace()
    args.cuda = cuda

    img_w = state_dict['image_branch.fc1.0.weight']
    txt_w = state_dict['text_branch.fc1.0.weight']
    args.dim_embed = int(img_w.shape[0])
    img_feat_dim   = int(img_w.shape[1])
    text_feat_dim  = int(txt_w.shape[1])

    keys = set(state_dict.keys())
    if 'fusion_layer.weight1' in keys:
        args.fusion = 'linear'
    elif any(k.startswith('fusion_layer.attention') for k in keys):
        args.fusion = 'gated'
    else:
        args.fusion = 'hadamard'

    if 'maxsep' in keys:
        args.use_msm     = True
        args.msm_uplift    = False
        args.num_classes = int(state_dict['maxsep'].shape[1])
    elif 'msm_canonical' in keys:
        args.use_msm     = True
        args.msm_uplift    = True
        args.num_classes = int(state_dict['msm_canonical'].shape[0])
    else:
        args.use_msm     = False
        args.msm_uplift    = False
        args.num_classes = int(state_dict['logits_layer.weight'].shape[0])

    args.use_aux    = 'img_aux.weight' in keys
    args.aux_weight = 0.5

    return args, img_feat_dim, text_feat_dim

def _ensure_msm_file(state_dict: dict, args) -> None:
    if not args.use_msm:
        return
    if args.msm_uplift:
        path = f'msms/prototypes{args.num_classes}_{args.num_classes - 1}.npy'
        if not os.path.exists(path):
            os.makedirs('msms', exist_ok=True)
            np.save(path, state_dict['msm_canonical'].numpy())
            print(f'  Reconstructed MSM file -> {path}')
    else:
        path = f'msms/prototypes{args.num_classes}_{args.dim_embed}.npy'
        if not os.path.exists(path):
            os.makedirs('msms', exist_ok=True)
            np.save(path, state_dict['maxsep'].numpy().T)
            print(f'  Reconstructed MSM file -> {path}')

def load_checkpoint(ckpt_path: str, cuda: bool):
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt       = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    print(f'  Checkpoint epoch: {ckpt.get("epoch", "?")}')

    model_args, img_dim, txt_dim = _detect_config(state_dict, cuda)
    _ensure_msm_file(state_dict, model_args)

    print(f'  dim_embed={model_args.dim_embed}  num_classes={model_args.num_classes}')
    print(f'  fusion={model_args.fusion}  use_msm={model_args.use_msm}'
          f'  msm_uplift={model_args.msm_uplift}  use_aux={model_args.use_aux}')
    print(f'  img_dim={img_dim}  txt_dim={txt_dim}')

    model = FOP(model_args, img_dim, txt_dim)
    model.load_state_dict(state_dict)
    model.eval()
    if cuda:
        model.cuda()
    return model, model_args

def run_food101(model, model_args, feat_dir, seed):
    print(f'\nLoading Food-101 test set from {feat_dir} ...')
    img, txt, lbl = food101.load_test(feat_dir)
    print(f'  Samples: {len(lbl)}')
    print('\nFull-modality evaluation:')
    top1, top5 = food101.test(model_args, model, img, txt, lbl,
                               img_missing_pct=0, text_missing_pct=0,
                               seed=seed, verbose=True)
    bench = food101.run_missing_benchmark(model_args, model, img, txt, lbl, seed=seed)
    return top1, top5, bench

def run_mm_imdb(model, model_args, feat_dir, seed, thresholds_path):
    print(f'\nLoading MM-IMDb test set from {feat_dir} ...')
    img_test, txt_test, lbl_test = imdb.load_test(feat_dir)
    print(f'  Test samples: {len(lbl_test)}')

    if thresholds_path:
        print(f'  Loading thresholds from {thresholds_path}')
        thresholds = np.load(thresholds_path)
    else:
        print('  Re-optimising thresholds on validation set ...')
        img_val, txt_val, lbl_val = imdb.load_val(feat_dir)
        val_logits = imdb.run_inference(model_args, model, img_val, txt_val,
                                         img_missing_pct=0, text_missing_pct=0, seed=seed)
        thresholds = imdb.find_thresholds(val_logits, lbl_val, n_steps=101)
        print(f'  Range: [{thresholds.min():.3f}, {thresholds.max():.3f}]  '
              f'mean: {thresholds.mean():.3f}')

    print('\nFull-modality evaluation:')
    f1_mac, f1_wt = imdb.evaluate(model_args, model, img_test, txt_test, lbl_test,
                                    thresholds=thresholds,
                                    img_missing_pct=0, text_missing_pct=0,
                                    seed=seed, verbose=True)
    bench = imdb.run_missing_benchmark(model_args, model, img_test, txt_test, lbl_test,
                                        thresholds=thresholds, seed=seed)
    return f1_mac, f1_wt, bench

def run_hateful_memes(model, model_args, feat_dir, seed):
    print(f'\nLoading Hateful Memes dev set from {feat_dir} ...')
    img, txt, lbl = hateful.load_test(feat_dir)
    print(f'  Samples: {len(lbl)}')
    print('\nFull-modality evaluation:')
    auroc, acc = hateful.evaluate(model_args, model, img, txt, lbl,
                                   img_missing_pct=0, text_missing_pct=0,
                                   seed=seed, verbose=True)
    bench = hateful.run_missing_benchmark(model_args, model, img, txt, lbl, seed=seed)
    return auroc, acc, bench

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a saved FOP checkpoint on the held-out test set.'
    )
    parser.add_argument('dataset',
                        choices=['food101', 'mm_imdb', 'hateful_memes'])
    parser.add_argument('checkpoint',  help='Path to checkpoint (.pth.tar)')
    parser.add_argument('--feat_dir',   default=None)
    parser.add_argument('--thresholds', default=None,
                        help='(MM-IMDb) Per-class threshold file (.npy).')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--no_cuda',    action='store_true')
    parser.add_argument('--out_csv',    default=None)
    cli = parser.parse_args()

    cuda     = torch.cuda.is_available() and not cli.no_cuda
    feat_dir = cli.feat_dir or FEAT_DIRS[cli.dataset]

    print(f'\n{"="*65}')
    print(f'  eval_checkpoint.py')
    print(f'  dataset    : {cli.dataset}')
    print(f'  checkpoint : {cli.checkpoint}')
    print(f'  feat_dir   : {feat_dir}')
    print(f'  cuda       : {cuda}   seed: {cli.seed}')
    print(f'{"="*65}')

    model, model_args = load_checkpoint(cli.checkpoint, cuda)

    if cli.dataset == 'food101':
        top1, top5, bench = run_food101(model, model_args, feat_dir, cli.seed)
        print(f'\n{"="*65}')
        print(f'  FINAL RESULT  --  Food-101 test set')
        print(f'  Top-1: {top1*100:.2f}%   Top-5: {top5*100:.2f}%')
        print(f'{"="*65}')

    elif cli.dataset == 'mm_imdb':
        f1_mac, f1_wt, bench = run_mm_imdb(model, model_args, feat_dir,
                                             cli.seed, cli.thresholds)
        print(f'\n{"="*65}')
        print(f'  FINAL RESULT  --  MM-IMDb test set')
        print(f'  F1-macro: {f1_mac*100:.2f}%   F1-weighted: {f1_wt*100:.2f}%')
        print(f'{"="*65}')

    elif cli.dataset == 'hateful_memes':
        auroc, acc, bench = run_hateful_memes(model, model_args, feat_dir, cli.seed)
        print(f'\n{"="*65}')
        print(f'  FINAL RESULT  --  Hateful Memes dev set')
        print(f'  AUROC: {auroc*100:.2f}%   Accuracy: {acc*100:.2f}%')
        print(f'{"="*65}')

    if cli.out_csv:
        bench.to_csv(cli.out_csv, index=False)
        print(f'\nBenchmark saved -> {cli.out_csv}')

if __name__ == '__main__':
    main()
