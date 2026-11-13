
from __future__ import division, print_function

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from dataset_base import DatasetConfig
from dataset_food101  import Food101Config
from dataset_imdb     import MMIMDbConfig
from dataset_hateful  import HatefulMemesConfig
from retrieval_model  import FOP

DATASET_CONFIGS: dict[str, type[DatasetConfig]] = {
    'food101':       Food101Config,
    'mm_imdb':       MMIMDbConfig,
    'hateful_memes': HatefulMemesConfig,
}

DEFAULT_FEAT_DIRS: dict[str, str] = {
    'food101':       './features_CLIP_Food101',
    'mm_imdb':       './features_CLIP_IMDb',
    'hateful_memes': './features_CLIP_HatefulMemes',
}

class RunningAverage:
    def __init__(self):
        self.s, self.n = 0.0, 0.0
    def update(self, v): self.s += v; self.n += 1
    def avg(self): return self.s / self.n if self.n > 0 else 0.0

def init_weights(m):
    import torch.nn as nn
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def apply_train_drop(feats: torch.Tensor, drop_pct: int, rng) -> torch.Tensor:
    if drop_pct <= 0:
        return feats
    N      = feats.shape[0]
    n_drop = int(round(N * drop_pct / 100.0))
    if n_drop == 0:
        return feats
    idx = rng.choice(N, size=n_drop, replace=False)
    feats = feats.clone()
    feats[idx] = 0.0
    return feats

def get_batch(idx: int, batch_size: int, labels, feats):
    s = idx * batch_size
    e = (idx + 1) * batch_size
    return np.asarray(feats[s:e]), np.asarray(labels[s:e])

def save_checkpoint(state: dict, directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    torch.save(state, os.path.join(directory, filename))

def get_method_name(flags) -> str:
    if not flags.use_msm and flags.alpha == 0: return 'CE'
    if flags.use_msm and flags.alpha == 0:
        if flags.msm_learn_q: return 'MSM_LEARN_Q'
        if flags.msm_linear: return 'MSM_LINEAR'
        if flags.msm_random_orthogonal: return 'MSM_RANDOM_ORTHO'
        return 'MSM_UPLIFT' if flags.msm_uplift else 'MSM'
    if not flags.use_msm and flags.alpha > 0: return 'FOP'
    return 'OURS'


def build_model(FLAGS, img_dim: int, txt_dim: int) -> FOP:
    model = FOP(FLAGS, img_dim, txt_dim)
    model.apply(init_weights)

    if FLAGS.use_msm and FLAGS.msm_uplift:
        if FLAGS.msm_linear:
            model.initialize_msm_linear()
        else:
            model.initialize_msm_projection(
                seed=FLAGS.projection_seed,
                trainable=bool(FLAGS.msm_learn_q),
            )

    if FLAGS.cuda:
        model.cuda()
        cudnn.benchmark = False
    return model

def build_optimizer(model: FOP, FLAGS) -> optim.Adam:
    if FLAGS.fusion == 'linear':
        params = [
            {'params': model.image_branch.fc1.parameters()},
            {'params': model.text_branch.fc1.parameters()},
            {'params': model.fusion_layer.weight1},
            {'params': model.fusion_layer.weight2},
        ]
    elif FLAGS.fusion in ('gated', 'crossattn'):
        params = [
            {'params': model.image_branch.fc1.parameters()},
            {'params': model.text_branch.fc1.parameters()},
            {'params': model.fusion_layer.attention.parameters()},
        ]
    else:
        params = [
            {'params': model.image_branch.fc1.parameters()},
            {'params': model.text_branch.fc1.parameters()},
            {'params': model.fusion_layer.parameters()},
        ]
    if not FLAGS.use_msm:
        params.append({'params': model.logits_layer.parameters()})
    elif FLAGS.msm_linear or FLAGS.msm_learn_q:
        params.append({'params': model.msm_expand.parameters()})
    if FLAGS.use_aux:
        params.append({'params': model.img_aux.parameters()})
        params.append({'params': model.txt_aux.parameters()})
    return optim.Adam(params, lr=FLAGS.lr, weight_decay=0.01)

def train_step(img_b, txt_b, lbl_b,
               model, optimizer, task_loss_fn, opl_fn, alpha,
               drop_img, drop_txt, rng,
               label_dtype: torch.dtype, cuda: bool) -> float:
    model.train()
    img_t = apply_train_drop(torch.from_numpy(img_b).float(), drop_img, rng)
    txt_t = apply_train_drop(torch.from_numpy(txt_b).float(), drop_txt, rng)
    lbl_t = torch.from_numpy(lbl_b).to(label_dtype)
    if cuda:
        img_t, txt_t, lbl_t = img_t.cuda(), txt_t.cuda(), lbl_t.cuda()
    img_t, txt_t = Variable(img_t), Variable(txt_t)
    comb, _, _     = model.train_forward(img_t, txt_t, lbl_t)
    loss      = task_loss_fn(comb[1], lbl_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    pre_args, _ = pre.parse_known_args()
    cfg_cls = DATASET_CONFIGS[pre_args.dataset]

    parser = argparse.ArgumentParser(
        description='Unified training script for all multimodal datasets.'
    )
    parser.add_argument('--dataset',        required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--feat_dir',       default=None)
    parser.add_argument('--fusion',         default='linear',
                        choices=['linear', 'gated', 'crossattn', 'hadamard'])
    parser.add_argument('--dim_embed',      type=int,   default=512)
    parser.add_argument('--use_msm',        action='store_true', default=False)
    parser.add_argument('--msm_uplift',       action='store_true', default=False,
                        help='Canonical C*(C-1) ETF + frozen isometric lift. Requires --use_msm.')
    parser.add_argument('--msm_linear',    action='store_true', default=False,
                        help='Trainable Xavier-initialized linear expansion; expanded prototypes are not ETF at initialization.')
    parser.add_argument('--msm_learn_q',    action='store_true', default=False,
                        help='Canonical ETF with a learnable orthogonal projection initialized at Q.')
    parser.add_argument('--msm_random_orthogonal', action='store_true', default=False,
                        help='Canonical ETF with a fixed random orthogonal projection; use --projection_seed.')
    parser.add_argument('--projection_seed', type=int, default=None,
                        help='Seed used only to construct the orthogonal projection Q.')
    parser.add_argument('--use_aux',        action='store_true')
    parser.add_argument('--aux_weight',     type=float)
    parser.add_argument('--alpha',          type=float, default=0.0)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--batch_size',     type=int,   default=512)
    parser.add_argument('--max_num_epoch',  type=int,   default=50)
    parser.add_argument('--train_img_drop', type=int,   default=0)
    parser.add_argument('--train_txt_drop', type=int,   default=0)
    parser.add_argument('--seed',           type=int,   default=1)
    parser.add_argument('--no_cuda',        action='store_true')

    parser.set_defaults(
        use_aux    = cfg_cls.default_use_aux,
        aux_weight = cfg_cls.default_aux_weight,
    )

    global FLAGS
    FLAGS, _ = parser.parse_known_args()
    FLAGS.cuda = torch.cuda.is_available() and not FLAGS.no_cuda

    if FLAGS.msm_uplift  and not FLAGS.use_msm:
        parser.error('--msm_uplift requires --use_msm')
    projection_modes = [
        FLAGS.msm_linear,
        FLAGS.msm_learn_q,
        FLAGS.msm_random_orthogonal,
    ]
    if any(projection_modes) and not FLAGS.use_msm:
        parser.error('MSM projection options require --use_msm')
    if sum(bool(x) for x in projection_modes) > 1:
        parser.error('--msm_linear, --msm_learn_q, and '
                     '--msm_random_orthogonal are mutually exclusive')
    if any(projection_modes):
        FLAGS.msm_uplift = True
    if FLAGS.projection_seed is None:
        FLAGS.projection_seed = FLAGS.seed

    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    feat_dir      = FLAGS.feat_dir or DEFAULT_FEAT_DIRS[FLAGS.dataset]
    ds_cfg        = cfg_cls(feat_dir)
    FLAGS.num_classes = ds_cfg.num_classes
    if FLAGS.use_msm and FLAGS.msm_uplift and FLAGS.dim_embed < FLAGS.num_classes - 1:
        parser.error(
            f'--dim_embed must be >= num_classes-1 ({FLAGS.num_classes - 1}) '
            'for MSM uplift projections'
        )
    method        = get_method_name(FLAGS)
    device        = torch.device('cuda' if FLAGS.cuda else 'cpu')

    print(f'\n{"="*65}')
    print(f'  dataset={FLAGS.dataset}  method={method}  fusion={FLAGS.fusion}')
    print(f'  dim_embed={FLAGS.dim_embed}  alpha={FLAGS.alpha}  '
          f'use_msm={FLAGS.use_msm}  msm_uplift={FLAGS.msm_uplift}  '
          f'msm_linear={FLAGS.msm_linear}')
    print(f'  train_img_drop={FLAGS.train_img_drop}%  '
          f'train_txt_drop={FLAGS.train_txt_drop}%')
    print(f'  use_aux={FLAGS.use_aux}  aux_weight={FLAGS.aux_weight}  '
          f'seed={FLAGS.seed}  cuda={FLAGS.cuda}')
    print(f'{"="*65}')

    if FLAGS.use_msm:
        msm_dim  = FLAGS.num_classes - 1 if FLAGS.msm_uplift else FLAGS.dim_embed
        msm_path = f'msms/prototypes{FLAGS.num_classes}_{msm_dim}.npy'
        if not os.path.exists(msm_path):
            print(f'\nERROR: MSM file not found: {msm_path}')
            print(f'Run:  python create_max_separated_matrix.py '
                  f'--classes {FLAGS.num_classes} --dim {msm_dim}')
            raise SystemExit(1)

    print('\nLoading data ...')
    img_train, txt_train, lbl_train = ds_cfg.load_train(FLAGS.seed)
    img_val,   txt_val,   lbl_val   = ds_cfg.load_val()
    img_test,  txt_test,  lbl_test  = ds_cfg.load_test()

    N_train, img_dim = img_train.shape
    _,       txt_dim = txt_train.shape
    print(f'  Train={N_train}  img_dim={img_dim}  txt_dim={txt_dim}')

    model        = build_model(FLAGS, img_dim, txt_dim)
    optimizer    = build_optimizer(model, FLAGS)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FLAGS.max_num_epoch, eta_min=1e-6)
    task_loss_fn = ds_cfg.task_loss(lbl_train, device)
    opl_fn       = ds_cfg.opl(device)
    print(f'  Parameters: {sum(p.nelement() for p in model.parameters()):,}')

    drop_tag = (f'_imgdrop{FLAGS.train_img_drop}_txtdrop{FLAGS.train_txt_drop}'
                if FLAGS.train_img_drop > 0 or FLAGS.train_txt_drop > 0 else '')
    if getattr(FLAGS, 'msm_learn_q', False):
        proj_tag = '_learnQ'
    elif getattr(FLAGS, 'msm_linear', False):
        proj_tag = '_linearXavier'
    elif getattr(FLAGS, 'msm_random_orthogonal', False):
        proj_tag = f'_randomOrtho{FLAGS.projection_seed}'
    elif FLAGS.msm_uplift:
        proj_tag = '_uplift'
    else:
        proj_tag = ''
    run_tag  = (f'{ds_cfg.short_name}_{FLAGS.fusion}_{method}_dim{FLAGS.dim_embed}_seed{FLAGS.seed}_aux{FLAGS.aux_weight}'
                f'{drop_tag}{proj_tag}')
    # Single hierarchy shared by checkpoints and the final benchmark CSV, so a
    # given run is never scattered across differently-shaped paths:
    #   {root}/{dataset}/{fusion}/aux{aux}/seed{seed}/dim{dim}/...
    run_dir  = (f'{ds_cfg.short_name}/{FLAGS.fusion}/{method}/aux{FLAGS.aux_weight}'
                f'/seed{FLAGS.seed}/dim{FLAGS.dim_embed}')
    save_dir = f'raw_ckpt/{run_dir}'
    best_dir = f'best/{run_dir}'

    os.makedirs('output', exist_ok=True)

    drop_rng    = np.random.default_rng(FLAGS.seed + 9999)
    num_batches = N_train // FLAGS.batch_size
    best_score  = 0.0
    loss_plot, score_plot = [], []

    for epoch in range(1, FLAGS.max_num_epoch + 1):
        print(f'Epoch {epoch:03d}/{FLAGS.max_num_epoch}  [{method}]  alpha={FLAGS.alpha}'
              f'  img_drop={FLAGS.train_img_drop}%  txt_drop={FLAGS.train_txt_drop}%')

        avg_loss = RunningAverage()
        for idx in tqdm(range(num_batches)):
            img_b, lbl_b = get_batch(idx, FLAGS.batch_size, lbl_train, img_train)
            txt_b, _     = get_batch(idx, FLAGS.batch_size, lbl_train, txt_train)
            loss = train_step(img_b, txt_b, lbl_b,
                              model, optimizer, task_loss_fn, opl_fn, FLAGS.alpha,
                              FLAGS.train_img_drop, FLAGS.train_txt_drop, drop_rng,
                              ds_cfg.label_dtype, FLAGS.cuda)
            avg_loss.update(loss)

        scheduler.step()
        epoch_loss = avg_loss.avg()
        loss_plot.append(epoch_loss)

        # Overwrite a single "last" checkpoint each epoch (rather than one file
        # per epoch) so resumability doesn't cost max_num_epoch files per run.
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()},
                        save_dir, 'checkpoint_last.pth.tar')

        print(f'  Loss: {epoch_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}')
        score = ds_cfg.val_score(model, FLAGS, img_val, txt_val, lbl_val)
        score_plot.append(score)

        if score > best_score:
            best_score = score
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()},
                            best_dir, 'best_model.pth.tar')
            print(f'  *** New best {ds_cfg.metric_label}: {best_score*100:.2f}%'
                  f'  [{method}] ***')

    best_ckpt = os.path.join(best_dir, 'best_model.pth.tar')
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        print(f'\nLoaded best checkpoint (epoch {ckpt["epoch"]}): {best_ckpt}')

    print('\nRunning final missing-modality benchmark ...')
    bench_df = ds_cfg.benchmark(model, FLAGS,
                                 img_val,  txt_val,  lbl_val,
                                 img_test, txt_test, lbl_test,
                                 seed=FLAGS.seed, run_tag=run_tag)
    bench_csv = f'output/{run_dir}/{run_tag}_missing_benchmark.csv'
    os.makedirs(os.path.dirname(bench_csv), exist_ok=True)
    bench_df.to_csv(bench_csv, index=False)
    print(f'Benchmark saved -> {bench_csv}')
    print(f'\nBest {ds_cfg.metric_label}: {best_score*100:.2f}%  [{method}]  '
          f'dim={FLAGS.dim_embed}')

if __name__ == '__main__':
    main()
