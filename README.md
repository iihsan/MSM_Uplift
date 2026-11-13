# Geometry-Preserving Uplift of ETF Prototypes for Robust Multimodal Learning under Missing Modalities

Training and evaluation code for UPMC Food-101, MM-IMDb, and Hateful
Memes using cross-entropy (CE), canonical Max-Separated Matrix (MSM)
prototypes, and MSM projection variants.

## 1. Dataset preparation

Download the precomputed CLIP features and extract them into:

Downloads: [Food-101](https://zenodo.org/records/24032786),
[MM-IMDb](https://zenodo.org/records/24033803), [Hateful
Memes](https://zenodo.org/records/24034415).

Extract the downloaded files into their respective directories, preserving the expected directory structure.

```text
project/
├── features_CLIP_Food101/
├── features_CLIP_IMDb/
├── features_CLIP_HatefulMemes/
├── train.py
├── requirements.txt
└── output/
```

## 2. Environment

Install the CUDA-enabled PyTorch build appropriate for your machine from
the [PyTorch installation
selector](https://pytorch.org/get-started/locally/), then install
project dependencies:

``` bash
pip install -r requirements.txt
```

## 3. Training

General command:

``` bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm [PROJECTION_FLAG] --seed 1
```

Supported datasets: `food101`, `mm_imdb`, `hateful_memes`.

Supported fusion strategies: `linear`, `hadamard`.

### 3.1 Main arguments

  -----------------------------------------------------------------------
  Argument                            Description
  ----------------------------------- -----------------------------------
  `--dataset`                         Dataset identifier: `food101`,
                                      `mm_imdb`, or `hateful_memes`.

  `--fusion`                          Fusion strategy: `linear`, or `hadamard`.

  `--dim_embed`                       Image/text/fused embedding
                                      dimension; commonly `512` for
                                      uplift experiments.

  `--use_msm`                         Enables MSM-based classification.

  `--seed`                            Training random seed (default:
                                      `1`).

  `--train_img_drop`                  Percentage of training samples
                                      whose image features are zeroed.

  `--train_txt_drop`                  Percentage of training samples
                                      whose text features are zeroed.

  `--lr`                              Learning rate (default: `1e-4`).

  `--batch_size`                      Batch size (default: `512`).

  `--max_num_epoch`                   Training epochs (default: `50`).

  `--no_cuda`                         Forces CPU execution.
  -----------------------------------------------------------------------

### 3.2 MSM methods and projection flags

Use `--use_msm` with **at most one** projection option. The projection
options are mutually exclusive.

  -------------------------------------------------------------------------------------------------
  Command option              Output method name   Initialization   Projection     Geometry at
                                                                    optimized?     initialization
  --------------------------- -------------------- ---------------- -------------- ----------------
  `--msm_uplift`              `MSM_UPLIFT`         Fixed            No             Preserved
                                                   orthogonal-row                  
                                                   isometric Q                               

  `--msm_learn_q`             `MSM_LEARN_Q`        Orthogonal-row Q Yes            Preserved
                                                                                   initially

  `--msm_linear`              `MSM_LINEAR`         Xavier-uniform   Yes            Not guaranteed
                                                   linear weights                  
  -------------------------------------------------------------------------------------------------

#### A. Fixed isometric MSM uplift

``` bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --seed 1
```

The canonical ETF prototypes are expanded using an orthogonal-row
projection Q satisfying (QQ\^`\top`{=tex}=I). Q is frozen during
training, so the expanded prototypes preserve the canonical ETF pairwise
geometry.

#### B. Learnable orthogonal initialization (Learn-Q)

``` bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_learn_q --seed 1
```

Q is initialized with orthonormal rows and then optimized through
backpropagation. ETF geometry is preserved at initialization, but
orthogonality is not constrained during training; the learned projection
can distort that geometry.

#### C. Learnable Xavier initialization (Linear Xavier)

``` bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_linear --seed 1
```

The projection layer is initialized using Xavier uniform initialization
and optimized through backpropagation. It is not initialized as an
orthogonal matrix, so ETF geometry is not guaranteed initially or during
training.

#### E. Canonical MSM (no uplift)

Omit all projection flags and set the embedding dimension to (C-1):

``` bash
python train.py --dataset food101 --fusion hadamard --dim_embed 100 --use_msm --seed 1
python train.py --dataset mm_imdb --fusion hadamard --dim_embed 21 --use_msm --seed 1
python train.py --dataset hateful_memes --fusion hadamard --dim_embed 1 --use_msm --seed 1
```

The class counts are 101, 22, and 2 respectively, giving canonical
dimensions 100, 21, and 1.

### 3.3 Cross-entropy baseline

Do not pass `--use_msm` for a standard CE classifier:

``` bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --seed 1
```

### 3.4 Training-time modality dropout

Training dropout zeroes the selected modality features for a randomly
selected percentage of training samples.

``` bash
# Drop image features for 70% of training samples
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_img_drop 70 --seed 1

# Drop text features for 70% of training samples
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_txt_drop 70 --seed 1
```

Image and text dropout can also be combined.

## 4. Multi-seed experiments

Example: fixed isometric uplift with five training seeds.

Linux/macOS:

``` bash
for seed in 1 2 3 4 5; do
    python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --seed "$seed"
done
```

Windows Command Prompt:

``` bat
for %s in (1 2 3 4 5) do python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --seed %s
```

Replace `--msm_uplift` with `--msm_learn_q`, or `--msm_linear` to run the other variants.

## 5. Checkpoints and output

The training script saves checkpoints under:

``` text
raw_ckpt/{dataset}/{fusion}/{method}/aux{aux_weight}/seed{seed}/dim{dim}/checkpoint_last.pth.tar
best/{dataset}/{fusion}/{method}/aux{aux_weight}/seed{seed}/dim{dim}/best_model.pth.tar
```

The missing-modality benchmark CSV is saved under:

``` text
output/{dataset}/{fusion}/{method}/aux{aux_weight}/seed{seed}/dim{dim}/
```

The method name identifies the run: `CE`, `MSM`, `MSM_UPLIFT`, `MSM_LEARN_Q`, or `MSM_LINEAR`, as applicable.

## 6. Evaluating a saved checkpoint

Training already evaluates the best checkpoint and writes its benchmark
CSV. To evaluate a checkpoint separately:

``` bash
python eval_checkpoint.py food101 PATH_TO_CHECKPOINT --seed 1 --out_csv evaluation.csv
```

General form:

``` bash
python eval_checkpoint.py [food101|mm_imdb|hateful_memes] PATH_TO_CHECKPOINT [--feat_dir PATH] [--thresholds FILE.npy] [--seed N] [--no_cuda] [--out_csv FILE.csv]
```

  -----------------------------------------------------------------------
  Argument                            Description
  ----------------------------------- -----------------------------------
  `dataset`                           Positional dataset: `food101`,
                                      `mm_imdb`, or `hateful_memes`.

  `checkpoint`                        Positional path to a `.pth.tar`
                                      checkpoint.

  `--feat_dir`                        Optional feature-directory
                                      override.

  `--thresholds`                      Optional MM-IMDb per-class
                                      threshold `.npy` file. If omitted,
                                      thresholds are optimized on the
                                      validation set.

  `--seed`                            Evaluation/missingness seed
                                      (default: `42`).

  `--no_cuda`                         Forces CPU evaluation.

  `--out_csv`                         Optional output path for the
                                      missing-modality benchmark CSV.
  -----------------------------------------------------------------------

## 7. Benchmark metrics and missingness

  Dataset         Metrics
  --------------- ---------------------------
  UPMC Food-101   `top1`, `top5`
  MM-IMDb         `f1_macro`, `f1_weighted`
  Hateful Memes   `auroc`, `acc`

CSV missingness columns:

-   `img_missing`: percentage of image modality missing.
-   `text_missing`: percentage of text modality missing.

For a text-missingness evaluation, keep `img_missing = 0` and vary
`text_missing`. For image-missingness evaluation, keep
`text_missing = 0` and vary `img_missing`.

For example, `text_missing = 70` means 70% of text features are zeroed
at evaluation, leaving 30% available.

## 8. Important experimental notes

-   `--use_msm` is required for all MSM configurations.
-   `--msm_uplift`, `--msm_learn_q`, and `--msm_linear` are alternative choices; do not combine them.
-   The three explicit projection flags automatically activate the
    uplift path.
-   Learn-Q is orthogonal only at initialization; the current code does
    not impose an orthogonality constraint during training.
-   Linear Xavier is trainable but does not have an ETF-preserving
    initialization.
-   For fair comparisons, keep dataset splits, fusion, embedding
    dimension, training configuration, evaluation seed, and missingness
    masks consistent.