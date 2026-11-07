# Geometry-Preserving Uplift of Equiangular Tight Frame Prototypes for Robust Multimodal Learning under Missing Modalities

This repository provides training and evaluation scripts for multimodal learning experiments using **MSM** and **MSM Uplift** across three benchmark datasets:

- **UPMC Food-101**
- **MM-IMDb**
- **Hateful Memes**

The implementation supports multiple fusion strategies, embedding dimensions, canonical and uplifted MSM configurations, and modality-dropout experiments to evaluate model robustness under missing or corrupted modalities.

---

## 1. Dataset Preparation

Before running experiments, download the precomputed CLIP embeddings for the required datasets.

| Dataset | Download Link | Expected Directory |
|---|---|---|
| UPMC Food-101 | [Zenodo – Food-101](https://zenodo.org/records/24032786) | `./features_CLIP_Food101` |
| MM-IMDb | [Zenodo – MM-IMDb](https://zenodo.org/records/24033803) | `./features_CLIP_IMDb` |
| Hateful Memes | [Zenodo – Hateful Memes](https://zenodo.org/records/24034415) | `./features_CLIP_HatefulMemes` |


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

**Note:** Ensure that the feature files are placed in the expected directories before starting training.

---

## 2. Environment Setup

Install the required Python dependencies:
Install the CUDA-enabled PyTorch build separately using the official selector: https://pytorch.org/get-started/locally/

# Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

It is recommended to use a dedicated Python virtual environment.

```bash
python -m venv venv
```

Activate the environment:

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Training

All experiments are executed through `train.py`.

### 3.1. General Command Structure

```bash
python train.py \
    --dataset [food101|mm_imdb|hateful_memes] \
    --fusion [linear|hadamard] \
    --dim_embed [1|22|100|512] \
    --use_msm \
    --msm_uplift \
    --seed [1|2|3|4|5]
```

### 3.2. Training Arguments

| Argument | Description |
|---|---|
| `--dataset` | Selects the dataset: `food101`, `mm_imdb`, or `hateful_memes`. |
| `--fusion` | Selects the fusion strategy: `linear` or `hadamard`. |
| `--dim_embed` | Specifies the embedding dimension used by the MSM configuration. |
| `--use_msm` | Enables the MSM CxC-1 configuration. |
| `--msm_uplift` | Enables the Uplifted MSM Cx512 configuration. |
| `--seed` | Sets the random seed for reproducibility. |
| `--train_img_drop` | Specifies the percentage of image modality dropped during training. |
| `--train_txt_drop` | Specifies the percentage of text modality dropped during training. |

**Important:**
- Use `--use_msm` to enable MSM.
- Add `--msm_uplift` to enable uplifted MSM.
- Omit `--msm_uplift` when running canonical MSM.
- Training modality-drop percentages are specified independently for image and text modalities.
- The `--seed` argument can be varied across multiple runs to evaluate reproducibility.

---

## 4. Dataset-Specific Experiments

The following commands provide the recommended configurations for each dataset.

### 4.1. UPMC Food-101

**Canonical MSM**

```bash
python train.py --dataset food101 --fusion hadamard --dim_embed 100 --use_msm --seed 1
```

**Uplifted MSM**

```bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --seed 1
```

**Training with 70% Image Modality Drop**

```bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_img_drop 70 --seed 1
```

**Training with 70% Text Modality Drop**

```bash
python train.py --dataset food101 --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_txt_drop 70 --seed 1
```

### 4.2. MM-IMDb

**Canonical MSM**

```bash
python train.py --dataset mm_imdb --fusion hadamard --dim_embed 22 --use_msm --seed 1
```

**Uplifted MSM**

```bash
python train.py --dataset mm_imdb --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --seed 1
```

**Training with 70% Image Modality Drop**

```bash
python train.py --dataset mm_imdb --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_img_drop 70 --seed 1
```

**Training with 70% Text Modality Drop**

```bash
python train.py --dataset mm_imdb --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_txt_drop 70 --seed 1
```

### 4.3. Hateful Memes

**Canonical MSM**

```bash
python train.py --dataset hateful_memes --fusion hadamard --dim_embed 1 --use_msm --seed 1
```

**Uplifted MSM**

```bash
python train.py --dataset hateful_memes --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --seed 1
```

**Training with 70% Image Modality Drop**

```bash
python train.py --dataset hateful_memes --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_img_drop 70 --seed 1
```

**Training with 70% Text Modality Drop**

```bash
python train.py --dataset hateful_memes --fusion hadamard --dim_embed 512 --use_msm --msm_uplift --train_txt_drop 70 --seed 1
```

---

## 5. Multi-Seed Experiments

For reproducibility, experiments can be repeated using multiple random seeds.

For example, to run the uplifted MSM configuration on Food-101 using five seeds:

```bash
for seed in 1 2 3 4 5; do
    python train.py \
        --dataset food101 \
        --fusion hadamard \
        --dim_embed 512 \
        --use_msm \
        --msm_uplift \
        --seed $seed
done
```

Repeat the same procedure for other datasets, fusion strategies, and embedding dimensions as required.

---

## 6. Experimental Results and Output

Each training run automatically generates a CSV file containing the missing-modality benchmark evaluation results.

All generated CSV files are saved in:

```text
output/
```

The output filenames follow the experiment-specific naming convention and include:

```text
output/**_missing_benchmark_**.csv
```

### 6.1. Evaluation Metrics

The CSV files contain the following columns, depending on the dataset:

| Dataset | Evaluation Metrics |
|---|---|
| UPMC Food-101 | `top1`, `top5` |
| MM-IMDb | `f1_macro`, `f1_weighted` |
| Hateful Memes | `auroc`, `acc` |

The missing-modality configuration is recorded using:

- `img_missing`: Percentage of missing image modality.
- `text_missing`: Percentage of missing text modality.

### 6.2. CSV Structure

**Food-101:**

```csv
img_missing,text_missing,top1,top5
```

**MM-IMDb:**

```csv
img_missing,text_missing,f1_macro,f1_weighted
```

**Hateful Memes:**

```csv
img_missing,text_missing,auroc,acc
```

---

## 7. Mapping Experimental Results to Paper Tables
Note: There's no need to run separate eval script, training script automatically does evaluation on test set. So CSV results are evaluated on unseen test set.
The following mapping explains how to extract the relevant CSV rows for the paper's experimental tables.

### 7.1. Table 1: Text Modality Missingness

For Table 1, use the model trained without additional training-time modality dropout.

Keep the image modality fully available (`img_missing = 0`) and vary text missingness.

| Paper Configuration | `img_missing` | `text_missing` |
|---|---:|---:|
| 100% Text Available| 0 | 0  |
| 90% Text Available | 0 | 10 |
| 70% Text Available | 0 | 30 |
| 50% Text Available | 0 | 50 |
| 30% Text Available | 0 | 70 |
| 10% Text Available | 0 | 90 |

**Interpretation:** `text_missing = 70` means that 70% of the text modality is missing, leaving 30% available.

### 7.2. Table 2: Cross-Modality Missingness

For Table 2, use models trained with modality-specific dropout.

| Paper Configuration | Training Configuration | `img_missing` | `text_missing` |
|---|---|---:|---:|
| 100% Image, 30% Text | Text-drop-70 model | 0 | 70 |
| 30% Image, 100% Text | Image-drop-70 model | 70 | 0 |

**Important:** 
- For the first configuration, use the model trained with `--train_txt_drop 70`.
- For the second configuration, use the model trained with `--train_img_drop 70`.
- Read the corresponding evaluation rows from the generated CSV files.

---