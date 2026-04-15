# C6 Week 6 — Ball Action Spotting (SN-BAS-2025)

> **Project 2, Task 2 — Group 06**
> Action spotting on the [SoccerNet Ball Action Spotting 2025 (SN-BAS-2025)](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025) dataset.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
  - [Step 1 – Store Clip Features (first run)](#step-1--store-clip-features-first-run)
  - [Step 2 – Train & Evaluate](#step-2--train--evaluate)
  - [Step 3 – Compute MACs / Parameter Count](#step-3--compute-macs--parameter-count)
  - [Download Pre-trained Weights](#download-pre-trained-weights)
- [Available Models & Experiments](#available-models--experiments)
- [Ablation Study](#ablation-study)
- [Results](#results)

---

## Overview

This repository contains the starter code and extended experiments for **Task 2: Ball Action Spotting** in the C6 Video Analysis module. Given a sequence of video frames from soccer matches, the model must detect which of 12 ball-action classes occurs and **at which exact frame** (spotting).

Key features implemented on top of the baseline:

- **Temporal encoders:** `none` (baseline), `lstm`, `tcn` (dilated causal conv), `bigru` (bidirectional GRU)
- **Loss functions:** `ce` (cross-entropy with class weighting), `focal` (Focal Loss γ=2)
- **Evaluation:** AP10 (primary metric, excludes FREE KICK and GOAL) and AP12 (all classes)
- **Inference:** Non-Maximum Suppression (NMS) over frame-level predictions

---

## Project Structure

```
c6/
├── main_spotting.py            # Entry point: train + evaluate
├── config/                     # JSON configuration files (one per experiment)
│   ├── baseline.json
│   ├── lstm.json
│   ├── tcn.json
│   ├── bigru_hd512_l1_dr01_lr8e4_ce.json
│   ├── bigru_hd512_l2_dr01_lr8e4_ce.json
│   └── ...
├── dataset/
│   ├── datasets.py             # Dataset loaders (train/val/test splits)
│   └── frame.py                # Frame-level reading and clip sampling
├── model/
│   ├── model_spotting.py       # Model definition (backbone + temporal head)
│   ├── bigru_head.py           # BiGRU temporal head
│   ├── temporal_heads.py       # LSTM and TCN temporal heads
│   └── modules.py              # Base classes (FCLayers, step, etc.)
├── util/
│   ├── eval_spotting.py        # mAP evaluation with temporal tolerance
│   ├── dataset.py              # Dataset utility functions
│   └── io.py                   # JSON I/O helpers
├── scripts/
│   ├── aggregate_bigru.py      # Print ablation results table
│   └── count_macs.py           # Measure FLOPs and parameter count
├── slurm/
│   └── job_bigru.sh            # SLURM job template
├── generate_configs.py         # Generate all ablation configs automatically
├── run_bigru_ablations.sh      # Submit ablation jobs (skips finished ones)
└── data/
    └── soccernetball/
        ├── class.txt
        ├── train.json
        ├── val.json
        └── test.json
```

---

## Requirements

- Python 3.11
- CUDA-capable GPU (≥ 8GB VRAM recommended, experiments used 24GB)
- Conda (recommended)

Key Python packages:

| Package | Version |
|---|---|
| torch | 2.3.1 |
| torchvision | 0.18.1 |
| timm | 1.0.3 |
| numpy | 1.26.4 |
| SoccerNet | ≥ 0.1.7 |
| tabulate | ≥ 0.9.0 |
| wandb | ≥ 0.16.0 |
| ptflops | ≥ 0.7 |

---

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate c6
```

### 2. Install PyTorch with the correct CUDA wheel

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### 1. Download the SN-BAS-2025 dataset

```bash
huggingface-cli download SoccerNet/SN-BAS-2025 \
    --repo-type dataset \
    --local-dir data/SoccerNet/SN-BAS-2025
```

> **Note:** Access requires accepting the SoccerNet NDA.

### 2. Extract video frames (one-time, takes a while)

```bash
python extract_frames_snb.py \
    --video_dir data/SoccerNet/SN-BAS-2025 \
    -o data/soccernetball/frames/398x224
```

---

## Configuration

All experiment settings live in `config/<model_name>.json`. Update these paths before running:

| Parameter | Description |
|---|---|
| `frame_dir` | Path to extracted frames |
| `save_dir` | Path for checkpoints and logs |
| `labels_dir` | Path to SN-BAS-2025 label files |

Other key parameters:

| Parameter | Description |
|---|---|
| `store_mode` | `"store"` on first run; `"load"` on subsequent runs |
| `feature_arch` | Backbone: `rny002`, `rny004`, `rny008` |
| `temporal_arch` | Temporal head: `none`, `lstm`, `tcn`, `bigru` |
| `loss` | Loss function: `ce`, `focal` |
| `clip_len` | Number of frames per clip (default: 50) |
| `stride` | Frame sampling stride (default: 2) |
| `batch_size` | Training batch size |
| `num_epochs` | Total training epochs (default: 20) |

---

## Running the Code

### Step 1 – Store Clip Features (first run only)

Set `"store_mode": "store"` in your config and run once to cache clips:

```bash
python main_spotting.py --model baseline
```

After it completes, switch to `"store_mode": "load"` for all subsequent runs.

### Step 2 – Train & Evaluate

```bash
# Baseline (no temporal modeling)
python main_spotting.py --model baseline

# LSTM temporal head
python main_spotting.py --model lstm

# TCN temporal head
python main_spotting.py --model tcn

# Best model — BiGRU (hd=512, 2 layers)
python main_spotting.py --model bigru_hd512_l2_dr01_lr8e4_ce
```

On a SLURM cluster:

```bash
sbatch slurm/job_bigru.sh bigru_hd512_l2_dr01_lr8e4_ce
```

### Step 3 – Compute MACs / Parameter Count

```bash
pip install ptflops
python scripts/count_macs.py
```

### Download Pre-trained Weights

The best model checkpoint (**BiGRU, hd=512, 2 layers**) is available for download:

| Resource | Link |
|---|---|
| Config (`.json`) | [⬇ Download from Google Drive](https://drive.google.com/file/d/1CFfQsDhea_EOQesX3AxOBZPI_N-secT5/view?usp=share_link) |
| Checkpoint (`.pt`) | [⬇ Download from Google Drive](https://drive.google.com/file/d/1KhFG1xkFEhU9JlNSdJKVozH-z2iLj2oz/view?usp=share_link) |

To run inference with the downloaded checkpoint:

```bash
# Place the config in config/ and the checkpoint here:
mkdir -p results/ablations/best_model/checkpoints
mv checkpoint_best.pt results/ablations/best_model/checkpoints/

# Set "only_test": true in the config, then:
python main_spotting.py --model best_model
```

---

## Available Models & Experiments

| Config | Description |
|---|---|
| `baseline` | RegNet-Y only, no temporal modeling |
| `lstm` | RegNet-Y + 1-layer LSTM (hd=512) |
| `tcn` | RegNet-Y + 3-layer dilated TCN |
| `bigru_hd256_l1_dr01_lr8e4_ce` | BiGRU (hd=256, 1 layer) |
| `bigru_hd512_l1_dr01_lr8e4_ce` | BiGRU (hd=512, 1 layer) |
| `bigru_hd1024_l1_dr01_lr8e4_ce` | BiGRU (hd=1024, 1 layer) |
| `bigru_hd512_l2_dr01_lr8e4_ce` | BiGRU (hd=512, 2 layers) ← **best** |
| `bigru_hd512_l3_dr01_lr8e4_ce` | BiGRU (hd=512, 3 layers) |

---

## Ablation Study

We used a **greedy one-at-a-time strategy**: sweep one hyperparameter while fixing all others to their best known value.

### Stages

| Stage | Variable | Values tested |
|---|---|---|
| 1 | Hidden dim | 256 / 512 / 1024 |
| 2 | Num layers | 1 / 2 / 3 |
| 3 | Dropout | 0.0 / 0.1 / 0.3 |
| 4 | Learning rate | 4e-4 / 8e-4 / 1e-3 |
| 5 | Loss function | CE / Focal (γ=2) |

### Running experiments

```bash
# Generate all configs
python generate_configs.py

# Submit stage 1 (hidden dim sweep)
bash run_bigru_ablations.sh --stage 1

# After stage 1 finishes, update best value and submit stage 2
python generate_configs.py --best_hidden 512
bash run_bigru_ablations.sh --stage 2

# Monitor jobs
squeue -u $USER

# Check results at any point
python scripts/aggregate_bigru.py
```

The launcher automatically skips experiments that already have results. Use `--dry-run` to preview without submitting.

All runs are tracked on W&B under project `mcv-c6-bas-bigru`.

---

## Results

### Best Model — BiGRU (hd=512, 2 layers)

| Metric | Score |
|---|---|
| **AP10** (primary, excl. FREE KICK & GOAL) | **31.87%** |

> AP10 is the primary evaluation metric as FREE KICK and GOAL have very few training samples, making their AP estimates unreliable.

