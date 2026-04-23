# Soccer Action Spotting

Best-performing variant from the Week-7 on **SoccerNet-Ball Action Spotting 2025**.

| Tolerance | AP10 | mAP12 |
|-----------|------|-------|
| δ = 0.5 s | 32.18 % | 26.82 % |
| δ = 1.0 s | **36.80 %** | **30.67 %** |

The final presentation slides can be found [Here](https://canva.link/8gezkl92xbs429k) and the PDF copy of the same is here in the GitHub

---

## Architecture

```
Video clip (B, T, 3, H, W)
    ↓
X3D-M backbone  →  (B, T, 192)          
    ↓
UNetTemporalHead  (depth=2)
    ├─ Encoder:  T → T/2 → T/4          
    ├─ Bottleneck: BiGRU at T/4         
    └─ Decoder:  T/4 → T/2 → T         
    ↓
FCLayers(192 → 13)                     
    ↓
Soft-CE loss with TGLS                 
```

---

## Setup

```bash
pip install -r requirements.txt
```

> W&B is optional — training runs without it; logging is silently disabled if `wandb` is not installed.

---

## Training

```bash
python main_spotting.py --model unet_d2_x3dm_tgls
```

Trains for 25 epochs (3 warm-up + 22 cosine annealing), saves `checkpoint_best.pt` based on validation AP.

---

## Evaluation

```bash
python eval_model.py --model unet_d2_x3dm_tgls --split test
```

Evaluates at δ = 0.5 s and δ = 1.0 s tolerances. Reports per-class AP, AP10 (excl. FREE KICK & GOAL), and mAP12.

---

## Best Checkpoint

Download [Here](https://drive.google.com/file/d/1L5-xgZv1PEbEE-8iFshcNPggQFvSeDTE/view?usp=sharing)


## Live Demo

Live at HuggingFace:  [kpurkayastha/unet-d2-x3dm-action-spotting](https://huggingface.co/spaces/kpurkayastha/unet-d2-x3dm-action-spotting)

Demo samples:  [Here](https://drive.google.com/drive/folders/14QL_GdDxbAGf7O0AIbm2QwPSDI7ZMgnr?usp=sharing)

