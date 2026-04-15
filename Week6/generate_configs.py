#!/usr/bin/env python3
"""
generate_configs.py
--------------------
Generates all BiGRU ablation configs using a greedy one-at-a-time strategy.

GREEDY STRATEGY:
  Stage 1: Sweep hidden_dim       (fix everything else to defaults)
  Stage 2: Sweep num_layers       (fix best hidden_dim from stage 1)
  Stage 3: Sweep dropout          (fix best from stages 1+2)
  Stage 4: Sweep learning_rate    (fix best from stages 1+2+3)
  Stage 5: Sweep loss function    (fix best from stages 1+2+3+4)

Usage:
  python generate_configs.py                        # generate with defaults
  python generate_configs.py --best_hidden 512      # after stage 1 finishes
  python generate_configs.py --best_hidden 512 --best_layers 2   # after stage 2
  ... and so on

The script always regenerates ALL configs, so re-run it as you learn best values.
"""

import json
import os
import argparse

# ── Base config (shared across ALL experiments) ──────────────────────────────
BASE = {
    "frame_dir": "/ghome/group06/Ouss/c6/data/soccernetball/frames/398x224",
    "save_dir": "/ghome/group06/Ouss/c6/results/ablations",
    "labels_dir": "/ghome/group06/Ouss/c6/data/SoccerNet/SN-BAS-2025/",
    "store_mode": "load",
    "task": "spotting",
    "batch_size": 4,
    "clip_len": 50,
    "stride": 2,
    "dataset": "soccernetball",
    "epoch_num_frames": 500000,
    "feature_arch": "rny002",
    "num_classes": 12,
    "num_epochs": 20,
    "warm_up_epochs": 3,
    "only_test": False,
    "device": "cuda",
    "num_workers": 4,
    "temporal_arch": "bigru",  # always BiGRU
    "wandb_project": "mcv-c6-bas-bigru",
    "wandb_entity": "am-oussama10-m",
}

# ── Greedy defaults (update these as stages finish) ──────────────────────────
DEFAULTS = {
    "hidden_dim": 512,
    "num_layers": 1,
    "dropout": 0.1,
    "learning_rate": 0.0008,
    "loss": "ce",
}


def make_config(name, overrides):
    cfg = {**BASE}
    cfg.update(DEFAULTS)
    cfg.update(overrides)
    cfg["gru_hidden_dim"] = cfg.pop("hidden_dim")
    cfg["gru_num_layers"] = cfg.pop("num_layers")
    cfg["gru_dropout"] = cfg.pop("dropout")
    return name, cfg


def generate_all(best):
    """
    best: dict with keys hidden_dim, num_layers, dropout, learning_rate, loss
    These represent the best values found so far.
    """
    experiments = []

    # ── STAGE 1: Hidden dim ──────────────────────────────────────────────────
    for hd in [256, 512, 1024]:
        name = f"bigru_hd{hd}_l1_dr01_lr8e4_ce"
        experiments.append(
            make_config(
                name,
                {
                    "hidden_dim": hd,
                    "num_layers": 1,
                    "dropout": 0.1,
                    "learning_rate": 0.0008,
                    "loss": "ce",
                    "stage": 1,
                    "ablation_var": "hidden_dim",
                },
            )
        )

    # ── STAGE 2: Num layers (uses best hidden_dim) ───────────────────────────
    for nl in [1, 2, 3]:
        name = f"bigru_hd{best['hidden_dim']}_l{nl}_dr01_lr8e4_ce"
        if nl == 1:
            # Already covered in stage 1 — skip duplicate if hidden_dim is default
            if best["hidden_dim"] == 512:
                continue
        experiments.append(
            make_config(
                name,
                {
                    "hidden_dim": best["hidden_dim"],
                    "num_layers": nl,
                    "dropout": 0.1,
                    "learning_rate": 0.0008,
                    "loss": "ce",
                    "stage": 2,
                    "ablation_var": "num_layers",
                },
            )
        )

    # ── STAGE 3: Dropout (uses best hidden_dim + num_layers) ─────────────────
    for dr in [0.0, 0.1, 0.3]:
        name = f"bigru_hd{best['hidden_dim']}_l{best['num_layers']}_dr{str(dr).replace('.','')}_lr8e4_ce"
        if dr == 0.1 and best["num_layers"] == 1 and best["hidden_dim"] == 512:
            continue  # already covered
        experiments.append(
            make_config(
                name,
                {
                    "hidden_dim": best["hidden_dim"],
                    "num_layers": best["num_layers"],
                    "dropout": dr,
                    "learning_rate": 0.0008,
                    "loss": "ce",
                    "stage": 3,
                    "ablation_var": "dropout",
                },
            )
        )

    # ── STAGE 4: Learning rate (uses best from stages 1+2+3) ─────────────────
    for lr in [0.0004, 0.0008, 0.001]:
        lr_tag = str(lr).replace("0.", "").replace(".", "")
        name = f"bigru_hd{best['hidden_dim']}_l{best['num_layers']}_dr{str(best['dropout']).replace('.','')}_lr{lr_tag}_ce"
        if lr == 0.0008 and best["dropout"] == 0.1:
            continue  # already covered
        experiments.append(
            make_config(
                name,
                {
                    "hidden_dim": best["hidden_dim"],
                    "num_layers": best["num_layers"],
                    "dropout": best["dropout"],
                    "learning_rate": lr,
                    "loss": "ce",
                    "stage": 4,
                    "ablation_var": "learning_rate",
                },
            )
        )

    # ── STAGE 5: Loss function (best config, CE vs Focal) ────────────────────
    for loss in ["ce", "focal"]:
        lr_tag = str(best["learning_rate"]).replace("0.", "").replace(".", "")
        name = f"bigru_hd{best['hidden_dim']}_l{best['num_layers']}_dr{str(best['dropout']).replace('.','')}_lr{lr_tag}_{loss}"
        if loss == "ce":
            continue  # the best CE config is already run in stage 4
        cfg_overrides = {
            "hidden_dim": best["hidden_dim"],
            "num_layers": best["num_layers"],
            "dropout": best["dropout"],
            "learning_rate": best["learning_rate"],
            "loss": "focal",
            "focal_gamma": 2.0,
            "stage": 5,
            "ablation_var": "loss",
        }
        experiments.append(make_config(name, cfg_overrides))

    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best_hidden",
        type=int,
        default=DEFAULTS["hidden_dim"],
        help="Best hidden_dim found in Stage 1",
    )
    parser.add_argument(
        "--best_layers",
        type=int,
        default=DEFAULTS["num_layers"],
        help="Best num_layers found in Stage 2",
    )
    parser.add_argument(
        "--best_dropout",
        type=float,
        default=DEFAULTS["dropout"],
        help="Best dropout found in Stage 3",
    )
    parser.add_argument(
        "--best_lr",
        type=float,
        default=DEFAULTS["learning_rate"],
        help="Best learning rate found in Stage 4",
    )
    parser.add_argument(
        "--output_dir", type=str, default="config", help="Where to write JSON configs"
    )
    args = parser.parse_args()

    best = {
        "hidden_dim": args.best_hidden,
        "num_layers": args.best_layers,
        "dropout": args.best_dropout,
        "learning_rate": args.best_lr,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    experiments = generate_all(best)

    names = []
    for name, cfg in experiments:
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=4)
        names.append(name)
        print(f"  Written: {path}")

    # Also write the experiment list for the SLURM launcher
    list_path = os.path.join(args.output_dir, "experiment_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"\nExperiment list: {list_path}")
    print(f"Total experiments: {len(names)}")

    # Print the stage breakdown
    print("\nStage breakdown:")
    stages = {}
    for name, cfg in experiments:
        s = cfg.get("stage", "?")
        stages.setdefault(s, []).append(name)
    for s in sorted(stages):
        print(f"  Stage {s} ({len(stages[s])} runs): {', '.join(stages[s])}")


if __name__ == "__main__":
    main()
