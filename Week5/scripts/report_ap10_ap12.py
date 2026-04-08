#!/usr/bin/env python3
"""
Run test-set inference and report per-class AP, AP12, and AP10.

AP12: mean AP over all 12 classes (standard metric).
AP10: mean AP over the first 10 classes (excludes FREE KICK and GOAL, indices 10-11
      in 0-based class order from data/soccernetball/class.txt).

With --mask_rare_inference, zero out predicted scores for classes 10-11 before AP
computation (simulates ignoring rare classes at inference time).

Usage (from C6_Week5/starter/):
    python ../scripts/report_ap10_ap12.py --model baseline_c6
"""
from __future__ import annotations
import argparse
import os
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Config name, e.g. baseline_c6")
    parser.add_argument(
        "--mask_rare_inference", action="store_true",
        help="Zero predicted scores for class indices 10-11 before computing AP"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Must be run from starter/ so relative paths (data/, config/) resolve correctly
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    starter_dir = os.path.join(repo_root, "starter")
    if not os.path.isdir(os.path.join(starter_dir, "config")):
        print(f"ERROR: starter/config not found. Run this script from C6_Week5/starter/ or adjust paths.")
        sys.exit(1)
    os.chdir(starter_dir)
    sys.path.insert(0, starter_dir)

    import numpy as np
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from sklearn.metrics import average_precision_score
    from tabulate import tabulate

    from util.io import load_json
    from util.dataset import load_classes
    from dataset.datasets import get_datasets
    from model.model_classification import Model

    config = load_json(os.path.join("config", args.model + ".json"))

    model_args = types.SimpleNamespace(
        frame_dir=config["frame_dir"],
        save_dir=os.path.join(config["save_dir"], args.model),
        store_dir=os.path.join(config["save_dir"], "splits"),
        labels_dir=config["labels_dir"],
        store_mode="load",
        task=config["task"],
        batch_size=config["batch_size"],
        clip_len=config["clip_len"],
        dataset=config["dataset"],
        epoch_num_frames=config["epoch_num_frames"],
        feature_arch=config["feature_arch"],
        learning_rate=config["learning_rate"],
        num_classes=config["num_classes"],
        num_epochs=config["num_epochs"],
        warm_up_epochs=config["warm_up_epochs"],
        only_test=True,
        device=config["device"],
        num_workers=config["num_workers"],
        seed=args.seed,
    )
    if "stride" in config:
        model_args.stride = config["stride"]
    if "overlap" in config:
        model_args.overlap = config["overlap"]

    classes, _, _, test_data = get_datasets(model_args)

    model = Model(args=model_args)
    ckpt_path = os.path.join(model_args.save_dir, "checkpoints", "checkpoint_best.pt")
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model.load(torch.load(ckpt_path, map_location=model_args.device))

    # Inference
    scores_list, labels_list = [], []
    for clip in tqdm(DataLoader(
        test_data, num_workers=model_args.num_workers, pin_memory=True,
        batch_size=model_args.batch_size
    )):
        pred = model.predict(clip["frame"])
        scores_list.append(pred)
        labels_list.append(clip["label"].numpy())

    scores = np.concatenate(scores_list, axis=0)  # (N, 12)
    labels = np.concatenate(labels_list, axis=0)  # (N, 12)

    if args.mask_rare_inference:
        scores[:, 10] = 0.0
        scores[:, 11] = 0.0

    ap = average_precision_score(labels, scores, average=None)

    class_names = list(classes.keys())
    table = [[name, f"{ap[i]*100:.2f}"] for i, name in enumerate(class_names)]
    print(tabulate(table, headers=["Class", "AP (%)"], tablefmt="grid"))

    ap12 = float(np.mean(ap))
    ap10 = float(np.mean(ap[:10]))

    summary = [
        ["AP12 (all 12 classes)", f"{ap12*100:.2f}"],
        ["AP10 (first 10 classes)", f"{ap10*100:.2f}"],
    ]
    print(tabulate(summary, tablefmt="grid"))


if __name__ == "__main__":
    main()
