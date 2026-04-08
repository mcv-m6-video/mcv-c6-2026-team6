#!/usr/bin/env python3
"""
main_classification.py  —  upgraded training script for MCV C6 / Task 1 BAC

Changes vs baseline
───────────────────
- Computes per-class pos_weight from training labels and passes to model.epoch()
- Passes pos_weight only when loss_type == 'weighted'
- Everything else identical to baseline (config-driven)
"""

# Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

# Local imports
from util.io import load_json, store_json
from util.eval_classification import evaluate
from dataset.datasets import get_datasets
from model.model_classification import Model


def _repo_root() -> str:
    # starter/ is inside the project root (one level up)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _fix_path(p: str, base: str) -> str:
    """
    Normalize config paths across machines.

    - Expands env vars and ~
    - Rewrites known old absolute roots into this checkout's repo root
    - Resolves relative paths against repo root
    """
    if not isinstance(p, str) or not p:
        return p

    p2 = os.path.expandvars(os.path.expanduser(p))

    old_roots = (
        "/home/kunal/Downloads/C6_W4",
        "/data/113-2/users/kpurkayastha/MCV/C6_Week5",
    )
    for old in old_roots:
        if p2.startswith(old + "/") or p2 == old:
            suffix = p2[len(old):].lstrip("/")
            return os.path.join(base, suffix) if suffix else base

    if not os.path.isabs(p2):
        return os.path.join(base, p2)

    return p2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str, required=True)
    parser.add_argument('--seed',   type=int, default=1)
    return parser.parse_args()


def update_args(args, config):
    base = _repo_root()
    frame_dir_cfg = _fix_path(config['frame_dir'], base)
    save_dir_cfg = _fix_path(config['save_dir'], base)
    labels_dir_cfg = _fix_path(config['labels_dir'], base)

    args.frame_dir        = frame_dir_cfg
    args.save_dir         = os.path.join(save_dir_cfg, args.model)
    args.store_dir        = os.path.join(save_dir_cfg, 'splits')
    args.labels_dir       = labels_dir_cfg
    args.store_mode       = config['store_mode']
    args.task             = config['task']
    args.batch_size       = config['batch_size']
    args.clip_len         = config['clip_len']
    args.dataset          = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch     = config['feature_arch']
    args.learning_rate    = config['learning_rate']
    args.num_classes      = config['num_classes']
    args.num_epochs       = config['num_epochs']
    args.warm_up_epochs   = config['warm_up_epochs']
    args.only_test        = config['only_test']
    args.device           = config['device']
    args.num_workers      = config['num_workers']

    if 'stride'  in config: args.stride  = config['stride']
    if 'overlap' in config: args.overlap = config['overlap']

    # New optional config keys (all have safe defaults in model code)
    args.temporal_arch  = config.get('temporal_arch',  'maxpool')
    args.loss_type      = config.get('loss_type',      'bce')
    args.use_tta        = config.get('use_tta',        False)
    args.label_smooth   = config.get('label_smooth',   0.0)
    args.grad_checkpoint = config.get('grad_checkpoint', False)
    args.focal_gamma    = config.get('focal_gamma',    2.0)
    args.focal_alpha    = config.get('focal_alpha',    0.25)
    args.label_source   = config.get('label_source',   'ball')  # "ball", "v2", or "both"
    args.label_map      = config.get('label_map',      None)    # optional dict raw_label->target_label

    return args


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('LR schedule: Linear Warmup ({}) + Cosine Annealing ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
                          num_steps_per_epoch * cosine_epochs),
    ])


def compute_pos_weight(train_data, num_classes, max_weight=50.0):
    """
    Scan all stored training labels to compute per-class pos_weight
    (= neg_count / pos_count).  Capped at max_weight to avoid explosion
    on extremely rare classes.
    """
    pos = np.zeros(num_classes, dtype=np.float64)
    neg = np.zeros(num_classes, dtype=np.float64)
    for labels_list in train_data._labels_store:
        label_vec = np.zeros(num_classes, dtype=np.float64)
        for lbl in labels_list:
            label_vec[lbl['label'] - 1] = 1.0
        pos += label_vec
        neg += (1.0 - label_vec)
    # Avoid division by zero for classes with 0 positives
    pos = np.maximum(pos, 1.0)
    weight = np.minimum(neg / pos, max_weight)
    print('pos_weight per class:', np.round(weight, 1))
    return torch.tensor(weight, dtype=torch.float32)


def main(args):
    print('Seed:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = os.path.join('config', args.model + '.json')
    config = load_json(config_path)
    args   = update_args(args, config)

    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets stored. Re-run with store_mode=load.')
        sys.exit(0)
    else:
        print('Datasets loaded.')

    # Compute pos_weight for weighted BCE
    pos_weight = None
    if args.loss_type == 'weighted':
        pos_weight = compute_pos_weight(train_data, args.num_classes)

    def worker_init_fn(worker_id):
        random.seed(worker_id + epoch * 100)

    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn)

    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn)

    model = Model(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)

        losses = []
        best_criterion = float('inf')
        epoch = 0

        print('START TRAINING')
        for epoch in range(num_epochs):
            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler,
                pos_weight=pos_weight)

            val_loss = model.epoch(val_loader, pos_weight=pos_weight)

            better = val_loss < best_criterion
            if better:
                best_criterion = val_loss

            print('[Epoch {:3d}] train={:.5f}  val={:.5f}{}'.format(
                epoch, train_loss, val_loss,
                '  ← best' if better else ''))

            losses.append({'epoch': epoch,
                           'train': train_loss, 'val': val_loss})

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'),
                           losses, pretty=True)
                if better:
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, 'checkpoint_best.pt'))

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    ap_score = evaluate(model, test_data)

    table = [[name, f'{ap_score[i]*100:.2f}']
             for i, name in enumerate(classes.keys())]
    print(tabulate(table, ['Class', 'Average Precision'], tablefmt='grid'))

    ap10 = float(np.mean(ap_score[:10]))
    ap12 = float(np.mean(ap_score))
    summary = [['AP12 (all)', f'{ap12*100:.2f}'],
               ['AP10 (excl. FREEKICK+GOAL)', f'{ap10*100:.2f}']]
    print(tabulate(summary, tablefmt='grid'))

    print('DONE')


if __name__ == '__main__':
    main(get_args())
