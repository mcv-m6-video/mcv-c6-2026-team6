#!/usr/bin/env python3
"""
eval_model.py  —  W7 version
Loads a trained checkpoint and reports AP10 / AP12 at multiple tolerances
(1.0 s coarser + 0.5 s tighter, per W7-Ex2) on the test (or val) set.

Usage:
    # evaluate the TGLS model using locally downloaded checkpoint
    python3 eval_model.py --model unet_d2_x3dm_tgls \
        --checkpoint "/home/kunal/Downloads/results (1)/results/week7/unet_d2_x3dm_tgls/checkpoints/checkpoint_best.pt"

    # evaluate every model that has a checkpoint (config save_dir paths)
    python3 eval_model.py --model all

    # val split, custom NMS
    python3 eval_model.py --model unet_d2_x3dm_tgls --split val --nms 3
"""

import argparse
import os
import sys
import torch
import numpy as np
from tabulate import tabulate

from util.io import load_json
from util.eval_spotting import evaluate
from util.dataset import load_classes
from dataset.frame import ActionSpotVideoDataset
from model.model_spotting import Model

AP10_EXCLUDE = {'FREE KICK', 'GOAL'}

ALL_MODELS = [
    'baseline',
    'w6_best_bigru',
    'bigru_x3dm',
    'unet_d1_rny002',
    'unet_d2_rny002',
    'unet_d2_rny002_tgls',
    'unet_d2_x3dm',
    'unet_d2_x3dm_tgls',
]

CLASS_NAMES = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT",
    "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]
AP10_MASK = [i for i in range(12) if CLASS_NAMES[i] not in AP10_EXCLUDE]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str, default='unet_d2_x3dm_tgls',
                   help='Config name or "all"')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Override checkpoint path (useful for locally downloaded weights)')
    p.add_argument('--split',      type=str, default='test', choices=['val', 'test'])
    p.add_argument('--nms',        type=int, default=5)
    p.add_argument('--tolerances', type=float, nargs='+', default=[1.0, 0.5],
                   help='Evaluation tolerances in seconds (W7-Ex2)')
    p.add_argument('--seed',       type=int, default=1)
    return p.parse_args()


def build_args_from_config(model_name, seed=1):
    config_path = os.path.join('config', f'{model_name}.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    config = load_json(config_path)

    class Args:
        pass

    a = Args()
    a.model             = model_name
    a.seed              = seed
    a.frame_dir         = config['frame_dir']
    a.save_dir          = config['save_dir'] + '/' + model_name
    a.store_dir         = config['save_dir'] + '/splits'
    a.labels_dir        = config['labels_dir']
    a.store_mode        = 'load'
    a.task              = config['task']
    a.batch_size        = config['batch_size']
    a.clip_len          = config['clip_len']
    a.dataset           = config['dataset']
    a.epoch_num_frames  = config['epoch_num_frames']
    a.feature_arch      = config['feature_arch']
    a.learning_rate     = config['learning_rate']
    a.num_classes       = config['num_classes']
    a.num_epochs        = config['num_epochs']
    a.warm_up_epochs    = config['warm_up_epochs']
    a.only_test         = True
    a.device            = config['device']
    a.num_workers       = config['num_workers']

    # Temporal head
    a.temporal_arch             = config.get('temporal_arch', 'none')
    a.gru_hidden_dim            = config.get('gru_hidden_dim', 512)
    a.gru_num_layers            = config.get('gru_num_layers', 2)
    a.gru_dropout               = config.get('gru_dropout', 0.1)
    a.lstm_hidden_dim           = config.get('lstm_hidden_dim', 512)
    a.lstm_num_layers           = config.get('lstm_num_layers', 1)
    a.tcn_num_layers            = config.get('tcn_num_layers', 3)
    a.tcn_kernel_size           = config.get('tcn_kernel_size', 3)
    a.transformer_nhead         = config.get('transformer_nhead', 4)
    a.transformer_num_layers    = config.get('transformer_num_layers', 2)
    a.transformer_dim_feedforward = config.get('transformer_dim_feedforward', 1024)

    # UNet params (W7)
    a.unet_depth        = config.get('unet_depth', 2)
    a.unet_hidden_dim   = config.get('unet_hidden_dim', None)
    a.unet_bottleneck   = config.get('unet_bottleneck', 'bigru')
    a.unet_gru_hidden   = config.get('unet_gru_hidden', 256)
    a.unet_gru_layers   = config.get('unet_gru_layers', 2)
    a.unet_dropout      = config.get('unet_dropout', 0.1)

    # TGLS (W7-3A)
    a.tgls_enabled      = config.get('tgls_enabled', False)
    a.tgls_sigma        = config.get('tgls_sigma', 0.55)
    a.tgls_window       = config.get('tgls_window', 5)

    # Loss / stride
    a.loss              = config.get('loss', 'ce')
    a.focal_gamma       = config.get('focal_gamma', 2.0)
    a.stride            = config.get('stride', 2)
    a.overlap           = config.get('overlap', 0.9)

    return a


def eval_one(model_name, split, nms_window, tolerances, seed=1, ckpt_override=None):
    try:
        args = build_args_from_config(model_name, seed)
    except FileNotFoundError as e:
        print(f'  [SKIP] {e}')
        return None

    ckpt_path = ckpt_override or os.path.join(args.save_dir, 'checkpoints', 'checkpoint_best.pt')
    if not os.path.exists(ckpt_path):
        print(f'  [SKIP] Checkpoint not found: {ckpt_path}')
        return None

    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_kwargs = {
        'stride': args.stride, 'overlap': 0,
        'dataset': args.dataset, 'labels_dir': args.labels_dir, 'task': args.task,
    }
    split_file = os.path.join('data', args.dataset, f'{split}.json')
    dataset = ActionSpotVideoDataset(classes, split_file, args.frame_dir,
                                     args.clip_len, **dataset_kwargs)

    model = Model(args=args)
    state = torch.load(ckpt_path, map_location=args.device)
    model.load(state)
    print(f'  Loaded: {ckpt_path}')

    _, _, per_tol = evaluate(model, dataset, nms_window=nms_window,
                             tolerances=tuple(tolerances))

    results_by_tol = {}
    class_names = list(classes.keys())
    for tol in tolerances:
        key = f'tol_{tol}s'
        mAP, ap_arr = per_tol[key]
        ap10 = float(np.mean([ap_arr[i] for i in AP10_MASK])) * 100
        ap12 = mAP * 100
        results_by_tol[tol] = (ap10, ap12, ap_arr)

    return results_by_tol, class_names


def print_tables(model_name, results_by_tol, class_names):
    print(f'\n{"="*60}')
    print(f'  Model: {model_name}')
    print(f'{"="*60}')

    for tol, (ap10, ap12, ap_arr) in sorted(results_by_tol.items(), reverse=True):
        print(f'\n  --- Tolerance δ = {tol}s ---')
        rows = []
        for i, cls in enumerate(class_names):
            excl = '  [excl.]' if cls in AP10_EXCLUDE else ''
            rows.append([cls + excl, f'{ap_arr[i]*100:.2f}'])
        print(tabulate(rows, headers=['Class', 'AP (%)'], tablefmt='grid'))
        print(tabulate([
            ['AP10  (excl. FREE KICK & GOAL)', f'{ap10:.2f}'],
            ['AP12  (all 12 classes)',          f'{ap12:.2f}'],
        ], tablefmt='simple'))


def main():
    args = get_args()
    models = ALL_MODELS if args.model == 'all' else [args.model]

    summary_rows = []
    for model_name in models:
        result = eval_one(
            model_name, args.split, args.nms, args.tolerances, args.seed,
            ckpt_override=(args.checkpoint if args.model != 'all' else None),
        )
        if result is None:
            continue

        results_by_tol, class_names = result
        print_tables(model_name, results_by_tol, class_names)

        row = [model_name]
        for tol in sorted(results_by_tol.keys(), reverse=True):
            ap10, ap12, _ = results_by_tol[tol]
            row += [f'{ap10:.2f}', f'{ap12:.2f}']
        summary_rows.append(row)

    if len(summary_rows) > 1:
        tol_headers = []
        for tol in sorted(args.tolerances, reverse=True):
            tol_headers += [f'AP10@{tol}s', f'AP12@{tol}s']
        print(f'\n{"="*60}')
        print(' SUMMARY')
        print(f'{"="*60}')
        print(tabulate(summary_rows, headers=['Model'] + tol_headers, tablefmt='grid'))


if __name__ == '__main__':
    main()
