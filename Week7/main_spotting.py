#!/usr/bin/env python3
"""
File containing the main training script.
W&B integration added for ablation study logging.
"""

# Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

# W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed — logging disabled.")

# Local imports
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model


# ── Class names in order (index 0 = class 1 = PASS, etc.) ──────────────────
CLASS_NAMES = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT",
    "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]
# AP10 excludes FREE KICK (idx 10) and GOAL (idx 11)
AP10_MASK = [i for i in range(12) if i not in (10, 11)]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def update_args(args, config):
    args.frame_dir          = config['frame_dir']
    args.save_dir           = config['save_dir'] + '/' + args.model
    args.store_dir          = config['save_dir'] + '/' + "splits"
    args.labels_dir         = config['labels_dir']
    args.store_mode         = config['store_mode']
    args.task               = config['task']
    args.batch_size         = config['batch_size']
    args.clip_len           = config['clip_len']
    args.dataset            = config['dataset']
    args.epoch_num_frames   = config['epoch_num_frames']
    args.feature_arch       = config['feature_arch']
    args.learning_rate      = config['learning_rate']
    args.num_classes        = config['num_classes']
    args.num_epochs         = config['num_epochs']
    args.warm_up_epochs     = config['warm_up_epochs']
    args.only_test          = config['only_test']
    args.device             = config['device']
    args.num_workers        = config['num_workers']

    # ── Optional fields with safe defaults ──────────────────────────────────
    args.temporal_arch              = config.get('temporal_arch', 'none')
    args.lstm_hidden_dim            = config.get('lstm_hidden_dim', 512)
    args.lstm_num_layers            = config.get('lstm_num_layers', 1)
    args.tcn_num_layers             = config.get('tcn_num_layers', 3)
    args.tcn_kernel_size            = config.get('tcn_kernel_size', 3)
    args.transformer_nhead          = config.get('transformer_nhead', 4)
    args.transformer_num_layers     = config.get('transformer_num_layers', 2)
    args.transformer_dim_feedforward= config.get('transformer_dim_feedforward', 1024)
    args.loss                       = config.get('loss', 'ce')
    args.focal_gamma                = config.get('focal_gamma', 2.0)
    args.wandb_project              = config.get('wandb_project', 'mcv-c6-bas')
    args.wandb_entity               = config.get('wandb_entity', None)

    # BiGRU bottleneck params (used by UNet head and standalone BiGRU)
    args.gru_hidden_dim             = config.get('gru_hidden_dim', 512)
    args.gru_num_layers             = config.get('gru_num_layers', 2)
    args.gru_dropout                = config.get('gru_dropout', 0.1)

    # UNet-head params (W7)
    args.unet_depth                 = config.get('unet_depth', 2)
    args.unet_hidden_dim            = config.get('unet_hidden_dim', None)
    args.unet_bottleneck            = config.get('unet_bottleneck', 'bigru')
    args.unet_gru_hidden            = config.get('unet_gru_hidden', 256)
    args.unet_gru_layers            = config.get('unet_gru_layers', 2)
    args.unet_dropout               = config.get('unet_dropout', 0.1)

    # TGLS params (W7-3A)
    args.tgls_enabled               = config.get('tgls_enabled', False)
    args.tgls_sigma                 = config.get('tgls_sigma', 0.55)
    args.tgls_window                = config.get('tgls_window', 5)

    # Stride / overlap
    args.stride                     = config.get('stride', 2)
    args.overlap                    = config.get('overlap', 0.9)

    # Checkpoint selection strategy: 'val_loss' or 'ap10'
    args.ckpt_select                = config.get('ckpt_select', 'val_loss')

    # Evaluation tolerances (W7-2): coarser 1s + tighter 0.5s
    args.eval_tolerances            = tuple(config.get('eval_tolerances', [1.0, 0.5]))

    return args


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
    ])


def compute_ap10(ap_array):
    """Mean AP over the 10 classes (excluding FREE KICK and GOAL)."""
    return float(np.mean([ap_array[i] for i in AP10_MASK]))


def main(args):
    # ── Seed ────────────────────────────────────────────────────────────────
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # ── W&B init ────────────────────────────────────────────────────────────
    run = wandb.init( # noqa
if WANDB_AVAILABLE:
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.model,
        config={
            'model':                    args.model,
            'seed':                     args.seed,
            'feature_arch':             args.feature_arch,
            'temporal_arch':            args.temporal_arch,
            'loss':                     args.loss,
            'focal_gamma':              args.focal_gamma,
            'learning_rate':            args.learning_rate,
            'batch_size':               args.batch_size,
            'clip_len':                 args.clip_len,
            'num_epochs':               args.num_epochs,
            'warm_up_epochs':           args.warm_up_epochs,
            'lstm_hidden_dim':          args.lstm_hidden_dim,
            'lstm_num_layers':          args.lstm_num_layers,
            'tcn_num_layers':           args.tcn_num_layers,
            'tcn_kernel_size':          args.tcn_kernel_size,
            'transformer_nhead':        args.transformer_nhead,
            'transformer_num_layers':   args.transformer_num_layers,
        }
    )

    # ── Dirs ─────────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets stored. Re-run with "store_mode": "load".')
        sys.exit()
    else:
        print('Datasets loaded correctly.')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Model(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # Log model param count
    total_params = sum(p.numel() for p in model._model.parameters())
    if WANDB_AVAILABLE:
        wandb.config.update({'total_params': total_params})
    print(f'Total model parameters: {total_params:,}')

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

        losses = []
        best_criterion = float('inf')
        epoch = 0

        print('START TRAINING EPOCHS')
        for epoch in range(num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            val_loss = model.epoch(val_loader)

            # ── Checkpoint selection (val_loss by default; AP10 on val
            # requires a val-video dataset — add later if needed) ─────────
            better = False
            current_ap10 = None
            if val_loss < best_criterion:
                best_criterion = val_loss
                better = True

            log_line = '[Epoch {}] Train loss: {:0.5f}  Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss)
            if current_ap10 is not None:
                log_line += '  AP10: {:0.3f}'.format(current_ap10)
            if better:
                log_line += '  *** best ***'
            print(log_line)

            # Get current LR from scheduler
            current_lr = optimizer.param_groups[0]['lr']

            # ── W&B per-epoch log ────────────────────────────────────────────
            wandb_epoch = {
                'epoch':         epoch,
                'train/loss':    train_loss,
                'val/loss':      val_loss,
                'lr':            current_lr,
                'best_criterion': best_criterion,
            }
            if current_ap10 is not None:
                wandb_epoch['val/AP10'] = current_ap10
            if WANDB_AVAILABLE:
                wandb.log(wandb_epoch)

            losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss,
                           'ap10': current_ap10})

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
                if better:
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, 'checkpoint_best.pt'))

    # ── Inference ──────────────────────────────────────────────────────────
    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    tolerances = args.eval_tolerances
    _, _, per_tol_results = evaluate(
        model, test_data, nms_window=5, tolerances=tolerances)

    # ── Per-tolerance tables + W&B logging ─────────────────────────────────
    results = {'model': args.model, 'tolerances': {}, 'per_class': {}}
    wandb_log = {}

    for tol in tolerances:
        key = f'tol_{tol}s'
        map_score, ap_score = per_tol_results[key]
        ap10 = compute_ap10(ap_score) * 100
        ap12 = map_score * 100

        print(f'\n===== Tolerance δ = {tol}s =====')
        table = []
        for i, class_name in enumerate(classes.keys()):
            ap_val = ap_score[i] * 100
            table.append([class_name, f"{ap_val:.2f}"])
        print(tabulate(table, ["Class", "Average Precision"], tablefmt="grid"))
        print(tabulate([
            ["AP10 (no FREE KICK / GOAL)", f"{ap10:.2f}"],
            ["AP12 (all classes)",         f"{ap12:.2f}"],
        ], ["Metric", "Value"], tablefmt="grid"))

        tol_tag = f'{tol}s'.replace('.', '_')
        wandb_log[f'test/AP10_{tol_tag}'] = ap10
        wandb_log[f'test/AP12_{tol_tag}'] = ap12
        for i, class_name in enumerate(classes.keys()):
            wandb_log[f'test/AP_{tol_tag}/{class_name.replace(" ", "_")}'] = ap_score[i] * 100

        results['tolerances'][key] = {'AP10': round(ap10, 4), 'AP12': round(ap12, 4)}
        results['per_class'][key] = {
            name: round(float(ap_score[i]) * 100, 4)
            for i, name in enumerate(classes.keys())
        }

    # Primary metrics are those at the first tolerance (1.0s by default)
    primary_key = f'tol_{tolerances[0]}s'
    primary_map, primary_ap = per_tol_results[primary_key]
    wandb_log['test/AP10'] = compute_ap10(primary_ap) * 100
    wandb_log['test/AP12'] = primary_map * 100
    results['AP10'] = round(compute_ap10(primary_ap) * 100, 4)
    results['AP12'] = round(primary_map * 100, 4)

    if WANDB_AVAILABLE:
        wandb.log(wandb_log)
    store_json(os.path.join(args.save_dir, 'results.json'), results, pretty=True)
    print(f'Results saved to {args.save_dir}/results.json')

    if WANDB_AVAILABLE:
        wandb.finish()
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())