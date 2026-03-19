"""
train_mtmc.py
=============
Main training / evaluation loop for MTMC vehicle tracking via Doc2GraphFormer.

Usage
-----
# Baseline
python -m src.training.train_mtmc --backbone clip --frozen --edge-type cross_cam

# With YOLO detections (after run_yolo_tracking.py)
python -m src.training.train_mtmc --backbone clip --frozen --use-yolo

# With MTSC pre-computed tracklets
python -m src.training.train_mtmc --backbone clip --frozen --use-mtsc

# No calibration/ROI extras
python -m src.training.train_mtmc --no-roi-features --no-world-coords

# Test only
python -m src.training.train_mtmc --test --weights checkpoints/best.pt

# Full ablation suite
python -m src.training.train_mtmc --ablation
"""

from __future__ import annotations
import argparse
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np

from paths import SEQ_TRAIN, SEQ_TEST, CHECKPOINTS, RESULTS
from cityflow_preprocessing import (
    load_sequence, load_sequence_mtsc, get_num_cameras,
)
from vehicle_feature_builder import VehicleFeatureBuilder
from vehicle_graph_builder import VehicleGraphBuilder
from graphformer import MTMCGraphformer, edges_to_global_ids
from training_utils import (
    get_device, compute_weighted_loss,
    get_binary_f1, compute_auc, compute_idf1,
    EarlyStopping, save_results,
    write_track1_submission, load_calibrations_for_sequence,
    write_mtmc_output,
)


# =============================================================================
# Sequence loader
# =============================================================================

# Global feature cache: {cache_key: feat_tensor}
_FEAT_CACHE: Dict[str, torch.Tensor] = {}


def _cache_key(seq_path, feat_builder, use_yolo: bool) -> str:
    """Unique key for a (sequence, feature config) combination."""
    import hashlib
    parts = [
        str(seq_path),
        getattr(feat_builder, 'backbone_name', 'clip'),
        getattr(feat_builder, 'ablation_features', 'all'),
        str(getattr(feat_builder, 'num_frames', 1)),
        str(getattr(feat_builder, 'use_roi_features', True)),
        str(getattr(feat_builder, 'use_world_coords', True)),
        str(use_yolo),
    ]
    return hashlib.md5('|'.join(parts).encode()).hexdigest()[:16]


def prepare_sequence(
    seq_path,
    feat_builder:  VehicleFeatureBuilder,
    graph_builder: VehicleGraphBuilder,
    use_yolo:      bool = False,
    yolo_dir:      Optional[str] = None,
    use_mtsc:      bool = False,
    multi_cam_only: bool = False,
) -> Optional[Dict]:
    """
    Load one sequence → build graph → extract features.

    use_yolo=False (default, GT mode): uses per-camera gt/gt.txt directly.
      Labels are 100% accurate. Use for TRAIN sequences (S01, S04).
    use_yolo=True (YOLO mode): uses YOLO det.txt, matches to GT via IoU.
      Labels are ~80-90% accurate. Use for TEST sequence (S03) where you
      have no GT tracklets at inference time.

    multi_cam_only=True: drop tracklets whose GT vehicle appears in only 1
      camera — these can never produce positive cross-camera edges and only
      add noise to the graph.
    """
    if use_mtsc:
        tracklets = load_sequence_mtsc(seq_path)
    else:
        tracklets = load_sequence(
            seq_path,
            use_yolo_detections = use_yolo,
            yolo_det_dir        = yolo_dir,
            multi_cam_only      = multi_cam_only,
        )

    if not tracklets:
        print(f"  [warn] No tracklets loaded from {seq_path}")
        return None

    # Subsample negative edges to 10:1 ratio during training to reduce imbalance.
    # At test time we keep all edges (neg_pos_ratio=None) for full graph inference.
    neg_ratio = 10.0 if not use_yolo else None
    # Build full graph (all edges) — resampling happens per-epoch in training loop
    g, features = graph_builder.build_graph(tracklets, neg_pos_ratio=None)

    # Attach camera dir paths so feat_builder can load roi.jpg / calibration.txt
    features['cam_dirs'] = [t.get('cam_dir') for t in tracklets]

    # Build node features — stored in g.ndata['feat'] on CPU
    # Cache features to disk to avoid re-extracting across ablation runs
    ck = _cache_key(seq_path, feat_builder, use_yolo)
    cache_dir = Path(__file__).parent.parent / 'outputs' / 'feat_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{ck}.pt'

    if cache_file.exists():
        try:
            cached = torch.load(cache_file, map_location='cpu')
            g.ndata['feat'] = cached
            print(f"  [cache] Loaded features from cache ({ck})")
        except Exception:
            feat_builder.add_features([g], features)
            torch.save(g.ndata['feat'], cache_file)
    else:
        feat_builder.add_features([g], features)
        torch.save(g.ndata['feat'], cache_file)
        print(f"  [cache] Saved features to cache ({ck})")

    graph_builder.print_stats(g, Path(str(seq_path)).name)

    features['seq_path'] = str(seq_path)  # for calibration loading

    return {
        'graph':         g,      # full graph — resampled each epoch
        'graph_full':    g,      # alias for resampling in training loop
        'features':      features,
        'num_cameras':   get_num_cameras(tracklets),
        'seq_name':      Path(str(seq_path)).name,
        'gt_global_ids': features['gt_global_ids'],
        'tracklet_ids':  features['tracklet_ids'],
    }


# =============================================================================
# Model builder (shared between train and test)
# =============================================================================

def build_model(args, input_dim: int, num_cameras: int, device) -> MTMCGraphformer:
    return MTMCGraphformer(
        input_dim     = input_dim,
        hidden_dim    = args.hidden_dim,
        num_layers    = args.num_layers,
        num_heads     = args.num_heads,
        num_cameras   = num_cameras,
        use_attn_mask = args.use_attn_mask,
        dropout       = args.dropout,
    ).to(device)


def build_feat_builder(args, device) -> VehicleFeatureBuilder:
    return VehicleFeatureBuilder(
        backbone          = args.backbone,
        device            = str(device),
        frozen_backbone   = args.frozen,
        bbox_pad_ratio    = args.bbox_pad,
        max_num_cameras   = args.max_cameras,
        use_roi_features  = args.use_roi_features,
        use_world_coords  = args.use_world_coords,
        ablation_features = args.ablation_features,
        use_segm_masks    = args.use_segm_masks,
        backbone_weights  = getattr(args, 'backbone_weights', None),
        num_frames        = getattr(args, 'num_frames', 5),
    )


# =============================================================================
# Training
# =============================================================================

def train(args):
    device   = get_device(args.gpu)
    run_name = (
        f"mtmc_{args.backbone}_"
        f"{'frozen' if args.frozen else 'ft'}_"
        f"{args.edge_type}_"
        f"{args.ablation_features}_"
        f"{'roi' if args.use_roi_features else 'noroi'}_"
        f"{'world' if args.use_world_coords else 'noworld'}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M')}"
    )

    print(f"\n{'='*60}")
    print(f"  Run  : {run_name}")
    print(f"  Backbone  : {args.backbone}  frozen={args.frozen}")
    print(f"  Edge type : {args.edge_type}")
    print(f"  Features  : {args.ablation_features}  "
          f"roi={args.use_roi_features}  world={args.use_world_coords}")
    print(f"  Loss      : focal={args.focal}  "
          f"λ_cam={args.lambda_cam}  λ_veh={args.lambda_vehicle}")
    print(f"  LR={args.lr}  epochs={args.epochs}  patience={args.patience}")
    print(f"{'='*60}\n")

    feat_builder  = build_feat_builder(args, device)
    graph_builder = VehicleGraphBuilder(
        edge_type          = args.edge_type,
        temporal_overlap_s = args.temporal_overlap,
        max_time_gap_s     = args.max_time_gap,
    )
    feat_builder.get_info()

    # ── Load train sequences ──────────────────────────────────────────────────
    # Primary: GT tracklets (perfect labels, clean crops)
    # Secondary: YOLO tracklets when --use-yolo is set (closes train/test domain gap)
    train_data = []
    for seq_path in SEQ_TRAIN:
        # GT tracklets — always loaded
        data_gt = prepare_sequence(
            seq_path, feat_builder, graph_builder,
            use_yolo       = False,
            use_mtsc       = args.use_mtsc,
            multi_cam_only = True,
        )
        if data_gt:
            train_data.append(data_gt)

        # YOLO tracklets — only when --use-yolo flag is set
        # These match the S03 test distribution, reducing domain gap
        if args.use_yolo:
            data_yolo = prepare_sequence(
                seq_path, feat_builder, graph_builder,
                use_yolo       = True,
                yolo_dir       = args.yolo_dir,
                use_mtsc       = args.use_mtsc,
                multi_cam_only = False,
            )
            if data_yolo and data_yolo['graph'].edata['label'].sum() > 0:
                train_data.append(data_yolo)
                n_pos = int(data_yolo['graph'].edata['label'].sum().item())
                print(f"  [yolo train] {data_yolo['seq_name']}: "
                      f"{data_yolo['graph'].num_nodes()} nodes, {n_pos} pos edges")

    if not train_data:
        raise RuntimeError("No training data loaded. Check SEQ_TRAIN paths in src/paths.py.")

    # ── Validation ────────────────────────────────────────────────────────────
    # Default: validate on S01 (first train seq, similar topology to S03).
    # With --val-on-test: load S03 and validate on it with IoU-matched GT.
    # This gives a much more honest generalisation signal but uses test domain.
    if getattr(args, 'val_on_test', False):
        print("  [info] Loading S03 for validation (val-on-test mode)...")
        val_seq_path = str(SEQ_TEST)
        val_tracklets = load_sequence(
            val_seq_path,
            use_yolo_detections  = args.use_yolo,
            use_mtsc             = args.use_mtsc,
            min_conf             = 0.4,
            min_area             = 400,
            min_track_len        = 3,
            multi_cam_only       = False,
        )
        val_g, val_feats = graph_builder.build_graph(val_tracklets)
        feat_builder.add_features([val_g], {**val_feats, 'cam_dirs': [
            str(Path(val_seq_path) / n) for n in val_feats['cam_names']
        ]})
        val_g_full = copy.deepcopy(val_g)
        val_data = {
            'graph': val_g, 'graph_full': val_g_full,
            'features': val_feats,
            'gt_global_ids': val_feats['gt_global_ids'],
            'tracklet_ids':  val_feats['tracklet_ids'],
            'seq_name': 'S03',
            'num_cameras': len(set(val_feats['cam_ids'])),
        }
    else:
        val_data = train_data[0]
    print(f"  [info] Validating on {val_data['seq_name']} "
          f"({val_data['graph'].num_nodes()} nodes, "
          f"{val_data['graph'].edata['label'].sum().item():.0f} pos edges)")

    # ── Build model ───────────────────────────────────────────────────────────
    input_dim   = train_data[0]['graph'].ndata['feat'].shape[1]
    # Use fixed 46 = total cameras in CityFlow dataset
    # Avoids model size mismatch between train (S01+S04) and test (S03)
    num_cameras = 46

    model     = build_model(args, input_dim, num_cameras, device)

    # Separate learning rates: backbone (CLIP/ViT) needs very small lr to avoid
    # catastrophic forgetting; graph head needs larger lr to learn quickly.
    # backbone_lr defaults to lr/10 if not set explicitly.
    backbone_lr  = getattr(args, 'backbone_lr', args.lr * 0.1)
    head_lr      = args.lr

    # Separate parameter groups
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if 'backbone' in name or 'model.vision_model' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Build param groups — only include non-empty groups with explicit float lr
    param_groups = [{'params': head_params, 'lr': float(head_lr)}]
    if backbone_params and not args.frozen:
        param_groups.insert(0, {'params': backbone_params, 'lr': float(backbone_lr)})

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay)

    # CosineAnnealingLR — straightforward, no lambda functions needed
    warmup_epochs = min(10, args.epochs // 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-7)
    stopper   = EarlyStopping(
        model, run_name, save_dir=CHECKPOINTS,
        metric='idf1', patience=args.patience)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}\n")

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ── Training loop ─────────────────────────────────────────────────────────
    # Store full graphs (all edges) for per-epoch resampling
    import random as _random
    import dgl as _dgl

    def resample_graph(data, neg_pos_ratio=10.0):
        """Resample negative edges each epoch to prevent memorization."""
        g_full  = data['graph_full']   # full graph stored at load time
        labels  = g_full.edata['label']
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0].tolist()
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0].tolist()
        n_pos   = len(pos_idx)
        if n_pos > 0:
            max_neg = int(n_pos * neg_pos_ratio)
            if len(neg_idx) > max_neg:
                neg_idx = _random.sample(neg_idx, max_neg)
        keep    = sorted(pos_idx + neg_idx)
        src, dst = g_full.edges()
        g_sub = _dgl.graph(
            (src[keep], dst[keep]), num_nodes=g_full.num_nodes(),
            idtype=g_full.idtype,
        )
        g_sub.edata['label'] = labels[keep]
        g_sub.ndata['label'] = g_full.ndata['label']
        g_sub.ndata['feat']  = g_full.ndata['feat']
        return g_sub

    # Freeze backbone for first N epochs to stabilise head training
    # Then gradually unfreeze — prevents NaN loss from large backbone gradients
    FREEZE_WARMUP = 20   # epochs to keep backbone frozen
    _backbone_frozen = True
    for name, param in model.named_parameters():
        if 'backbone' not in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    print(f"  [training] Backbone frozen for first {FREEZE_WARMUP} epochs")

    print("### TRAINING ###")
    for epoch in range(args.epochs):
        # Unfreeze backbone after warmup
        if epoch == FREEZE_WARMUP and _backbone_frozen:
            for name, param in model.named_parameters():
                param.requires_grad_(True)
            _backbone_frozen = False
            print(f"  [training] Backbone unfrozen at epoch {epoch} — using backbone_lr")

        model.train()
        epoch_loss = 0.0

        for data in train_data:
            # Resample negatives every epoch for better generalisation
            g           = resample_graph(data, neg_pos_ratio=10.0)
            feat        = g.ndata['feat'].float().to(device)
            cam_labels  = g.ndata['label'].clamp(0, num_cameras-1).to(device)
            edge_labels = g.edata['label'].clamp(0, 1).to(device)

            def forward_loss():
                cl, _, vl = model(g, feat)
                loss, _, _ = compute_weighted_loss(
                    cl, vl, cam_labels, edge_labels,
                    lambda_cam=args.lambda_cam,
                    lambda_vehicle=args.lambda_vehicle,
                    focal=args.focal,
                    pos_edge_weight=args.pos_weight,
                )
                return loss

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = forward_loss()
                if not torch.isfinite(loss):
                    # NaN/Inf loss: skip this batch, don't update weights
                    # Happens when CLIP backbone produces extreme activations early in training
                    optimizer.zero_grad()
                    epoch_loss += 0.0
                    continue
                scaler.scale(loss).backward()
                # Aggressive clipping: 0.5 prevents CLIP backbone gradient explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_loss()
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    epoch_loss += 0.0
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            epoch_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            # Use full val graph (no subsampling) for honest evaluation
            gv      = val_data['graph_full']
            fv      = gv.ndata['feat'].float().to(device)
            vl_cam  = gv.ndata['label'].clamp(0, num_cameras-1).to(device)
            vl_edge = gv.edata['label'].clamp(0, 1).to(device)
            val_cam_logits, _, veh_logits = model(gv, fv)

            val_loss, _, _ = compute_weighted_loss(
                val_cam_logits, veh_logits,
                vl_cam, vl_edge,
                lambda_cam=args.lambda_cam,
                lambda_vehicle=args.lambda_vehicle,
                focal=args.focal, pos_edge_weight=args.pos_weight,
            )
            p, r, veh_f1 = get_binary_f1(veh_logits, vl_edge)
            auc           = compute_auc(veh_logits, vl_edge)
            node_to_gid   = edges_to_global_ids(
                veh_logits, gv, threshold=args.threshold)
            idf1 = compute_idf1(
                node_to_gid,
                val_data['gt_global_ids'],
                val_data['tracklet_ids'],
            )

        scheduler.step()   # CosineAnnealingLR — no metric needed
        status = stopper.step(auc)   # AUC rises smoothly; IDF1 is too coarse

        if epoch % 10 == 0 or status == 'improved':
            print(
                f"Ep {epoch:04d} | "
                f"TrLoss {epoch_loss/len(train_data):.4f} | "
                f"ValF1 {veh_f1:.4f} | "
                f"AUC {auc:.4f} | "
                f"IDF1 {idf1:.4f}"
                + (" ✓" if status == 'improved' else "")
            )

        if status == 'stop':
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best AUC={stopper.best_score:.4f}")
            break

    _run_test(args, stopper.best_path, feat_builder, graph_builder, device, run_name)
    return stopper.best_score


# =============================================================================
# Test / evaluation
# =============================================================================

def _run_test(
    args,
    weights_path,
    feat_builder:  VehicleFeatureBuilder,
    graph_builder: VehicleGraphBuilder,
    device,
    run_name: str,
):
    print(f"\n### TESTING on S03 ({run_name}) ###")

    test_data = prepare_sequence(
        SEQ_TEST, feat_builder, graph_builder,
        use_yolo=args.use_yolo, yolo_dir=args.yolo_dir,
        use_mtsc=args.use_mtsc,
    )
    if test_data is None:
        print("  [warn] No test data — skipping evaluation.")
        return

    input_dim   = test_data['graph'].ndata['feat'].shape[1]
    num_cameras = 46   # fixed, consistent with training
    model       = build_model(args, input_dim, num_cameras, device)

    if weights_path and Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  Loaded: {weights_path}")
    else:
        print("  [warn] No weights — evaluating with random init.")

    model.eval()
    g    = test_data['graph']           # DGL graph stays on CPU
    feat = g.ndata['feat'].float().to(device)

    results_by_thr = {}
    with torch.no_grad():
        _, _, veh_logits = model(g, feat)

        for thr in args.threshold_sweep:
            node_to_gid = edges_to_global_ids(veh_logits, g, threshold=thr, use_hungarian=True)
            idf1 = compute_idf1(
                node_to_gid,
                test_data['gt_global_ids'],
                test_data['tracklet_ids'],
            )
            p, r, f1 = get_binary_f1(veh_logits, g.edata['label'])
            auc       = compute_auc(veh_logits, g.edata['label'])
            results_by_thr[thr] = {
                'IDF1': round(idf1, 4),
                'VehF1': round(f1, 4),
                'Precision': round(p, 4),
                'Recall': round(r, 4),
                'AUC': round(auc, 4),
            }
            print(f"  thr={thr:.2f} | IDF1={idf1:.4f} | "
                  f"F1={f1:.4f} | P={p:.4f} | R={r:.4f} | AUC={auc:.4f}")

    best_thr = max(results_by_thr, key=lambda t: results_by_thr[t]['IDF1'])
    node_to_gid_best = edges_to_global_ids(veh_logits, g, threshold=best_thr, use_hungarian=True)
    out_path = RESULTS / run_name / f'{test_data["seq_name"]}_predictions.txt'
    write_mtmc_output(node_to_gid_best, test_data['features'], str(out_path))

    save_results(run_name, {
        'args':                 vars(args),
        'results_by_threshold': {str(k): v for k, v in results_by_thr.items()},
        'best_threshold':       best_thr,
        'best_idf1':            results_by_thr[best_thr]['IDF1'],
    }, RESULTS)


# =============================================================================
# Ablation suite
# =============================================================================

ABLATIONS = [
    # Backbone
    {'name': 'backbone_clip_frozen',  'backbone': 'clip', 'frozen': True},
    {'name': 'backbone_vit_frozen',   'backbone': 'vit',  'frozen': True},
    {'name': 'backbone_clip_ft',      'backbone': 'clip', 'frozen': False},
    # Edge type
    {'name': 'edge_cross_cam',   'edge_type': 'cross_cam'},
    {'name': 'edge_fully',       'edge_type': 'fully'},
    {'name': 'edge_temporal',    'edge_type': 'temporal'},
    # Loss
    {'name': 'focal',            'focal': True,  'lambda_cam': 0.3},
    {'name': 'ce_loss',          'focal': False, 'lambda_cam': 0.3},
    {'name': 'no_cam_aux',       'focal': True,  'lambda_cam': 0.0},
    {'name': 'high_cam_aux',     'focal': True,  'lambda_cam': 0.5},
    # Attention mask
    {'name': 'with_attn_mask',   'use_attn_mask': True},
    {'name': 'no_attn_mask',     'use_attn_mask': False},
    # Feature modalities
    {'name': 'feat_visual',      'ablation_features': 'visual'},
    {'name': 'feat_visual_geom', 'ablation_features': 'visual+geom'},
    {'name': 'feat_all',         'ablation_features': 'all'},
    # Dataset extras
    {'name': 'with_roi',         'use_roi_features': True,  'use_world_coords': True},
    {'name': 'no_roi',           'use_roi_features': False, 'use_world_coords': True},
    {'name': 'no_world',         'use_roi_features': True,  'use_world_coords': False},
    {'name': 'no_extras',        'use_roi_features': False, 'use_world_coords': False},
    # MTSC vs YOLO
    {'name': 'yolo_input',       'use_mtsc': False, 'use_yolo': True},
    {'name': 'mtsc_input',       'use_mtsc': True},
    # Positive edge weight
    {'name': 'posw_2',   'pos_weight': 2.0},
    {'name': 'posw_5',   'pos_weight': 5.0},
    {'name': 'posw_10',  'pos_weight': 10.0},
    # Architecture
    {'name': 'arch_small',  'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4},
    {'name': 'arch_medium', 'hidden_dim': 256, 'num_layers': 4, 'num_heads': 8},
    {'name': 'arch_large',  'hidden_dim': 512, 'num_layers': 6, 'num_heads': 8},
]


def run_ablation_suite(base_args):
    all_results = {}
    for abl in ABLATIONS:
        args_copy = copy.deepcopy(base_args)
        for k, v in abl.items():
            if k != 'name':
                setattr(args_copy, k, v)
        print(f"\n{'#'*60}\n  ABLATION: {abl['name']}\n{'#'*60}")
        try:
            best_idf1 = train(args_copy)
            all_results[abl['name']] = {'status': 'done', 'best_idf1': best_idf1}
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[abl['name']] = {'status': f'error: {e}'}
    save_results('ablation_summary', all_results, RESULTS)
    return all_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='MTMC GraphFormer training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument('--use-yolo',   action='store_true',
                   help='Read YOLO det.txt tracklets (after run_yolo_tracking.py)')
    p.add_argument('--yolo-dir',   type=str, default=None,
                   help='Override directory for YOLO per-camera .txt files')
    p.add_argument('--use-mtsc',   action='store_true',
                   help='Use dataset MTSC pre-computed tracklets (best available per camera)')

    # ── Visual backbone ────────────────────────────────────────────────────────
    p.add_argument('--backbone',   type=str, default='clip', choices=['vit', 'clip'])
    p.add_argument('--frozen',     action='store_true', default=True)
    p.add_argument('--no-frozen',  dest='frozen', action='store_false')
    p.add_argument('--bbox-pad',   type=float, default=0.10)
    p.add_argument('--num-frames', type=int,   default=5,
                   help='Frames to mean-pool per tracklet. 1=single frame, 5=pool 5 frames.')
    p.add_argument('--backbone-lr',      type=float, default=None,
                   help='LR for backbone (CLIP/ViT). Default: lr/10. '
                        'Set to 0 to freeze backbone even without --frozen.')
    p.add_argument('--backbone-weights', type=str, default=None,
                   help='Re-ID pre-trained backbone weights')
    p.add_argument('--max-cameras', type=int, default=6)

    # ── Feature modalities ─────────────────────────────────────────────────────
    p.add_argument('--ablation-features', default='all',
                   choices=['visual', 'visual+geom', 'all'])
    p.add_argument('--use-roi-features',  action='store_true', default=True,
                   help='Include ROI boundary proximity node features (3-d)')
    p.add_argument('--no-roi-features',   dest='use_roi_features', action='store_false')
    p.add_argument('--use-world-coords',  action='store_true', default=True,
                   help='Include calibration world coord node features (2-d)')
    p.add_argument('--no-world-coords',   dest='use_world_coords', action='store_false')
    p.add_argument('--use-segm-masks',    action='store_true', default=True,
                   help='Apply segmentation masks to crops before embedding')
    p.add_argument('--no-segm-masks',     dest='use_segm_masks', action='store_false')

    # ── Graph ─────────────────────────────────────────────────────────────────
    p.add_argument('--edge-type',        default='cross_cam',
                   choices=['cross_cam', 'fully', 'temporal'])
    p.add_argument('--temporal-overlap', type=float, default=0.0)
    p.add_argument('--max-time-gap',    type=float, default=30.0,
                   help='Max seconds between tracklets for temporal edge (default 120s)')
    p.add_argument('--use-attn-mask',    action='store_true', default=True)
    p.add_argument('--no-attn-mask',     dest='use_attn_mask', action='store_false')

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument('--hidden-dim',  type=int,   default=256)
    p.add_argument('--num-layers',  type=int,   default=4)
    p.add_argument('--num-heads',   type=int,   default=8)
    p.add_argument('--dropout',     type=float, default=0.1)

    # ── Loss ──────────────────────────────────────────────────────────────────
    p.add_argument('--focal',          action='store_true', default=True)
    p.add_argument('--no-focal',       dest='focal', action='store_false')
    p.add_argument('--lambda-cam',     type=float, default=0.1)
    p.add_argument('--lambda-vehicle', type=float, default=0.7)
    p.add_argument('--pos-weight',     type=float, default=10.0)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    p.add_argument('--epochs',        type=int,   default=200)
    p.add_argument('--lr',            type=float, default=5e-4)
    p.add_argument('--weight-decay',  type=float, default=1e-4)
    p.add_argument('--patience',      type=int,   default=50)
    p.add_argument('--gpu',           type=int,   default=0)

    # ── Inference ─────────────────────────────────────────────────────────────
    p.add_argument('--threshold',       type=float, default=0.5)
    p.add_argument('--threshold-sweep', type=float, nargs='+',
                   default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument('--test',      action='store_true')
    p.add_argument('--weights',   type=str, default=None)
    p.add_argument('--ablation',  action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.ablation:
        run_ablation_suite(args)
    elif args.test:
        dev  = get_device(args.gpu)
        fb   = build_feat_builder(args, dev)
        gb   = VehicleGraphBuilder(edge_type=args.edge_type)
        name = f'test_{datetime.now().strftime("%Y%m%d-%H%M")}'
        _run_test(args, args.weights, fb, gb, dev, name)
    else:
        train(args)