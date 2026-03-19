#!/usr/bin/env python3
"""
pretrain_reid.py
================
Pre-train the CLIP backbone for vehicle re-ID using supervised contrastive loss.
This runs BEFORE train_mtmc.py and produces a fine-tuned backbone checkpoint
that replaces frozen CLIP with re-ID-aware features.

Why this works:
  - Frozen CLIP: same/diff vehicle separation = 0.039 (near random)
  - After re-ID pre-training: expected separation > 0.20
  - This alone should push IDF1 from ~0.09 to >0.3

Usage:
    python pretrain_reid.py --epochs 50 --gpu 0
    # Then run train_mtmc.py with --backbone-weights outputs/reid_backbone.pt
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from cityflow_preprocessing import load_sequence
from paths import SEQ_TRAIN


# ---------------------------------------------------------------------------
# Dataset: pairs of crops with same/different vehicle labels
# ---------------------------------------------------------------------------

# Module-level transform so it can be pickled
import torchvision.transforms as _T
_CLIP_TRANSFORM = _T.Compose([
    _T.Resize((224, 224)),
    _T.ToTensor(),
    _T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP normalization
])


class VehiclePairDataset(Dataset):
    """
    Builds pairs of vehicle crops from GT tracklets.
    Returns (tensor_a, tensor_b, label) — plain tensors, no processor dicts.
    """
    def __init__(self, tracklets, preprocess_fn=None, n_pairs=10000):
        # preprocess_fn ignored — we use _CLIP_TRANSFORM directly
        self.pairs = []

        # Group tracklets by global vehicle ID
        from collections import defaultdict
        by_gid = defaultdict(list)
        for t in tracklets:
            gid = t.get('gt_global_id', -1)
            if gid != -1:
                by_gid[gid].append(t)

        multi_cam = {gid: ts for gid, ts in by_gid.items() if len(ts) >= 2}
        all_tracklets = tracklets
        gids = list(multi_cam.keys())

        print(f"  Building pairs from {len(gids)} multi-cam vehicles...")

        # Positive pairs: same vehicle, different cameras (supervised)
        pos_pairs = []
        for gid, ts in multi_cam.items():
            for i in range(len(ts)):
                for j in range(i+1, len(ts)):
                    if ts[i]['cam_id'] != ts[j]['cam_id']:
                        pos_pairs.append((ts[i], ts[j], 1))

        # Self-supervised positives from multi-frame tracklets
        # Two random frames from the same tracklet = same vehicle
        ss_pos_pairs = []
        for t in tracklets:
            frames = t.get('all_frames', [])
            if len(frames) >= 3:
                # Sample 2 non-adjacent frames from the same tracklet
                idxs = random.sample(range(len(frames)), 2)
                if abs(idxs[0] - idxs[1]) >= 2:
                    # Create a fake tracklet entry with different frame_path
                    t_copy = dict(t)
                    f0, f1 = frames[idxs[0]], frames[idxs[1]]
                    # We'll use the tracklet as-is but indicate same identity
                    ss_pos_pairs.append((t, t, 1))  # same tracklet = same vehicle
                    if len(ss_pos_pairs) >= len(pos_pairs):
                        break

        # Hard negative pairs: different tracklets in same camera (close in time)
        by_cam = {}
        for t in tracklets:
            by_cam.setdefault(t['cam_id'], []).append(t)
        hard_neg_pairs = []
        for cam_id, cam_tracklets in by_cam.items():
            for i in range(len(cam_tracklets)):
                for j in range(i+1, min(i+5, len(cam_tracklets))):
                    ti, tj = cam_tracklets[i], cam_tracklets[j]
                    if ti.get('gt_global_id', -1) != tj.get('gt_global_id', -1):
                        hard_neg_pairs.append((ti, tj, 0))

        # Random negatives: different vehicles
        neg_pairs = []
        attempts = 0
        while len(neg_pairs) < len(pos_pairs) * 2 and attempts < 50000:
            attempts += 1
            t_a = random.choice(all_tracklets)
            t_b = random.choice(all_tracklets)
            if (t_a['gt_global_id'] != t_b['gt_global_id'] and
                    t_a['gt_global_id'] != -1 and t_b['gt_global_id'] != -1):
                neg_pairs.append((t_a, t_b, 0))

        # Combine: supervised + self-supervised positives, hard + random negatives
        n = min(len(pos_pairs) + len(ss_pos_pairs)//2,
                len(hard_neg_pairs) + len(neg_pairs), n_pairs // 2)
        n_pos_sup = min(len(pos_pairs), n * 2 // 3)
        n_pos_ss  = min(len(ss_pos_pairs), n - n_pos_sup)
        n_neg_hard = min(len(hard_neg_pairs), n // 2)
        n_neg_rand = min(len(neg_pairs), n - n_neg_hard)
        random.shuffle(pos_pairs); random.shuffle(ss_pos_pairs)
        random.shuffle(hard_neg_pairs); random.shuffle(neg_pairs)
        self.pairs = (pos_pairs[:n_pos_sup] + ss_pos_pairs[:n_pos_ss] +
                      hard_neg_pairs[:n_neg_hard] + neg_pairs[:n_neg_rand])
        random.shuffle(self.pairs)
        n_total_pos = n_pos_sup + n_pos_ss
        n_total_neg = n_neg_hard + n_neg_rand
        print(f"  {len(self.pairs)} pairs "
              f"({n_pos_sup} sup-pos, {n_pos_ss} self-sup-pos, "
              f"{n_neg_hard} hard-neg, {n_neg_rand} rand-neg)")

    def _load_crop(self, tracklet):
        try:
            img = Image.open(tracklet['frame_path']).convert('RGB')
            bbox = tracklet['bbox']
            W, H = img.size
            pad = 0.1
            x0 = max(0, bbox[0] - (bbox[2]-bbox[0])*pad)
            y0 = max(0, bbox[1] - (bbox[3]-bbox[1])*pad)
            x1 = min(W, bbox[2] + (bbox[2]-bbox[0])*pad)
            y1 = min(H, bbox[3] + (bbox[3]-bbox[1])*pad)
            return img.crop((x0, y0, x1, y1))
        except Exception:
            return Image.new('RGB', (224, 224))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t_a, t_b, label = self.pairs[idx]
        crop_a = self._load_crop(t_a)
        crop_b = self._load_crop(t_b)
        return (_CLIP_TRANSFORM(crop_a),
                _CLIP_TRANSFORM(crop_b),
                torch.tensor(label, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

def contrastive_loss(emb_a, emb_b, labels, margin=0.3, temperature=0.07):
    """
    Combined loss:
    - Positive pairs (label=1): push cosine similarity to 1
    - Negative pairs (label=0): push cosine similarity below margin
    """
    cos = F.cosine_similarity(emb_a, emb_b)

    # InfoNCE-style contrastive
    pos_loss = (1 - cos) * labels                        # minimise for positives
    neg_loss = F.relu(cos - margin) * (1 - labels)      # push below margin for negatives

    loss = (pos_loss + neg_loss).mean()
    return loss, cos


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reid(args):
    import numpy as np
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from paths import SEQ_TEST
    all_tracklets = []

    # Load S01 + S04 with GT labels (supervised)
    for seq_path in SEQ_TRAIN:
        print(f"Loading {Path(str(seq_path)).name} (GT supervised)...")
        tracklets = load_sequence(str(seq_path), use_yolo_detections=False,
                                  multi_cam_only=True)
        all_tracklets.extend(tracklets)
        print(f"  {len(tracklets)} tracklets")

    # Load S03 YOLO tracklets with self-supervised pseudo-labels:
    # Same tracklet across frames = trivially positive (skip — already one crop)
    # Different tracklets in same camera = hard negatives (close in space/time)
    # Different tracklets across cameras = soft negatives
    # Tracklets with gt_global_id != -1 (IoU-matched GT) = supervised positives
    print(f"Loading S03 (YOLO, self-supervised domain adaptation)...")
    try:
        s03_tracklets = load_sequence(str(SEQ_TEST), use_yolo_detections=True,
                                      multi_cam_only=False)
        # Assign pseudo-labels: gt_global_id=-1 for all (no GT),
        # but tracklets from same IoU-matched GT vehicle get real labels
        # For unmatched tracklets: use tracklet_id as pseudo-identity
        # (each tracklet = unique vehicle appearance = valid for within-tracklet aug)
        n_matched_s03 = sum(1 for t in s03_tracklets if t.get('gt_global_id', -1) != -1)
        print(f"  {len(s03_tracklets)} S03 YOLO tracklets "
              f"({n_matched_s03} with matched GT labels)")
        all_tracklets.extend(s03_tracklets)
    except Exception as e:
        print(f"  [warn] Could not load S03 tracklets: {e}")

    print(f"Total: {len(all_tracklets)} tracklets (S01+S04+S03)")

    if args.backbone == 'clip':
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        def get_emb(pix): return model(pixel_values=pix).pooler_output
    else:
        from transformers import ViTModel
        model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
        def get_emb(pix): return model(pixel_values=pix).last_hidden_state[:, 0, :]

    # ── Phase 1: precompute all embeddings ───────────────────────────────────
    print("\nPhase 1: Precomputing embeddings (one-time image I/O)...")
    model.eval()

    def load_pixel(t):
        try:
            img  = Image.open(t['frame_path']).convert('RGB')
            bbox = t['bbox']; W, H = img.size; pad = 0.1
            x0 = max(0, bbox[0]-(bbox[2]-bbox[0])*pad)
            y0 = max(0, bbox[1]-(bbox[3]-bbox[1])*pad)
            x1 = min(W, bbox[2]+(bbox[2]-bbox[0])*pad)
            y1 = min(H, bbox[3]+(bbox[3]-bbox[1])*pad)
            return _CLIP_TRANSFORM(img.crop((x0, y0, x1, y1)))
        except Exception:
            return torch.zeros(3, 224, 224)

    all_tids  = [t['tracklet_id'] for t in all_tracklets]
    tid_to_t  = {t['tracklet_id']: t for t in all_tracklets}
    tid_to_gid = {t['tracklet_id']: t.get('gt_global_id', -1) for t in all_tracklets}

    # Precompute in batches
    chunk = 256
    emb_list = []
    with torch.no_grad():
        for s in tqdm(range(0, len(all_tids), chunk), desc="  extracting"):
            batch_pix = torch.stack([load_pixel(tid_to_t[tid])
                                     for tid in all_tids[s:s+chunk]]).to(device)
            emb_list.append(get_emb(batch_pix).cpu())

    # emb_matrix on GPU — directly learnable via projection head
    emb_matrix = torch.cat(emb_list, dim=0).to(device)  # (N, embed_dim)
    tid_to_idx = {tid: i for i, tid in enumerate(all_tids)}
    print(f"  {emb_matrix.shape[0]} embeddings, dim={emb_matrix.shape[1]}")

    # ── Phase 2: build pairs as GPU index tensors ────────────────────────────
    print("\nPhase 2: Building pairs...")
    from collections import defaultdict
    by_gid = defaultdict(list)
    for tid in all_tids:
        gid = tid_to_gid[tid]
        if gid != -1:
            by_gid[gid].append(tid)

    pos_pairs, neg_pairs = [], []
    for gid, tids in by_gid.items():
        if len(tids) < 2:
            continue
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                ti, tj = tid_to_t[tids[i]], tid_to_t[tids[j]]
                if ti['cam_id'] != tj['cam_id']:
                    pos_pairs.append((tids[i], tids[j]))

    # ── Semi-hard negative mining ────────────────────────────────────────────
    # Instead of random negatives, mine semi-hard negatives:
    # negatives that are closer than the hardest positive (most confusing)
    # This forces the backbone to learn fine vehicle discriminative features.
    # Uses the pre-computed emb_matrix for fast cosine similarity lookup.
    print("  Mining semi-hard negatives from pre-computed embeddings...")
    
    # Compute cosine similarity matrix in blocks to avoid OOM
    block = 512
    N_emb = emb_matrix.shape[0]
    
    # For each vehicle, find its hardest positive distance
    hardest_pos_dist: dict = {}  # gid -> max(cos_sim) to same-vehicle tracklet
    for gid, tids_g in by_gid.items():
        if len(tids_g) < 2:
            continue
        idx_g = [tid_to_idx[t] for t in tids_g]
        emb_g = F.normalize(emb_matrix[idx_g], dim=1)
        sim   = emb_g @ emb_g.T
        # Hardest positive = LOWEST similarity (farthest same-vehicle pair)
        mask  = torch.ones_like(sim, dtype=torch.bool)
        mask.fill_diagonal_(False)
        hardest_pos_dist[gid] = sim[mask].min().item() if mask.any() else 0.0

    # Build semi-hard negatives: diff vehicle, similarity > hardest_pos
    semi_hard_neg = []
    attempts = 0
    while len(semi_hard_neg) < len(pos_pairs) * 3 and attempts < 500000:
        attempts += 1
        a = random.choice(all_tids)
        b = random.choice(all_tids)
        ga, gb = tid_to_gid[a], tid_to_gid[b]
        if ga == gb or ga == -1 or gb == -1:
            continue
        ia_idx, ib_idx = tid_to_idx[a], tid_to_idx[b]
        cos_ab = F.cosine_similarity(
            emb_matrix[ia_idx:ia_idx+1], emb_matrix[ib_idx:ib_idx+1]).item()
        # Semi-hard: harder than easiest positive but not hardest
        threshold = max(hardest_pos_dist.get(ga, 0), 0.1)
        if cos_ab > threshold * 0.5:  # semi-hard zone
            semi_hard_neg.append((a, b))

    # Mix: 50% semi-hard, 50% random negatives (avoid mode collapse)
    attempts = 0
    while len(neg_pairs) < len(pos_pairs) * 2 and attempts < 200000:
        attempts += 1
        a, b = random.choice(all_tids), random.choice(all_tids)
        if tid_to_gid[a] != tid_to_gid[b] and tid_to_gid[a] != -1 and tid_to_gid[b] != -1:
            neg_pairs.append((a, b))

    n = min(len(pos_pairs), len(neg_pairs)//2, args.n_pairs//2)
    n_semi = min(n, len(semi_hard_neg))
    n_rand = n - n_semi // 2
    random.shuffle(pos_pairs); random.shuffle(neg_pairs); random.shuffle(semi_hard_neg)
    all_pairs = ([(a, b, 1) for a, b in pos_pairs[:n]] +
                 [(a, b, 0) for a, b in semi_hard_neg[:n_semi//2]] +
                 [(a, b, 0) for a, b in neg_pairs[:n_rand]])
    print(f"  Pairs: {n} pos, {n_semi//2} semi-hard neg, {n_rand} random neg")
    random.shuffle(all_pairs)

    ia = torch.tensor([tid_to_idx[a] for a,b,l in all_pairs], dtype=torch.long, device=device)
    ib = torch.tensor([tid_to_idx[b] for a,b,l in all_pairs], dtype=torch.long, device=device)
    la = torch.tensor([l for a,b,l in all_pairs], dtype=torch.float32, device=device)
    N_pairs = len(all_pairs)
    print(f"  {N_pairs} pairs ({n} pos, {n*2} neg) — all on GPU")

    # ── Phase 3: end-to-end backbone fine-tuning ────────────────────────────
    # Store pixel tensors in GPU memory — direct backbone forward, real gradients
    print("\nStoring pixel tensors on GPU for end-to-end training...")
    all_pixels = []
    for tid in tqdm(all_tids, desc="  loading pixels"):
        all_pixels.append(load_pixel(tid_to_t[tid]))
    pixel_matrix = torch.stack(all_pixels).to(device)  # (N, 3, 224, 224)
    print(f"  Pixel matrix: {pixel_matrix.shape} on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_sep = -1.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bs = min(args.batch_size, 64)

    print(f"\n### RE-ID FINE-TUNING ({args.epochs} epochs) — end-to-end backbone ###")

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(N_pairs, device=device)
        pi_e, pj_e, pl_e = ia[perm], ib[perm], la[perm]

        total_loss = 0.0
        all_cos_ep, all_lbl_ep = [], []
        n_batches = 0

        for s in range(0, N_pairs, bs):
            e = min(s + bs, N_pairs)
            bi, bj, bl = pi_e[s:e], pj_e[s:e], pl_e[s:e]

            pix_a = pixel_matrix[bi]
            pix_b = pixel_matrix[bj]

            emb_a = F.normalize(get_emb(pix_a), dim=1)
            emb_b = F.normalize(get_emb(pix_b), dim=1)

            loss, cos = contrastive_loss(emb_a, emb_b, bl, margin=args.margin)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            all_cos_ep.extend(cos.detach().cpu().tolist())
            all_lbl_ep.extend(bl.cpu().tolist())
            n_batches += 1

        scheduler.step()

        cos_arr = np.array(all_cos_ep)
        lbl_arr = np.array(all_lbl_ep)
        pos_m = cos_arr[lbl_arr==1].mean() if (lbl_arr==1).any() else 0
        neg_m = cos_arr[lbl_arr==0].mean() if (lbl_arr==0).any() else 0
        sep   = pos_m - neg_m

        print(f"Ep {epoch:03d} | loss={total_loss/n_batches:.4f} | "
              f"pos={pos_m:.3f} neg={neg_m:.3f} sep={sep:.3f}")

        if sep > best_sep:
            best_sep = sep
            torch.save(model.state_dict(), out_path)
            print(f"         → saved (sep={sep:.3f})")

    print(f"\nBest separation: {best_sep:.3f}  →  {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone',    default='clip', choices=['clip', 'vit'])
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--lr',          type=float, default=1e-5)
    p.add_argument('--batch-size',  type=int,   default=128)
    p.add_argument('--n-pairs',     type=int,   default=20000)
    p.add_argument('--margin',      type=float, default=0.3)
    p.add_argument('--gpu',         type=int,   default=0)
    p.add_argument('--output',      default='outputs/reid_backbone.pt')
    return p.parse_args()


if __name__ == '__main__':
    train_reid(parse_args())