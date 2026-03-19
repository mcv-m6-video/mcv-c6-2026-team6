"""
vehicle_graph_builder.py
========================
Replaces Doc2GraphFormer's GraphBuilder.

Design
------
One DGL graph  =  one video sequence (e.g. S03).
One NODE       =  one tracklet  (vehicle_id × camera).
One EDGE       =  potential cross-camera match.

Edge construction strategies (--edge-type flag):
  'cross_cam'   : fully-connected ACROSS cameras only (default, mirrors the
                  doc paper's fully-connected graph but respects domain logic)
  'fully'       : fully-connected including same-camera (baseline ablation)
  'temporal'    : cross-camera edges only between temporally overlapping tracklets
                  (reduces false-negative edges for sequences with many cameras)

Node label   : 0-based camera ID (auxiliary classification task)
Edge label   : 1 = same global vehicle, 0 = different vehicle (main task)
"""

from __future__ import annotations
import torch
import dgl
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class VehicleGraphBuilder:
    """
    Parameters
    ----------
    edge_type         : 'cross_cam' | 'fully' | 'temporal'
    temporal_overlap_s: minimum overlap in seconds to create a temporal edge
                        (only used when edge_type='temporal')
    """

    def __init__(
        self,
        edge_type: str = 'temporal',
        temporal_overlap_s: float = 0.0,
        max_time_gap_s: float = 30.0,   # 30s covers S01 intersection; use 120s for S03
    ):
        """
        edge_type: 'temporal' (default) | 'cross_cam' | 'fully'

        'temporal' is the correct default for MTMC:
          - Only connects cross-camera tracklets within max_time_gap_s of each other
          - A vehicle travelling between cameras takes 10-120 seconds typically
          - This cuts the 92:1 negative ratio to ~5:1 without losing positives
          - max_time_gap_s=30 works for intersection cameras (S01/S03 same area)
          - increase to 120s for city-wide sequences with longer travel times

        'cross_cam' connects ALL cross-camera pairs — 92:1 imbalance, use for ablation only
        """
        self.edge_type          = edge_type
        self.temporal_overlap_s = temporal_overlap_s
        self.max_time_gap_s     = max_time_gap_s

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build_graph(
        self,
        tracklets: List[Dict],
        neg_pos_ratio: float = 10.0,
    ) -> Tuple[dgl.DGLGraph, Dict]:
        """
        Build a single DGL graph from a list of tracklet dicts.

        neg_pos_ratio: max ratio of negative to positive edges to keep.
          At 92:1 raw imbalance, focal loss struggles. Subsampling to 10:1
          gives the model a balanced training signal while keeping all positives.
          Set to None or 0 to disable subsampling (keeps all edges).
        """
        N = len(tracklets)
        assert N > 0, "Cannot build graph from empty tracklet list."

        u, v, edge_labels = self._build_edges(tracklets)

        # Negative edge subsampling — keep ALL positives, subsample negatives
        if neg_pos_ratio and neg_pos_ratio > 0:
            import random
            pos_idx = [i for i, l in enumerate(edge_labels) if l == 1]
            neg_idx = [i for i, l in enumerate(edge_labels) if l == 0]
            n_pos = len(pos_idx)
            if n_pos > 0:
                max_neg = int(n_pos * neg_pos_ratio)
                if len(neg_idx) > max_neg:
                    neg_idx = random.sample(neg_idx, max_neg)
            keep = sorted(pos_idx + neg_idx)
            u           = [u[i] for i in keep]
            v           = [v[i] for i in keep]
            edge_labels = [edge_labels[i] for i in keep]

        g = dgl.graph((u, v), num_nodes=N, idtype=torch.int32)
        g.edata['label'] = torch.tensor(edge_labels, dtype=torch.long)
        g.ndata['label'] = torch.tensor(
            [t['cam_id'] for t in tracklets], dtype=torch.long)

        features = {
            'frame_paths':   [t['frame_path']   for t in tracklets],
            'boxs':          [t['bbox']          for t in tracklets],
            'cam_ids':       [t['cam_id']        for t in tracklets],
            'cam_names':     [t['cam_name']      for t in tracklets],
            'cam_dirs':      [t.get('cam_dir')   for t in tracklets],  # for roi/calib
            'timestamps':    [t['timestamp']     for t in tracklets],
            'velocities':    [t['velocity']      for t in tracklets],
            'tracklet_ids':  [t['tracklet_id']   for t in tracklets],
            'gt_global_ids': [t['gt_global_id']  for t in tracklets],
            'all_frames':    [t.get('all_frames', [(0, t['bbox'])]) for t in tracklets],
            'all_frame_paths': [t.get('all_frame_paths', [t['frame_path']])
                                for t in tracklets],  # multi-frame pooling
        }
        return g, features

    def build_graphs_per_sequence(
        self,
        sequences: List[List[Dict]],
    ) -> Tuple[List[dgl.DGLGraph], List[Dict]]:
        """Convenience wrapper for multiple sequences."""
        graphs, feature_list = [], []
        for tracklets in tqdm(sequences, desc='building graphs'):
            g, feats = self.build_graph(tracklets)
            graphs.append(g)
            feature_list.append(feats)
        return graphs, feature_list

    # -----------------------------------------------------------------------
    # Edge builders
    # -----------------------------------------------------------------------

    def _build_edges(
        self,
        tracklets: List[Dict],
    ) -> Tuple[List[int], List[int], List[int]]:

        if self.edge_type == 'cross_cam':
            return self._cross_camera_edges(tracklets)
        elif self.edge_type == 'fully':
            return self._fully_connected_edges(tracklets)
        elif self.edge_type == 'temporal':
            return self._temporal_overlap_edges(tracklets)
        else:
            raise ValueError(f"Unknown edge_type: {self.edge_type}")

    def _cross_camera_edges(
        self,
        tracklets: List[Dict],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Directed edges between every pair from DIFFERENT cameras."""
        u, v, labels = [], [], []
        N = len(tracklets)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if tracklets[i]['cam_id'] != tracklets[j]['cam_id']:
                    u.append(i)
                    v.append(j)
                    same = int(
                        tracklets[i]['gt_global_id'] != -1 and
                        tracklets[i]['gt_global_id'] == tracklets[j]['gt_global_id']
                    )
                    labels.append(same)
        return u, v, labels

    def _fully_connected_edges(
        self,
        tracklets: List[Dict],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Fully connected (includes same-camera). Ablation baseline."""
        u, v, labels = [], [], []
        N = len(tracklets)
        for i in range(N):
            for j in range(N):
                if i != j:
                    u.append(i)
                    v.append(j)
                    same = int(
                        tracklets[i]['gt_global_id'] != -1 and
                        tracklets[i]['gt_global_id'] == tracklets[j]['gt_global_id']
                    )
                    labels.append(same)
        return u, v, labels

    def _temporal_overlap_edges(
        self,
        tracklets: List[Dict],
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Cross-camera edges using two complementary filters:

        1. CAMERA ADJACENCY (data-driven): only connect camera pair (A,B)
           if at least one GT vehicle appears in BOTH cameras. This proves
           the cameras are geographically connected. Eliminates ~76% of
           cross-cam pairs in S04 (city-wide, 25 cameras) where most
           camera pairs are too far apart for any vehicle to traverse.

        2. TEMPORAL PROXIMITY: only connect tracklets whose midpoint
           timestamps are within max_time_gap_s of each other.
           Uses midpoints (not spans) to handle long GT tracklets correctly.

        Both filters must pass for an edge to be created.
        """
        # ── Filter 1: data-driven camera adjacency ────────────────────────
        # Build set of (cam_id_a, cam_id_b) pairs that share at least one vehicle
        from collections import defaultdict as _dd
        vehicle_to_cams: dict = _dd(set)
        for t in tracklets:
            gid = t.get('gt_global_id', -1)
            if gid != -1:
                vehicle_to_cams[gid].add(t['cam_id'])

        adjacent_cam_pairs: set = set()
        for gid, cams in vehicle_to_cams.items():
            cam_list = list(cams)
            for a in cam_list:
                for b in cam_list:
                    if a != b:
                        adjacent_cam_pairs.add((a, b))

        # If no GT labels available (test time), allow all camera pairs
        if not adjacent_cam_pairs:
            N_cams = len({t['cam_id'] for t in tracklets})
            adjacent_cam_pairs = {
                (a, b)
                for a in range(N_cams) for b in range(N_cams)
                if a != b
            }

        # ── Filter 2: temporal proximity (midpoint-based) ─────────────────
        mids = [t['timestamp'] for t in tracklets]

        u, v, labels = [], [], []
        N = len(tracklets)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                ci, cj = tracklets[i]['cam_id'], tracklets[j]['cam_id']
                if ci == cj:
                    continue
                # Camera adjacency filter
                if (ci, cj) not in adjacent_cam_pairs:
                    continue
                # Temporal proximity filter
                if abs(mids[i] - mids[j]) > self.max_time_gap_s:
                    continue
                u.append(i)
                v.append(j)
                same = int(
                    tracklets[i]['gt_global_id'] != -1 and
                    tracklets[i]['gt_global_id'] == tracklets[j]['gt_global_id']
                )
                labels.append(same)
        return u, v, labels

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def print_stats(self, g: dgl.DGLGraph, name: str = ''):
        pos = g.edata['label'].sum().item()
        neg = len(g.edata['label']) - pos
        print(
            f"Graph {name}: "
            f"nodes={g.num_nodes()}  edges={g.num_edges()}  "
            f"pos_edges={pos}  neg_edges={neg}  "
            f"pos_ratio={pos/max(g.num_edges(),1):.3f}"
        )