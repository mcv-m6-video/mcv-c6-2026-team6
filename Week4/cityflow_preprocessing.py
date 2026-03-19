"""
cityflow_preprocessing.py
=========================
Parse CityFlow / AIC22 Track-1 annotations into a flat list of tracklet dicts.

When use_yolo_detections=True the full filter chain from filter_tracklets.py
is applied automatically (same criteria the GT evaluator uses):
  conf → area → ROI mask → per-frame NMS → min tracklet length
  optionally: cross-camera filter (drop vehicles seen in only 1 camera)

CityFlow directory layout expected:
  {seq_root}/
    c001/
      img1/          JPEG frames  000001.jpg ...
      gt/gt.txt      MOT format: frame,id,x,y,w,h,conf,class,vis
      roi.jpg        (optional) binary ROI mask
      det/det.txt    (optional) YOLO tracking output
    c002/ ...
  gt_mtmc.txt        optional, at seq_root or one level up
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

from filter_tracklets import (
    load_mot_file, load_roi_mask,
    filter_camera_tracklets, filter_multi_camera,
    _assign_track_ids,
)
from dataset_extras import load_detection_ensemble

# Week-3 utility: cleaner det loader with integrated NMS+conf+area+ROI filter
# Used at test time (S03) to load YOLO detections with consistent filtering.
try:
    from utils import load_detections_aicity as _load_dets_aicity
    _HAVE_AICITY_LOADER = True
except ImportError:
    _HAVE_AICITY_LOADER = False


def load_sequence(
    seq_path: str,
    fps: float = 10.0,
    use_yolo_detections: bool = False,
    yolo_det_dir: Optional[str] = None,
    det_subpath: str = 'det/det.txt',
    apply_filter:   bool  = True,
    min_conf:       float = 0.3,
    min_area:       float = 400.0,
    min_track_len:  int   = 3,
    nms_iou:        float = 0.5,
    multi_cam_only: bool  = False,
) -> List[Dict]:
    """
    Load one CityFlow sequence and return a flat list of tracklet dicts.

    Filtering (YOLO mode only):
      - active==0 rows skipped in GT mode via _read_gt_file
      - conf / area / ROI / NMS / track-length applied in YOLO mode
      - multi_cam_only: drop vehicles seen in fewer than 2 cameras

    Returns list of dicts with keys:
      tracklet_id, cam_id, cam_name, bbox, frame_path,
      timestamp, velocity, gt_global_id, all_frames
    """
    seq_path   = Path(seq_path)
    cameras    = sorted([d for d in seq_path.iterdir()
                         if d.is_dir() and d.name.startswith('c')])
    cam_id_map = {cam.name: idx for idx, cam in enumerate(cameras)}
    global_ids = _load_mtmc_gt(seq_path)

    # ── Step 1: load raw detections ──────────────────────────────────────────
    all_cam_raw: Dict[str, Dict[int, List]] = {}

    for cam_dir in cameras:
        cam_name = cam_dir.name

        if use_yolo_detections:
            det_file = (Path(yolo_det_dir) / f'{cam_name}.txt'
                        if yolo_det_dir else cam_dir / det_subpath)
            if not det_file.exists():
                # Fall back to detection ensemble from det/ directory
                ensemble = load_detection_ensemble(str(cam_dir),
                                                   min_conf=min_conf,
                                                   nms_iou=nms_iou)
                if ensemble:
                    print(f"  [ensemble fallback] {cam_name}: "
                          f"no det.txt, using 3-detector ensemble")
                    # Convert ensemble format to MOT tracklets
                    # Assign synthetic track IDs per detection (no tracking)
                    # These will be short single-frame tracklets — the graph
                    # can still link them via appearance
                    from collections import defaultdict as _dd
                    mot = _dd(list)
                    for frame_no, dets in sorted(ensemble.items()):
                        for i, det in enumerate(dets):
                            tid = frame_no * 1000 + i  # unique synthetic ID
                            mot[tid].append([frame_no,
                                             det[0], det[1], det[2], det[3],
                                             det[4]])
                    mot = dict(mot)
                else:
                    print(f"  [skip] {cam_name}: no det file or ensemble at {cam_dir}")
                    continue
            else:
                mot = load_mot_file(str(det_file))

            if apply_filter:
                roi_path = cam_dir / 'roi.jpg'
                mot = filter_camera_tracklets(
                    mot,
                    roi_mask      = load_roi_mask(str(roi_path) if roi_path.exists() else None),
                    min_conf      = min_conf,
                    min_area      = min_area,
                    min_track_len = min_track_len,
                    nms_iou       = nms_iou,
                )

            all_cam_raw[cam_name] = {
                tid: [(int(d[0]), [d[1], d[2], d[3], d[4]]) for d in dets]
                for tid, dets in mot.items()
            }

        else:
            gt_file = cam_dir / 'gt' / 'gt.txt'
            if gt_file.exists():
                tracks = _read_gt_file(gt_file)
                if tracks:
                    all_cam_raw[cam_name] = tracks
                else:
                    print(f"  [warn] GT file empty after parsing: {gt_file}")
            else:
                print(f"  [warn] GT file not found: {gt_file}")

    # ── Step 2: cross-camera filter ──────────────────────────────────────────
    if apply_filter and multi_cam_only:
        if use_yolo_detections:
            # YOLO mode: use filter_multi_camera with gt_mtmc.txt
            mot_fmt = {
                cam: {tid: [[f, b[0], b[1], b[2], b[3], 1.0] for f, b in dets]
                      for tid, dets in raw.items()}
                for cam, raw in all_cam_raw.items()
            }
            gt_mtmc = seq_path / 'gt_mtmc.txt'
            mot_fmt = filter_multi_camera(
                mot_fmt,
                gt_mtmc_path=str(gt_mtmc) if gt_mtmc.exists() else None,
                min_cameras=2,
            )
            all_cam_raw = {
                cam: {tid: [(int(d[0]), [d[1], d[2], d[3], d[4]]) for d in dets]
                      for tid, dets in tracks.items()}
                for cam, tracks in mot_fmt.items()
            }
        else:
            # GT mode: filter directly using global_ids from gt_mtmc.txt.
            # Keep only vehicle IDs that appear in >= 2 cameras.
            # This avoids the YOLO track ID matching problem entirely.
            from collections import defaultdict as _dd
            gid_cameras: Dict = _dd(set)
            for cam_name, tracks in all_cam_raw.items():
                for vid in tracks:
                    gid = global_ids.get((cam_name, vid))
                    if gid is not None:
                        gid_cameras[gid].add(cam_name)

            valid_gids = {gid for gid, cams in gid_cameras.items()
                          if len(cams) >= 2}

            if valid_gids:
                filtered_raw: Dict = {}
                for cam_name, tracks in all_cam_raw.items():
                    kept = {vid: dets for vid, dets in tracks.items()
                            if global_ids.get((cam_name, vid)) in valid_gids}
                    if kept:
                        filtered_raw[cam_name] = kept
                n_before = sum(len(v) for v in all_cam_raw.values())
                n_after  = sum(len(v) for v in filtered_raw.values())
                print(f"  [multi_cam_filter] GT mode: kept {n_after}/{n_before} "
                      f"tracklets ({len(valid_gids)} multi-cam vehicles)")
                all_cam_raw = filtered_raw
            else:
                print(f"  [warn] multi_cam_filter: no multi-cam vehicles found "
                      f"— check gt_mtmc.txt exists at {seq_path}")

    # ── Step 3: match YOLO tracklets to GT to get gt_global_ids ─────────────
    # Only needed for YOLO mode — GT mode already has correct IDs
    if use_yolo_detections:
        # Flatten all_cam_raw into a preliminary tracklet list for matching
        _prelim = []
        for cam_dir in [d for d in seq_path.iterdir()
                        if d.is_dir() and d.name.startswith('c')]:
            for vid, dets in all_cam_raw.get(cam_dir.name, {}).items():
                _prelim.append({
                    'cam_name': cam_dir.name,
                    'all_frames': dets,
                    'gt_global_id': -1,
                })
        _prelim = _match_yolo_to_gt(_prelim, seq_path)
        # Build a lookup: (cam_name, vid) -> gt_global_id
        _prelim_iter = iter(_prelim)
        _gt_id_lookup: Dict[tuple, int] = {}
        for cam_dir in [d for d in seq_path.iterdir()
                        if d.is_dir() and d.name.startswith('c')]:
            for vid in all_cam_raw.get(cam_dir.name, {}):
                entry = next(_prelim_iter, None)
                if entry:
                    _gt_id_lookup[(cam_dir.name, vid)] = entry['gt_global_id']
    else:
        _gt_id_lookup = {}

    # ── Step 3: build tracklet dicts ─────────────────────────────────────────
    tracklets: List[Dict] = []

    for cam_dir in cameras:
        cam_name = cam_dir.name
        # Find frame directory — try img1/, vdo/, frames/ in order
        img_dir = next(
            (cam_dir / d for d in ['img1', 'vdo', 'frames', 'images']
             if (cam_dir / d).exists()),
            cam_dir / 'img1'   # fallback (will warn when accessed)
        )

        for vid, dets in all_cam_raw.get(cam_name, {}).items():
            dets   = sorted(dets, key=lambda d: d[0])
            mid    = len(dets) // 2
            frame_no, bbox = dets[mid]

            img_path = img_dir / f'{frame_no:06d}.jpg'
            if not img_path.exists():
                cands    = list(img_dir.glob(f'{frame_no:06d}.*'))
                img_path = cands[0] if cands else img_path

            dx = dets[-1][1][0] - dets[0][1][0] if len(dets) > 1 else 0.0
            dy = dets[-1][1][1] - dets[0][1][1] if len(dets) > 1 else 0.0

            tracklets.append({
                'tracklet_id':  f'{cam_name}_{vid}',
                'cam_id':       cam_id_map[cam_name],
                'cam_name':     cam_name,
                'cam_dir':      str(cam_dir),
                'bbox':         bbox,
                'frame_path':   str(img_path),
                'timestamp':    frame_no / fps,
                'velocity':     [float(dx), float(dy)],
                'gt_global_id': (
                    _gt_id_lookup.get((cam_name, vid),
                                      global_ids.get((cam_name, vid), -1))
                    if use_yolo_detections else
                    global_ids.get((cam_name, vid), -1)
                ),
                'all_frames':   dets,
            })

    return tracklets


def split_tracklets_by_camera(tracklets: List[Dict]) -> Dict[str, List[Dict]]:
    by_cam: Dict[str, List] = defaultdict(list)
    for t in tracklets:
        by_cam[t['cam_name']].append(t)
    return dict(by_cam)


def get_num_cameras(tracklets: List[Dict]) -> int:
    return len({t['cam_id'] for t in tracklets})


# ── Private helpers ──────────────────────────────────────────────────────────



def _match_yolo_to_gt(
    tracklets: list,
    seq_path: 'Path',
    iou_threshold: float = 0.3,
) -> list:
    """
    For YOLO-tracked detections, assign gt_global_id by matching each
    YOLO tracklet to the best-overlapping GT tracklet in the same camera.

    iou_threshold lowered to 0.3 (from 0.4) since YOLO boxes drift slightly.
    min_match scales with tracklet length: max(1, len // 3), so short tracklets
    (3-5 frames) need only 1 matching frame instead of always requiring 2.
    """
    gt_by_cam: Dict[str, Dict[int, List]] = {}
    global_ids = _load_mtmc_gt(seq_path)

    for cam_dir in sorted(seq_path.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith('c'):
            continue
        gt_file = cam_dir / 'gt' / 'gt.txt'
        if not gt_file.exists():
            continue
        gt_by_cam[cam_dir.name] = _read_gt_file(gt_file)

    if not gt_by_cam and not global_ids:
        return tracklets

    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    updated = []
    for t in tracklets:
        cam_name  = t['cam_name']
        cam_gt    = gt_by_cam.get(cam_name, {})
        if not cam_gt:
            updated.append(t)
            continue

        yolo_frames = {f: bbox for f, bbox in t['all_frames']}
        track_len   = len(yolo_frames)
        min_match   = max(1, track_len // 3)

        best_vid   = -1
        best_count = 0
        for gt_vid, gt_dets in cam_gt.items():
            matches = sum(
                1 for gt_frame, gt_bbox in gt_dets
                if gt_frame in yolo_frames
                and _iou(yolo_frames[gt_frame], gt_bbox) >= iou_threshold
            )
            if matches > best_count:
                best_count = matches
                best_vid   = gt_vid

        if best_vid != -1 and best_count >= min_match:
            global_id = global_ids.get((cam_name, best_vid), -1)
            if global_id != -1:
                t = dict(t)
                t['gt_global_id'] = global_id

        updated.append(t)

    matched = sum(1 for t in updated if t['gt_global_id'] != -1)
    total   = len(updated)
    print(f"  [gt_match] {total} tracklets: {matched} matched ({matched/max(total,1)*100:.1f}%)")
    return updated

def _read_gt_file(gt_path: Path, min_visibility: float = 0.0) -> Dict[int, List]:
    """
    Parse gt.txt into {local_track_id: [(frame, [x1,y1,x2,y2]), ...]}.

    CityFlow gt.txt format (comma-separated, 8 or 9 columns):
      frame, id, x, y, w, h, conf_or_active, class, [visibility]

    Column 6 in CityFlow is ambiguous across dataset versions:
      - Some versions: 1=active, 0=ignore  (MOTChallenge convention)
      - Some versions: always 1 (confidence placeholder, no ignore rows)
      - Some versions: -1 for crowd/ignore regions
    We skip only rows where col6 is explicitly 0 (MOT ignore flag).
    We do NOT filter on visibility here to avoid dropping valid GT tracklets.
    """
    result: Dict[int, List] = defaultdict(list)
    n_total = n_skipped = 0
    with open(gt_path, newline='') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#') or len(row) < 6:
                continue
            n_total += 1
            # Only skip rows explicitly marked as ignore (col6 == 0)
            # Do NOT skip col6 == -1 (those are valid CityFlow annotations)
            if len(row) >= 7:
                try:
                    flag = int(float(row[6]))
                    if flag == 0:
                        n_skipped += 1
                        continue
                except (ValueError, IndexError):
                    pass
            frame = int(row[0])
            vid   = int(row[1])
            x, y  = float(row[2]), float(row[3])
            w, h  = float(row[4]), float(row[5])
            result[vid].append((frame, [x, y, x + w, y + h]))
    if n_total == 0:
        print(f"  [warn] _read_gt_file: {gt_path} is empty or unreadable")
    elif not result:
        print(f"  [warn] _read_gt_file: {gt_path} had {n_total} rows but all skipped "
              f"(col6==0 for {n_skipped}). Check gt.txt format.")
    return dict(result)


def _load_mtmc_gt(seq_path: Path) -> Dict[tuple, int]:
    mapping: Dict[tuple, int] = {}
    for candidate in [seq_path / 'gt_mtmc.txt', seq_path.parent / 'gt_mtmc.txt']:
        if not candidate.exists():
            continue
        with open(candidate) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                cam_raw  = parts[0]
                vid      = int(parts[1])
                cam_name = cam_raw if cam_raw.startswith('c') else f'c{int(cam_raw):03d}'
                mapping[(cam_name, vid)] = vid
        break
    return mapping


# =============================================================================
# MTSC-based sequence loader (uses dataset pre-computed tracklets)
# =============================================================================

def load_sequence_mtsc(
    seq_path: str,
    fps: float = 10.0,
    apply_filter: bool = True,
    min_area:     float = 400.0,
    min_track_len: int  = 3,
    nms_iou:      float = 0.5,
    multi_cam_only: bool = False,
) -> List[Dict]:
    """
    Like load_sequence() but uses the best available MTSC pre-computed
    tracklet file instead of YOLO output.

    MTSC files are provided by the CityFlow dataset and already have
    single-camera consistent track IDs — better than running tracking
    from scratch for cameras your YOLO was not fine-tuned on.

    Falls back to gt.txt if no MTSC file is found for a camera.
    """
    from dataset_extras import load_best_mtsc

    seq_path   = Path(seq_path)
    cameras    = sorted([d for d in seq_path.iterdir()
                         if d.is_dir() and d.name.startswith('c')])
    cam_id_map = {cam.name: idx for idx, cam in enumerate(cameras)}
    global_ids = _load_mtmc_gt(seq_path)

    all_cam_raw: Dict[str, Dict[int, List]] = {}

    for cam_dir in cameras:
        cam_name = cam_dir.name
        mtsc_tracklets = load_best_mtsc(str(cam_dir))

        if mtsc_tracklets is not None:
            # Convert from {tid: [[frame,x1,y1,x2,y2,conf],...]} 
            # to internal   {tid: [(frame,[x1,y1,x2,y2]),...]}
            from filter_tracklets import load_roi_mask, filter_camera_tracklets
            roi_path = cam_dir / 'roi.jpg'
            roi_mask = load_roi_mask(str(roi_path) if roi_path.exists() else None)

            if apply_filter:
                mtsc_tracklets = filter_camera_tracklets(
                    mtsc_tracklets,
                    roi_mask      = roi_mask,
                    min_conf      = 0.0,   # MTSC confs are all 1.0
                    min_area      = min_area,
                    min_track_len = min_track_len,
                    nms_iou       = nms_iou,
                )

            all_cam_raw[cam_name] = {
                tid: [(int(d[0]), [d[1], d[2], d[3], d[4]]) for d in dets]
                for tid, dets in mtsc_tracklets.items()
            }
        else:
            # Fall back to GT
            gt_file = cam_dir / 'gt' / 'gt.txt'
            if gt_file.exists():
                all_cam_raw[cam_name] = _read_gt_file(gt_file)

    # Reuse the same dict → tracklet conversion as load_sequence
    tracklets: List[Dict] = []
    for cam_dir in cameras:
        cam_name = cam_dir.name
        # Find frame directory — try img1/, vdo/, frames/ in order
        img_dir = next(
            (cam_dir / d for d in ['img1', 'vdo', 'frames', 'images']
             if (cam_dir / d).exists()),
            cam_dir / 'img1'   # fallback (will warn when accessed)
        )
        for vid, dets in all_cam_raw.get(cam_name, {}).items():
            dets   = sorted(dets, key=lambda d: d[0])
            mid    = len(dets) // 2
            frame_no, bbox = dets[mid]
            img_path = img_dir / f'{frame_no:06d}.jpg'
            if not img_path.exists():
                cands = list(img_dir.glob(f'{frame_no:06d}.*'))
                img_path = cands[0] if cands else img_path
            dx = dets[-1][1][0] - dets[0][1][0] if len(dets) > 1 else 0.0
            dy = dets[-1][1][1] - dets[0][1][1] if len(dets) > 1 else 0.0
            tracklets.append({
                'tracklet_id':  f'{cam_name}_{vid}',
                'cam_id':       cam_id_map[cam_name],
                'cam_name':     cam_name,
                'cam_dir':      str(cam_dir),
                'bbox':         bbox,
                'frame_path':   str(img_path),
                'timestamp':    frame_no / fps,
                'velocity':     [float(dx), float(dy)],
                'gt_global_id': (
                    _gt_id_lookup.get((cam_name, vid),
                                      global_ids.get((cam_name, vid), -1))
                    if use_yolo_detections else
                    global_ids.get((cam_name, vid), -1)
                ),
                'all_frames':   dets,
            })
    return tracklets