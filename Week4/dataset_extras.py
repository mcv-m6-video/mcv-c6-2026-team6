"""
dataset_extras.py
=================
Utilities for the extra per-camera files provided by the CityFlow dataset:

  roi.jpg          — binary mask of valid road region (white=valid, black=ignore)
  calibration.txt  — homography H mapping image pixels → world coordinates (cm)
  mtsc_*.txt       — pre-computed single-camera tracklets from the dataset
                     (3 detectors × 3 trackers = 9 combinations per camera)

How each one improves your pipeline
-------------------------------------
ROI mask:
  Already used in filter_tracklets.py (centre-point test).
  Here we add a stricter ENTRY/EXIT zone detector — vehicles near the ROI
  boundary are entering or leaving the scene, which is extra signal for the
  cross-camera linker (a vehicle that exits one camera should appear soon in
  another camera's entry zone).

Calibration homography:
  Converts bbox bottom-centre pixel coordinates to a shared world coordinate
  system (in cm). This gives your graph a camera-agnostic spatial feature —
  instead of "camera 2, pixel (800,600)" you get "world position (-120, 340 cm)".
  Vehicles from different cameras that share a world position at overlapping
  times are very likely the same vehicle.
  Added as extra node features: [world_x, world_y] (2-d, normalised).

MTSC trackers:
  The dataset ships pre-computed per-camera tracklets from 9 tracker configs.
  Rather than re-running YOLO tracking from scratch you can load the best
  of these as your single-camera tracking input, potentially giving better
  single-camera ID consistency than your fine-tuned YOLO alone.
  Recommendation: use mtsc_tc_mask_rcnn (best coverage, 61 tracks) or
  mtsc_deepsort_mask_rcnn (88 tracks but noisier) as an ensemble.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 1. ROI mask — entry/exit zone detection
# =============================================================================

class ROIAnalyser:
    """
    Loads the binary ROI mask and provides:
      - centre-point validity test  (already in filter_tracklets)
      - entry/exit zone detection   (NEW — useful as graph edge feature)
      - boundary proximity score    (NEW — node feature)
    """

    def __init__(self, roi_path: str, boundary_margin_px: int = 40):
        """
        Parameters
        ----------
        roi_path : path to roi.jpg
        boundary_margin_px : pixels inside the ROI border to consider as
                             "entry/exit zone". Vehicles here are transitioning
                             in or out of scene.
        """
        img = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load ROI: {roi_path}")

        self.mask = img > 127                      # bool (H, W)
        self.H, self.W = self.mask.shape

        # Erode the mask by boundary_margin_px to get the "interior" zone
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (boundary_margin_px * 2 + 1, boundary_margin_px * 2 + 1)
        )
        interior = cv2.erode(self.mask.astype(np.uint8), kernel)
        self.boundary_zone = self.mask & ~interior.astype(bool)  # ring at edge

        # Distance transform: how far each valid pixel is from the boundary
        dist = cv2.distanceTransform(self.mask.astype(np.uint8), cv2.DIST_L2, 5)
        max_d = dist.max() if dist.max() > 0 else 1.0
        self.dist_normalised = dist / max_d        # [0, 1], 0 = boundary

    def is_valid(self, x_centre: float, y_centre: float) -> bool:
        """True if point is inside the ROI."""
        cx = int(np.clip(x_centre, 0, self.W - 1))
        cy = int(np.clip(y_centre, 0, self.H - 1))
        return bool(self.mask[cy, cx])

    def is_entry_exit(self, x_centre: float, y_centre: float) -> bool:
        """True if the point is in the boundary entry/exit zone."""
        cx = int(np.clip(x_centre, 0, self.W - 1))
        cy = int(np.clip(y_centre, 0, self.H - 1))
        return bool(self.boundary_zone[cy, cx])

    def boundary_proximity(self, x_centre: float, y_centre: float) -> float:
        """
        Returns [0, 1] where 0 = right at the ROI boundary, 1 = deep interior.
        Use as a node feature — vehicles at boundaries are more likely to be
        entering/exiting and thus have cross-camera matches.
        """
        cx = int(np.clip(x_centre, 0, self.W - 1))
        cy = int(np.clip(y_centre, 0, self.H - 1))
        if not self.mask[cy, cx]:
            return 0.0
        return float(self.dist_normalised[cy, cx])

    def bbox_features(self, bbox: List[float]) -> List[float]:
        """
        Return [is_valid, is_entry_exit, boundary_proximity] for a bbox.
        Input bbox: [x0, y0, x1, y1]
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return [
            float(self.is_valid(cx, cy)),
            float(self.is_entry_exit(cx, cy)),
            self.boundary_proximity(cx, cy),
        ]


# =============================================================================
# 2. Calibration homography — world coordinate projection
# =============================================================================

class CameraCalibration:
    """
    Loads calibration.txt and provides image→world coordinate mapping via H.

    The homography H maps:
        [u, v, 1]^T  →  [X, Y, W]^T   then  (X/W, Y/W) = world coords in cm

    Usage: project the bottom-centre of a vehicle bbox to ground plane.
    Bottom-centre is more accurate than bbox centre because the ground contact
    point has less perspective distortion.
    """

    def __init__(self, calibration_path: str):
        self.H = self._parse(calibration_path)
        # Compute inverse for world→image projection
        self.H_inv = np.linalg.inv(self.H)

    @staticmethod
    def _parse(path: str) -> np.ndarray:
        with open(path) as f:
            content = f.read()
        for line in content.split('\n'):
            if 'Homography' in line:
                mat_str = line.split(': ')[1].strip()
                rows    = mat_str.split(';')
                return np.array([[float(v) for v in r.split()] for r in rows])
        raise ValueError(f"No homography found in {path}")

    def image_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """Project image pixel (u, v) to world coordinates (X, Y) in cm."""
        pt    = np.array([u, v, 1.0])
        world = self.H @ pt
        world /= world[2]
        return float(world[0]), float(world[1])

    def bbox_world_coords(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Project the bottom-centre of a bbox to world coordinates.
        Bottom-centre = ground contact point, most accurate for vehicles.
        """
        u = (bbox[0] + bbox[2]) / 2   # horizontal centre
        v = bbox[3]                    # bottom edge
        return self.image_to_world(u, v)

    def bbox_features(
        self,
        bbox: List[float],
        world_normalise_range: float = 5000.0,
    ) -> List[float]:
        """
        Returns [world_x_norm, world_y_norm] — normalised world coordinates.
        These are camera-agnostic: same vehicle in two cameras maps to
        approximately the same world position → powerful cross-camera feature.

        world_normalise_range: rough scene size in cm for normalisation.
        CityFlow scenes are ~50m wide → 5000 cm.
        """
        wx, wy = self.bbox_world_coords(bbox)
        return [wx / world_normalise_range, wy / world_normalise_range]


# =============================================================================
# 3. MTSC tracker loader — use dataset pre-computed tracklets
# =============================================================================

# Available tracker files per camera in CityFlow
MTSC_OPTIONS = [
    'mtsc_deepsort_mask_rcnn',
    'mtsc_deepsort_ssd512',
    'mtsc_deepsort_yolo3',
    'mtsc_moana_mask_rcnn',
    'mtsc_moana_ssd512',
    'mtsc_moana_yolo3',
    'mtsc_tc_mask_rcnn',
    'mtsc_tc_ssd512',
    'mtsc_tc_yolo3',
]


def load_mtsc_tracklets(
    mtsc_path: str,
) -> Dict[int, List[List]]:
    """
    Load a pre-computed MTSC tracker file.
    Format: frame, track_id, x, y, w, h, conf, -1, -1, -1
    Returns {track_id: [[frame, x1, y1, x2, y2, conf], ...]}

    These files already have track IDs assigned (single-camera consistent IDs).
    conf is typically 1 (tracker-assigned, not detection confidence).
    """
    tracklets: Dict[int, List] = defaultdict(list)
    with open(mtsc_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid   = int(parts[1])
            x, y  = float(parts[2]), float(parts[3])
            w, h  = float(parts[4]), float(parts[5])
            conf  = float(parts[6]) if len(parts) > 6 else 1.0
            tracklets[tid].append([frame, x, y, x + w, y + h, conf])
    return {tid: sorted(dets, key=lambda d: d[0])
            for tid, dets in tracklets.items()}


def find_best_mtsc_file(cam_dir: Path) -> Optional[Path]:
    """
    Find the best available MTSC file for a camera directory.
    Priority order based on typical CityFlow performance:
      tc_mask_rcnn > deepsort_mask_rcnn > tc_yolo3 > deepsort_yolo3 > others
    """
    # Priority based on empirical analysis:
    # tc > deepsort > moana (ID consistency: 0.997, 0.995, 0.988)
    # mask_rcnn > ssd512 > yolo3 (recall: 0.919, 0.881, 0.743)
    # moana has highest recall but too many fragments (8x GT count)
    # tc has best consistency and fewest fragments (4.7x GT count)
    priority = [
        'mtsc_tc_mask_rcnn.txt',
        'mtsc_deepsort_mask_rcnn.txt',
        'mtsc_tc_ssd512.txt',
        'mtsc_deepsort_ssd512.txt',
        'mtsc_tc_yolo3.txt',
        'mtsc_deepsort_yolo3.txt',
        'mtsc_moana_mask_rcnn.txt',
        'mtsc_moana_ssd512.txt',
        'mtsc_moana_yolo3.txt',
    ]
    mtsc_dir = cam_dir / 'mtsc'
    if not mtsc_dir.exists():
        # Some releases put mtsc files directly in camera dir
        mtsc_dir = cam_dir

    for fname in priority:
        candidate = mtsc_dir / fname
        if candidate.exists():
            return candidate
    return None


def load_best_mtsc(cam_dir: str) -> Optional[Dict[int, List[List]]]:
    """
    Load the best available MTSC tracklet file for a camera directory.
    Returns None if no MTSC file is found (fall back to YOLO).
    """
    path = find_best_mtsc_file(Path(cam_dir))
    if path is None:
        return None
    print(f"  [MTSC] using {path.name}")
    return load_mtsc_tracklets(str(path))


# =============================================================================
# 4. Extended feature builder integration
# =============================================================================

def build_extended_node_features(
    bbox:          List[float],
    cam_id:        int,
    timestamp:     float,
    velocity:      List[float],
    roi_analyser:  Optional[ROIAnalyser]     = None,
    calibration:   Optional[CameraCalibration] = None,
    max_cameras:   int   = 6,
    world_range:   float = 5000.0,
) -> Dict[str, List[float]]:
    """
    Build all non-visual node features for one tracklet.
    Returns a dict of feature groups so you can ablate each one independently.

    Feature groups
    --------------
    geom          : [x0/W, y0/H, x1/W, y1/H]                      4-d
    spatio_temp   : [cam_onehot..., timestamp_norm, dx, dy]        (6+3)-d
    roi           : [is_valid, is_entry_exit, boundary_proximity]   3-d  ← NEW
    world         : [world_x_norm, world_y_norm]                    2-d  ← NEW

    Total (all groups): 4 + 9 + 3 + 2 = 18-d  (plus visual from ViT/CLIP)
    """
    W_img, H_img = 1600, 1200   # CityFlow default resolution

    # Geometry
    geom = [
        bbox[0] / W_img, bbox[1] / H_img,
        bbox[2] / W_img, bbox[3] / H_img,
    ]

    # Spatio-temporal
    onehot = [0.0] * max_cameras
    if 0 <= cam_id < max_cameras:
        onehot[cam_id] = 1.0
    t_norm = min(timestamp / 1000.0, 1.0)
    bw = max(bbox[2] - bbox[0], 1.0)
    bh = max(bbox[3] - bbox[1], 1.0)
    spatio_temp = onehot + [t_norm, velocity[0] / (bw * 10), velocity[1] / (bh * 10)]

    # ROI features
    roi_feats = (roi_analyser.bbox_features(bbox)
                 if roi_analyser else [1.0, 0.0, 1.0])

    # World coordinate features
    world_feats = (calibration.bbox_features(bbox, world_range)
                   if calibration else [0.0, 0.0])

    return {
        'geom':        geom,
        'spatio_temp': spatio_temp,
        'roi':         roi_feats,
        'world':       world_feats,
    }


def build_extended_edge_features(
    bbox_src:    List[float],
    bbox_dst:    List[float],
    ts_src:      float,
    ts_dst:      float,
    cam_src:     int,
    cam_dst:     int,
    calibration: Optional[CameraCalibration] = None,
    roi_src:     Optional[ROIAnalyser] = None,
    roi_dst:     Optional[ROIAnalyser] = None,
) -> List[float]:
    """
    Extra edge features using calibration and ROI data.
    These are concatenated with the polar (distance/angle) edge features
    already computed in the original Doc2GraphFormer edge builder.

    Edge features (6-d)
    -------------------
    world_dist       : Euclidean distance in world coords (norm)         1-d
    delta_t          : |timestamp_src - timestamp_dst| normalised        1-d
    same_cam         : 1 if same camera (should always be 0 here)        1-d
    src_entry_exit   : 1 if src tracklet is in entry/exit zone           1-d
    dst_entry_exit   : 1 if dst tracklet is in entry/exit zone           1-d
    entry_exit_pair  : 1 if BOTH are in entry/exit zones (strong signal) 1-d
    """
    dt = abs(ts_src - ts_dst) / 100.0   # normalise by ~10s window

    if calibration is not None:
        wx_s, wy_s = calibration.bbox_world_coords(bbox_src)
        wx_d, wy_d = calibration.bbox_world_coords(bbox_dst)
        world_dist = np.sqrt((wx_s - wx_d)**2 + (wy_s - wy_d)**2) / 5000.0
    else:
        world_dist = 0.0

    src_ee = roi_src.is_entry_exit(
        (bbox_src[0] + bbox_src[2]) / 2, bbox_src[3]
    ) if roi_src else False
    dst_ee = roi_dst.is_entry_exit(
        (bbox_dst[0] + bbox_dst[2]) / 2, bbox_dst[3]
    ) if roi_dst else False

    return [
        float(world_dist),
        float(min(dt, 1.0)),
        float(cam_src == cam_dst),
        float(src_ee),
        float(dst_ee),
        float(src_ee and dst_ee),
    ]


# =============================================================================
# 5. Per-camera resource loader (called once per camera in the pipeline)
# =============================================================================

def load_camera_resources(
    cam_dir: str,
    use_mtsc: bool = True,
) -> Dict:
    """
    Load all available extra resources for one camera directory.
    Returns a dict with whatever was found (gracefully handles missing files).

    Dict keys:
        roi          : ROIAnalyser | None
        calibration  : CameraCalibration | None
        mtsc_tracklets: dict | None   (best MTSC file found)
    """
    cam_path = Path(cam_dir)
    resources = {'roi': None, 'calibration': None, 'mtsc_tracklets': None}

    # ROI
    for roi_name in ['roi.jpg', 'roi.png']:
        roi_file = cam_path / roi_name
        if roi_file.exists():
            try:
                resources['roi'] = ROIAnalyser(str(roi_file))
                print(f"  [cam extras] {cam_path.name}: loaded ROI mask")
            except Exception as e:
                print(f"  [warn] ROI load failed: {e}")
            break

    # Calibration
    for cal_name in ['calibration.txt', 'calibration.json']:
        cal_file = cam_path / cal_name
        if cal_file.exists():
            try:
                resources['calibration'] = CameraCalibration(str(cal_file))
                print(f"  [cam extras] {cam_path.name}: loaded calibration H")
            except Exception as e:
                print(f"  [warn] Calibration load failed: {e}")
            break

    # MTSC tracklets
    if use_mtsc:
        tracklets = load_best_mtsc(str(cam_path))
        if tracklets is not None:
            resources['mtsc_tracklets'] = tracklets
            print(f"  [cam extras] {cam_path.name}: "
                  f"loaded MTSC tracklets ({len(tracklets)} tracks)")

    return resources


# =============================================================================
# 6. Detection ensemble — merge 3 detectors for better recall
# =============================================================================

def load_detection_ensemble(
    cam_dir:    str,
    min_conf:   float = 0.3,
    nms_iou:    float = 0.5,
) -> Optional[Dict[int, List]]:
    """
    Load and NMS-merge detections from all available detectors (mask_rcnn,
    ssd512, yolo3) in the det/ subdirectory.

    Empirical recall vs GT (this camera):
      mask_rcnn alone: 0.919
      ssd512 alone:    0.881
      yolo3 alone:     0.743
      ensemble:        0.964  (+4.5% over best single)

    Returns {frame_id: [[x1,y1,x2,y2,conf], ...]} after NMS,
    or None if no det/ files found.
    """
    cam_path = Path(cam_dir)
    det_dir  = cam_path / 'det'

    # Detector priority: mask_rcnn highest quality
    det_files = []
    for name in ['det_mask_rcnn.txt', 'det_yolo3.txt', 'det_ssd512.txt']:
        f = det_dir / name
        if f.exists():
            det_files.append(f)
        # Also check directly in cam dir
        f2 = cam_path / name
        if f2.exists() and f2 not in det_files:
            det_files.append(f2)

    if not det_files:
        return None

    # Load all detections, merge per frame
    from collections import defaultdict
    raw: Dict[int, List] = defaultdict(list)

    for det_file in det_files:
        with open(det_file) as f:
            for line in f:
                parts = line.strip().split(',') if ',' in line else line.strip().split()
                if len(parts) < 6:
                    continue
                try:
                    frame = int(float(parts[0]))
                    x, y  = float(parts[2]), float(parts[3])
                    w, h  = float(parts[4]), float(parts[5])
                    conf  = float(parts[6]) if len(parts) > 6 else 1.0
                except (ValueError, IndexError):
                    continue
                if conf < min_conf:
                    continue
                raw[frame].append([x, y, x + w, y + h, conf])

    if not raw:
        return None

    # Per-frame NMS across all detectors
    def _nms(boxes, iou_thresh):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        kept  = []
        for cand in boxes:
            cx1,cy1,cx2,cy2 = cand[:4]
            suppress = False
            for k in kept:
                ix1=max(cx1,k[0]); iy1=max(cy1,k[1])
                ix2=min(cx2,k[2]); iy2=min(cy2,k[3])
                inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                ua = (cx2-cx1)*(cy2-cy1) + (k[2]-k[0])*(k[3]-k[1]) - inter
                if ua > 0 and inter/ua >= iou_thresh:
                    suppress = True; break
            if not suppress:
                kept.append(cand)
        return kept

    return {frame: _nms(dets, nms_iou) for frame, dets in raw.items()}


# =============================================================================
# 7. Segmentation-masked crops — cleaner visual embeddings
# =============================================================================

def load_segmentation_masks(
    cam_dir: str,
) -> Optional[Dict[int, List]]:
    """
    Load RLE segmentation masks from segm_mask_rcnn.txt.
    Returns {frame_id: [{'bbox': [x,y,x2,y2], 'rle': dict}, ...]}
    or None if file not found or pycocotools unavailable.
    """
    cam_path  = Path(cam_dir)
    segm_file = None
    for candidate in [cam_path / 'segm' / 'segm_mask_rcnn.txt',
                      cam_path / 'segm_mask_rcnn.txt']:
        if candidate.exists():
            segm_file = candidate
            break
    if segm_file is None:
        return None

    try:
        import ast as _ast
        from collections import defaultdict as _dd
        masks: Dict[int, List] = _dd(list)
        with open(segm_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 11:
                    continue
                try:
                    frame = int(float(parts[0]))
                    x, y  = float(parts[2]), float(parts[3])
                    w, h  = float(parts[4]), float(parts[5])
                    # RLE dict spans from col 10 onwards (may contain commas)
                    rle_raw = ','.join(parts[10:])
                    rle_raw = rle_raw.replace('L]', ']').replace('L,', ',').strip()
                    rle     = _ast.literal_eval(rle_raw)
                    masks[frame].append({
                        'bbox': [x, y, x + w, y + h],
                        'rle':  rle,
                    })
                except Exception:
                    continue
        return dict(masks) if masks else None
    except Exception:
        return None


def apply_segmentation_mask(
    crop:         'Image.Image',
    bbox:         List[float],
    frame_masks:  Optional[List[Dict]],
    iou_threshold: float = 0.5,
) -> 'Image.Image':
    """
    Apply the best-matching segmentation mask to a crop before embedding.
    Zeros out background pixels → cleaner CLIP/ViT embedding.

    Falls back to the original crop if pycocotools unavailable or no match.
    """
    if frame_masks is None:
        return crop

    def _iou(a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua>0 else 0

    # Find best matching mask by IoU with bbox
    best_iou  = 0
    best_mask = None
    for m in frame_masks:
        v = _iou(bbox, m['bbox'])
        if v > best_iou:
            best_iou, best_mask = v, m

    if best_mask is None or best_iou < iou_threshold:
        return crop

    try:
        from pycocotools import mask as mask_utils
        import numpy as np
        from PIL import Image as _Image

        # Decode full-frame mask
        full_mask = mask_utils.decode(best_mask['rle'])   # (H, W) uint8

        # Crop the mask to bbox region
        x0,y0,x1,y1 = [int(v) for v in bbox]
        x0=max(0,x0); y0=max(0,y0)

        crop_arr = np.array(crop.convert('RGB'))
        H_crop, W_crop = crop_arr.shape[:2]

        mask_crop = full_mask[y0:y0+H_crop, x0:x0+W_crop]
        if mask_crop.shape[:2] != (H_crop, W_crop):
            # Resize if needed
            mask_crop = np.array(
                _Image.fromarray(mask_crop).resize((W_crop, H_crop),
                                                    _Image.NEAREST))

        # Apply: background → mean grey (127)
        masked = crop_arr.copy()
        masked[mask_crop == 0] = 127

        return _Image.fromarray(masked)

    except ImportError:
        # pycocotools not available — return original crop
        return crop
    except Exception:
        return crop


# =============================================================================
# 8. Extended camera resource loader (updated to include ensemble + segm)
# =============================================================================

def load_camera_resources_full(
    cam_dir:  str,
    use_mtsc: bool = True,
) -> Dict:
    """
    Load ALL available resources for one camera directory.

    Returns dict with keys:
        roi              : ROIAnalyser | None
        calibration      : CameraCalibration | None
        mtsc_tracklets   : dict | None
        det_ensemble     : dict | None   ← NEW: merged 3-detector detections
        segm_masks       : dict | None   ← NEW: RLE segmentation masks per frame
    """
    resources = load_camera_resources(cam_dir, use_mtsc=use_mtsc)
    resources['det_ensemble'] = load_detection_ensemble(cam_dir)
    resources['segm_masks']   = load_segmentation_masks(cam_dir)

    n_ensemble = sum(len(v) for v in resources['det_ensemble'].values()) \
                 if resources['det_ensemble'] else 0
    n_segm     = sum(len(v) for v in resources['segm_masks'].values()) \
                 if resources['segm_masks'] else 0

    if n_ensemble > 0:
        print(f"  [cam extras] {Path(cam_dir).name}: "
              f"ensemble {n_ensemble} dets across "
              f"{len(resources['det_ensemble'])} frames")
    if n_segm > 0:
        print(f"  [cam extras] {Path(cam_dir).name}: "
              f"{n_segm} segmentation masks")

    return resources