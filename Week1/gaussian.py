from typing import Tuple, List
import argparse
import os
import cv2
import numpy as np
import json

from tqdm import tqdm

from utils import get_video_props, read_frames, open_video_writer


def compute_gaussian_model(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given stacked grayscale frames shape (N,H,W), return mean and var arrays (H,W) float32."""
    frames_f = frames.astype(np.float32)
    mean = np.mean(frames_f, axis=0)
    var = np.var(frames_f, axis=0)
    return mean.astype(np.float32), var.astype(np.float32)


def make_bounding_box(mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int]]:
    """Compute bounding boxes from a binary mask.

    Args:
        mask: Binary mask as a numpy array. Can be bool, 0/1, or 0/255 uint8.
        min_area: Minimum area in pixels for a box to be returned (filters noise).

    Returns:
        List of boxes as (x_min, y_min, x_max, y_max).
    """
    mask_u = (mask.astype(np.uint8) * 255)

    # find contours and bounding rects
    contours, _ = cv2.findContours(mask_u.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h))
    return boxes


def process_video(input_path: str,
                  pct_train: float = 25.0,
                  alphas: List[float] = [2.5],
                  fourcc: str = 'mp4v',
                  morph: bool = True):

    cap0 = cv2.VideoCapture(input_path)
    if not cap0.isOpened():
        raise IOError(f"Cannot open video: {input_path}")
    total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    pct_train = float(pct_train)
    if pct_train <= 0:
        raise ValueError("pct_train must be > 0")
    if pct_train > 100:
        pct_train = 100.0

    N = max(1, int(np.round(total_frames * (pct_train / 100.0))))
    cap0.release()

    gray_frames = []
    for f in read_frames(input_path, max_frames=N, gray=True):
        gray_frames.append(f)
    if len(gray_frames) == 0:
        raise ValueError("Input video contains no frames")

    stack = np.stack(gray_frames, axis=0)
    mean, var = compute_gaussian_model(stack)
    std = np.sqrt(var)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, N)

    _, fps, width, height = get_video_props(cap)
    frame_size = (width, height)

    # writers
    mask_writers = {}
    fg_writers = {}

    for alpha in alphas:
        mask_writers[alpha] = open_video_writer(
            f"T1_mask_{alpha}.mp4", fourcc, fps, frame_size, True
        )
        fg_writers[alpha] = open_video_writer(
            f"T1_fg_{alpha}.mp4", fourcc, fps, frame_size, True
        )

    frame_idx = N

    preds_dict = {alpha: {} for alpha in alphas}

    for _ in tqdm(range(N, total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # COMPUTATION
        # we could do  (R+G+B)/3 instead of just gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        diff = np.abs(gray - mean)

        alphas_np = np.array(alphas, dtype=np.float32).reshape(-1, 1, 1)
        thresholds = alphas_np * (std + 2)
        diff_expanded = diff[None, :, :]   # shape (1, H, W)

        fg_masks_all = diff_expanded > thresholds

        for idx, alpha in enumerate(alphas):
            fg_mask = fg_masks_all[idx]

            mask_img = (fg_mask.astype('uint8') * 255)

            # MORPHOLOGY
            if morph:
                mask_img = cv2.medianBlur(mask_img, 5)

                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel_open)

                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
                mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel_close)
                mask_img = cv2.dilate(mask_img, kernel_close, iterations=1)

            mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            fg_mask2 = (mask_img > 0)

            # BOXES
            boxes = make_bounding_box(mask_img, min_area=70*70)

            if len(boxes) > 0:
                preds_dict[alpha][str(frame_idx)] = [
                    [int(x1), int(y1), int(x2), int(y2), 1.0]
                    for (x1, y1, x2, y2) in boxes
                ]

            alpha_overlay = 0.5
            overlay = np.zeros_like(frame, dtype=frame.dtype)
            overlay[:] = (0, 0, 255)
            blended = cv2.addWeighted(frame, 1.0 - alpha_overlay, overlay, alpha_overlay, 0)
            fg_frame = np.where(fg_mask2[:, :, None], blended, frame)

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cv2.rectangle(mask_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(fg_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(fg_frame, f"{i+1}", (x1, max(y1-6,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,0), 1, cv2.LINE_AA)

            mask_writers[alpha].write(mask_bgr)
            fg_writers[alpha].write(fg_frame)

        frame_idx += 1

    cap.release()
    for alpha in alphas:
        mask_writers[alpha].release()
        fg_writers[alpha].release()
        out_pred = f"T1_preds_{alpha}.json"
        with open(out_pred, 'w') as f:
            json.dump(preds_dict[alpha], f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian background extractor")
    p.add_argument('input', help='Input video path')
    p.add_argument('--pct-train', type=float, default=25.0, help='Percentage of frames to build the model (0-100)')
    p.add_argument('--alpha', type=float, nargs='+', default=[2.5],
                   help='One or more threshold values (e.g. --alpha 2.0 2.5 3.0)')
    p.add_argument('--fourcc', default='mp4v', help="FourCC code for output video (default 'mp4v')")
    p.add_argument('--no-morph', action='store_true', help='Disable morphological filtering of the mask')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    base, _ = os.path.splitext(os.path.basename(args.input))

    process_video(
        args.input,
        pct_train=args.pct_train,
        alphas=args.alpha,
        fourcc=args.fourcc,
        morph=(not args.no_morph)
    )
