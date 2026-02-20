from typing import Tuple
import argparse
import os
import cv2
import numpy as np

from tqdm import tqdm

from utils import get_video_props, read_frames, open_video_writer


def compute_gaussian_model(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given stacked grayscale frames shape (N,H,W), return mean and var arrays (H,W) float32."""
    # ensure float for accurate mean/var
    frames_f = frames.astype(np.float32)
    mean = np.mean(frames_f, axis=0)
    var = np.var(frames_f, axis=0)
    return mean.astype(np.float32), var.astype(np.float32)


def process_video(input_path: str, out_mask: str, out_fg: str, pct_train: float = 10.0, alpha: float = 2.5, fourcc: str = 'mp4v'):
    """Process the video.

    pct_train: percentage of frames (0-100) to use for building the model from the start of the video.
    """
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

    _, fps, width, height = get_video_props(cap)
    frame_size = (width, height)

    # writers
    mask_writer = open_video_writer(out_mask, fourcc, fps, frame_size, is_color=True)
    fg_writer = open_video_writer(out_fg, fourcc, fps, frame_size, is_color=True)

    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        diff = np.abs(gray - mean)
        fg_mask = diff > (alpha * std)

        mask_img = (fg_mask.astype('uint8') * 255)

        mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

        fg_frame = (frame * fg_mask[:, :, None].astype(frame.dtype))

        mask_writer.write(mask_bgr)
        fg_writer.write(fg_frame)

        frame_idx += 1

    cap.release()
    mask_writer.release()
    fg_writer.release()


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian background extractor (first N frames model)")
    p.add_argument('input', help='Input video path')
    p.add_argument('--pct-train', type=float, default=10.0, help='Percentage of frames to build the model (0-100)')
    p.add_argument('--alpha', type=float, default=2.5, help='Threshold in multiples of std dev')
    p.add_argument('--fourcc', default='mp4v', help="FourCC code for output video (default 'mp4v')")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    base, _ = os.path.splitext(os.path.basename(args.input))

    out_mask = f"T1_mask.mp4"
    out_fg = f"T1_fg.mp4"

    process_video(args.input, out_mask, out_fg, pct_train=args.pct_train, alpha=args.alpha, fourcc=args.fourcc)
