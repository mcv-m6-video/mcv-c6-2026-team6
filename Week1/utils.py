from typing import Generator, Tuple, Optional
import cv2
import numpy as np


def get_video_props(cap: cv2.VideoCapture) -> Tuple[int, float, int, int]:
	"""Return (frame_count, fps, width, height) from an opened VideoCapture."""
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	return frame_count, fps, width, height


def read_frames(path: str, max_frames: Optional[int] = None, gray: bool = True) -> Generator[np.ndarray, None, None]:
	"""Yield frames from video file at `path`.

	Args:
		path: input video path
		max_frames: stop after this many frames (None => all)
		gray: convert frames to grayscale if True
	Yields:
		uint8 ndarray frames (H,W) if gray else (H,W,3)
	"""
	cap = cv2.VideoCapture(path)
	if not cap.isOpened():
		raise IOError(f"Cannot open video: {path}")

	count = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if gray:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		yield frame
		count += 1
		if max_frames is not None and count >= max_frames:
			break

	cap.release()


def open_video_writer(path: str, fourcc: str, fps: float, frame_size: Tuple[int, int], is_color: bool = True) -> cv2.VideoWriter:
	"""Create and return a cv2.VideoWriter.

	fourcc: 4-char code like 'mp4v' or 'XVID'
	frame_size: (width, height)
	"""
	fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
	writer = cv2.VideoWriter(path, fourcc_code, fps, frame_size, isColor=is_color)
	if not writer.isOpened():
		raise IOError(f"Cannot open video writer for {path}")
	return writer
