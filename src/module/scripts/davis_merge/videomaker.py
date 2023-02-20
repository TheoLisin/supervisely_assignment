import cv2
from typing import Iterator, Tuple
from pathlib import Path
from tqdm import tqdm
from logging import getLogger

from module.scripts.davis_merge.annotation import PathT

logger = getLogger(__name__)


def sequence_to_video(
    frames: Iterator[cv2.Mat],
    save_path: PathT,
    name: str,
    size: Tuple[int, int, int],
    framerate: float = 15.0,
):
    """Build video from sequence.

    Args:
        frames (List[cv2.Mat]): frames sorted by time
        save_path (PathT): path to save video
        name (str): name of file
        size (Tuple[int, int, int]): size of frame in final video (h, w, ch).
            All frames will be reduced to this size.
        framerate (float): Defaults to 15.0.
    """
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    save_path_with_name = str(Path(save_path) / f"{name}.mp4")
    writer = cv2.VideoWriter(save_path_with_name, fourcc, framerate, (size[1], size[0]))

    for frame in tqdm(frames, desc="Creation awesome video:"):
        if frame.shape != size:
            frame = _normalize_frame_size(frame, size)
        writer.write(frame)

    writer.release()
    logger.info("Video is ready!")


def _normalize_frame_size(frame: cv2.Mat, size: Tuple[int, int, int]) -> cv2.Mat:
    h, w, _ = frame.shape
    max_h, max_w, _ = size

    h_diff = max_h - h
    w_diff = max_w - w

    left, top = w_diff // 2, h_diff // 2
    right, bottom = w_diff - left, h_diff - top

    return cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0,
    )
