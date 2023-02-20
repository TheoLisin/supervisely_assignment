from pathlib import Path
import cv2
from typing import List

from module.scripts.davis_merge.videomaker import (
    _normalize_frame_size,
    sequence_to_video,
)


def test_size_normalization(samples: List[cv2.Mat]):
    h = max([s.shape[0] for s in samples])
    w = max([s.shape[1] for s in samples])
    size = (h, w, 3)

    for sample in samples:
        assert _normalize_frame_size(sample, size).shape == size


def test_video_length(samples: List[cv2.Mat], tmp_path: Path):
    h = max([s.shape[0] for s in samples])
    w = max([s.shape[1] for s in samples])
    size = (h, w, 3)

    sequence_to_video(iter(samples), tmp_path, "test_len", size)

    cap = cv2.VideoCapture(str(tmp_path / "test_len.mp4"))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert frame_count == len(samples)
