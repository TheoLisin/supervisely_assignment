import pytest
import numpy as np
import os
from typing import Tuple
from pathlib import Path
from PIL import Image

from module.scripts.splitmerge.split import split_generator, split_image


@pytest.mark.parametrize(
    ("win_shape", "shifts"),
    [
        ((32, 32), (32, 32)),
        ((32, 64), (17, 23)),
        ((800, 800), (3, 3)),
    ],
)
def test_splits_equal_shapes(
    img_sample: np.ndarray,
    win_shape: Tuple[int, int],
    shifts: Tuple[int, int],
):
    """Checks if all splits are the same size"""

    for split, _, _ in split_generator(img_sample, win_shape, shifts):
        assert split.shape == (*win_shape, 3)


@pytest.mark.parametrize(
    ("win_shape", "shifts"),
    [
        ((32, 32), (32, 32)),
        ((32, 64), (17, 23)),
        ((800, 800), (3, 3)),
        ((0.2, 0.2), (0.1, 0.1)),
    ],
)
def test_split_image(
    win_shape: Tuple[int, int],
    shifts: Tuple[int, int],
    lena_path: Path,
    tmp_path: Path,
):
    subfolder = tmp_path / lena_path.name.split(".")[0]
    lena = Image.open(lena_path)
    nplena = np.array(lena)

    fl = nplena.shape[0]
    sh = shifts[0]
    wl = win_shape[0]
    y_pos_num = (fl - abs(sh - wl)) // sh + int(wl >= sh)
    
    fl = nplena.shape[1]
    sh = shifts[1]
    wl = win_shape[1]
    x_pos_num = (fl - abs(sh - wl)) // sh + int(wl >= sh)

    split_image(lena_path, win_shape, shifts, tmp_path)

    assert x_pos_num * y_pos_num == len(os.listdir(subfolder))
