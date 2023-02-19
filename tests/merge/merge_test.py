import pytest
from pathlib import Path
from typing import Tuple

from module.scripts.splitmerge.split import split_image
from module.scripts.splitmerge.merge import _get_params, SliceParams, merge_image


def test_simple_get_params():
    name = "lena_win_1_2_sh_11_12_pos_21_22.jpg"
    ans = {
        "win_shape": (1, 2),
        "shift_shape": (11, 12),
        "pos": (21, 22),
    }

    assert _get_params(name) == SliceParams(**ans)


@pytest.mark.parametrize(
    ("win_shape", "shifts"),
    [
        ((32, 32), (32, 32)),
        ((32, 64), (17, 23)),
        ((800, 800), (3, 3)),
        ((0.2, 0.2), (0.1, 0.1)),
    ],
)
def test_merge_diff_shapes(
    win_shape: Tuple[int, int],
    shifts: Tuple[int, int],
    lena_path: Path,
    tmp_path: Path,
):
    split_image(lena_path, win_shape, shifts, tmp_path)
    merge_image(tmp_path / "lena", tmp_path, lena_path)


@pytest.mark.parametrize(
    ("win_shape", "shifts"),
    [
        ((32, 32), (32, 32)),
        ((32, 64), (17, 23)),
        ((800, 800), (3, 3)),
        ((0.2, 0.2), (0.1, 0.1)),
    ],
)
def test_merge_diff_shapes_one_channel(
    win_shape: Tuple[int, int],
    shifts: Tuple[int, int],
    gslena_path: Path,
    tmp_path: Path,
):
    split_image(gslena_path, win_shape, shifts, tmp_path)
    merge_image(tmp_path / "gslena", tmp_path, gslena_path)


def test_merge_wrong_shapes(
    lena_path: Path,
    tmp_path: Path,
):
    win_shape = (32, 32)
    shifts = (64, 64)
    split_image(lena_path, win_shape, shifts, tmp_path)

    with pytest.raises(
        ValueError, match="Can't restore image: shift step is bigger then slice window.",
    ):
        merge_image(tmp_path / "lena", tmp_path, lena_path)