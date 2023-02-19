import numpy as np
import re
from logging import getLogger
from dataclasses import dataclass, field
from typing import Generator, Union, Optional, Dict, Tuple, List
from pathlib import Path
from PIL.Image import open as open_img, fromarray

PathT = Union[str, Path]
NAME_REGX = r"(?P<name>\w+)\_(win_(?P<win>(\d+_\d+))_sh_(?P<sh>(\d+_\d+))_pos_(?P<pos>(\d+_\d+))).*"  # noqa: E501
COMP_REGX = re.compile(NAME_REGX)

logger = getLogger(__name__)


@dataclass
class SliceParams:
    """Slice parametrs dataclass"""
    pos: Tuple[int, int]
    name: str
    img_path: Path = field(init=False, repr=False, compare=False)
    win_shape: Tuple[int, int]
    shift_shape: Tuple[int, int]
    img_size: Tuple[int, int] = field(init=False, compare=False)


def merge_image(
    splits_path: PathT,
    save_path: PathT,
    original: Optional[PathT] = None,
    img_format: str = "png",
) -> None:
    """Merge image from splits and compare with original.

    Args:
        splits_path (PathT): folder with splits
        save_path (PathT): path to save restored image
        original (Optional[PathT], optional): original image for comparison.
            Defaults to None.
        img_format (str): Save format.
            Image may changes after being saved
            due to compression. To avoid this use .png.
            Defaults to "png".

    Raises:
        ValueError: image can't be restored if shift size > sliding window.
        ValueError: original image doesn't match with restored
    """
    base_params, row_image = next(_split_read_generator(splits_path))
    win_h, win_w = base_params.win_shape
    shift_h, shift_w = base_params.shift_shape
    max_h, max_w = base_params.img_size
    rows = []

    for sp, img in _split_read_generator(splits_path):
        pos_y, pos_x = sp.pos

        if (win_h < shift_h) or (win_w < shift_w):
            raise ValueError(
                "Can't restore image: shift step is bigger then slice window."
            )

        row_image = _concate_split_over_axis(
            cur_image=row_image,
            next_slice=img,
            window_side_len=win_w,
            max_val=max_w,
            cur_pos=pos_x,
            shift_win_diff=win_w - shift_w,
        )

        if row_image.shape[1] == max_w:
            rows.append((pos_y, row_image))

    reconstrucrted_img = rows[0][1]

    for y, row in rows:
        reconstrucrted_img = _concate_split_over_axis(
            cur_image=reconstrucrted_img,
            next_slice=row,
            window_side_len=win_h,
            max_val=max_h,
            cur_pos=y,
            shift_win_diff=win_h - shift_h,
            axis=0,
        )

    if original is not None:
        orig_image = np.array(open_img(original))
        if not (orig_image == reconstrucrted_img).all():
            raise ValueError("Original image doesn't equal to reconstructed.")
        if img_format != "png":
            logger.warning(
                "Images may change after being saved due to compression. Use a .png format"
            )

    img_to_save = fromarray(reconstrucrted_img)
    img_to_save.save(Path(save_path) / f"{base_params.name}.{img_format}")


def _concate_split_over_axis(
    cur_image: np.ndarray,
    next_slice: np.ndarray,
    window_side_len: int,
    max_val: int,
    cur_pos: int,
    shift_win_diff: int,
    axis: int = 1,
) -> np.ndarray:
    sl = [slice(None) for _ in range(next_slice.ndim)]  # for control axis in slice
    if cur_pos == 0:
        return next_slice
    elif cur_pos + window_side_len == max_val:  # last slice in row
        fin_pos = window_side_len - (max_val - cur_image.shape[axis])
        sl[axis] = slice(fin_pos, window_side_len)
        return np.concatenate((cur_image, next_slice[tuple(sl)]), axis=axis)

    sl[axis] = slice(shift_win_diff, window_side_len)
    return np.concatenate((cur_image, next_slice[tuple(sl)]), axis=axis)


def _split_read_generator(
    splits_path: PathT,
) -> Generator[Tuple[SliceParams, np.ndarray], None, None]:
    """Load and yield next split in (Left->Right, Up->Down) order."""
    if isinstance(splits_path, str):
        splits_path = Path(splits_path)

    splits_params: List[SliceParams] = []

    for split in splits_path.iterdir():
        sp = _get_params(split.name)
        sp.img_path = split
        splits_params.append(sp)

    splits_params = sorted(splits_params, key=lambda x: x.pos)
    max_h, max_w = splits_params[-1].pos
    max_h += splits_params[-1].win_shape[0]
    max_w += splits_params[-1].win_shape[1]

    for cur_params in splits_params:
        cur_params.img_size = (max_h, max_w)
        img = np.array(open_img(cur_params.img_path))
        yield cur_params, img


def _get_params(img_name: str) -> SliceParams:
    parsed = re.search(COMP_REGX, img_name)
    split_params: Dict[str, Tuple[int, int]] = {}

    def parse_x_y(raw: str) -> Tuple[int, int]:
        raw_splitted = raw.split("_")
        y, x = (int(i) for i in raw_splitted)
        return y, x  # noqa: WPS331

    if parsed is not None:
        split_params["win_shape"] = parse_x_y(parsed.group("win"))
        split_params["shift_shape"] = parse_x_y(parsed.group("sh"))
        split_params["pos"] = parse_x_y(parsed.group("pos"))
    else:
        raise ValueError("Wrong image name.")

    return SliceParams(**split_params, name=parsed.group("name"))
