import numpy as np
from PIL.Image import Image, open as open_img, fromarray
from typing import Union, Tuple, List, Generator
from logging import getLogger
from pathlib import Path

ImgT = Union[np.ndarray, Image]
ShapeT = Union[int, float]

logger = getLogger(__name__)


def split_image(
    img_path: Union[str, Path],
    win_shape: Tuple[ShapeT, ShapeT],
    shift_shape: Tuple[ShapeT, ShapeT],
    out_path: Union[str, Path],
) -> None:
    """Split image and save to new subfolder.

    Args:
        img_path (Union[str, Path]): path to image
        win_shape (Tuple[ShapeT, ShapeT]): sliding windows shape (height, width)
        shift_shape (Tuple[ShapeT, ShapeT]): shift shape (y-shift, x-shift)
        out_path (Union[str, Path]): folder where to save; 
            splits will be placed into subfolder
    """
    if isinstance(img_path, str):
        img_path = Path(img_path)

    if isinstance(out_path, str):
        out_path = Path(out_path)

    img_name = str(img_path.name).split(".")[0]
    img_format = str(img_path.name).split(".")[-1]

    img = open_img(img_path)
    split_pos = 0

    subfolder = out_path / img_name

    try:
        subfolder.mkdir()
    except FileExistsError:
        logger.warning(
            "The image {name} has already been processed.".format(name=img_name),
        )
        logger.warning("Check the image name or choose a different save directory")

    for split, yi, xi in split_generator(img, win_shape, shift_shape):
        split_name = "{name}_win_{hw}_{ww}_sh_{hs}_{ws}_pos_{yi}_{xi}.{form}".format(
            name=img_name,
            hw=win_shape[0],
            ww=win_shape[1],
            hs=shift_shape[0],
            ws=win_shape[1],
            yi=yi,
            xi=xi,
            form=img_format,
        )
        pil_img = fromarray(split)

        pil_img.save(subfolder / split_name)
        split_pos += 1


def split_generator(
    img: ImgT,
    win_shape: Tuple[ShapeT, ShapeT],
    shift_shape: Tuple[ShapeT, ShapeT],
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """Generate splits for the image with sliding window.

    Args:
        img (ImgT): image (PIL, cv2 or numpy array)
        win_shape (Tuple[ShapeT, ShapeT]): sliding windows shape;
            (height, width) in pixels or %
        shift_shape (Tuple[ShapeT, ShapeT]): shifts for x and y axis
            in pixels or %; the first component is y

    Yields:
        Generator[np.ndarray, None, None]: current split
    """
    npimg = np.array(img)

    if len(npimg.shape) == 3:
        img_h, img_w, _ = npimg.shape
    else:
        img_h, img_w = npimg.shape

    px_h, px_w = _shape_converter(win_shape, img_h, img_w)
    px_h_shift, px_w_shift = _shape_converter(shift_shape, img_h, img_w)

    for yi in _make_grid(img_h, px_h_shift, px_h):
        for xi in _make_grid(img_w, px_w_shift, px_w):
            yield npimg[yi : yi + px_h, xi : xi + px_w], yi, xi


def _shape_converter(
    shapes: Tuple[ShapeT, ShapeT],
    img_h: int,
    img_w: int,
) -> Tuple[int, int]:
    """Checks for type matching and convert shapes to pixels.

    Args:
        shapes (Tuple[ShapeT, ShapeT]): shapes in px or %
        img_h (int): image height
        img_w (int): image width

    Raises:
        TypeError: all shape must be the same type
        ValueError: float shape must be in (0, 1)

    Returns:
        Tuple[int, int]: shapes in pixels
    """
    hshape, wshape = shapes

    if type(hshape) is not type(wshape):
        raise TypeError("Height and width must be the same type float or int.")

    if isinstance(hshape, float):
        if abs(hshape) >= 1 or abs(wshape) >= 1:
            raise ValueError("Float shape must be in (0, 1)")

        px_w = max(1, int(wshape * img_w))
        px_h = max(1, int(hshape * img_h))

        if px_w == 1 or px_h == 1:
            logger.warning("Sliding window/shift shape contains 1px value.")

        return px_w, px_h

    return hshape, wshape


def _make_grid(full_len: int, shift: int, win_len: int) -> List[int]:
    grid = list(range(0, full_len - win_len, shift))
    grid.append(full_len - win_len)
    return grid
