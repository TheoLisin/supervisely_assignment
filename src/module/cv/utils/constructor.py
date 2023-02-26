import os
import cv2
import numpy as np
import imutils
from numpy.random import Generator
from pathlib import Path
from typing import Dict, List, Tuple, TypeVar

from module.cv.utils.img_data import ClassProps, PrimitiveObject, GenParams

AnyT = TypeVar("AnyT")


def add_gausian_light(
    img: cv2.Mat,
    center: Tuple[int, int],
    x_sigma: float,
    y_sigma: float,
    rotation: int,
    max_intensity: float,
    color: int,
) -> cv2.Mat:
    """Add light Gaussian spot to image.

    Args:
        img (cv2.Mat): raw image
        center (Tuple[int, int]): spot center
        x_sigma (float): gaussian sigma for x
        y_sigma (float): gaussian sigma for y
        rotation (int): degree of spot rotation
        max_intensity (float): max spot intensity in center (from 0 to 1)
        color (int): light will have color (color, color, color)

    Returns:
        cv2.Mat: image with gausian light.
    """
    max_intensity = max(0, min(1, max_intensity))
    rotation = np.deg2rad(rotation)

    def rotate_axis(
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        radian: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_r = x_vec * np.cos(radian) - y_vec * np.sin(radian)
        y_r = x_vec * np.sin(radian) + y_vec * np.cos(radian)

        return x_r, y_r

    def gaus(
        mat: np.ndarray,
        sigma: float,
        shift: np.ndarray,
    ) -> np.ndarray:  # noqa: WPS430
        epow = -((mat - shift) ** 2) / sigma**2
        return 1 / abs(sigma) * np.exp(epow)

    size = img.shape
    light = np.ones_like(img) * color
    x_linspace = np.arange(0, size[1], 1)
    y_linspace = np.arange(0, size[0], 1)[:, np.newaxis]

    x_rot, y_rot = rotate_axis(x_linspace, y_linspace, rotation)
    x_sh, y_sh = rotate_axis(np.array([center[0]]), np.array([center[1]]), rotation)

    x_gaus = gaus(x_rot, x_sigma, x_sh)
    y_gaus = gaus(y_rot, y_sigma, y_sh)

    mask = x_gaus * y_gaus
    mask /= np.max(mask)
    mask = mask * (mask > (1 / 255)).astype(int) * max_intensity
    mask = mask[:, :, None]

    ligth_img = img * (1 - mask) + light * mask

    return np.uint8(ligth_img)


def rotate_and_resize(img: cv2.Mat, angle: int, size_factor: float) -> cv2.Mat:
    """Rotate and resize (downscale) image.

    Args:
        img (cv2.Mat): raw image
        angle (int): rotation angle
        size_factor (float): the number by which
            the original size is multiplied; must be in [0.1, 1]

    Returns:
        cv2.Mat: rotated and resized image
    """
    size_factor = max(0.1, min(1, size_factor))
    h = int(img.shape[0] * size_factor)
    w = int(img.shape[1] * size_factor)

    rot_img = imutils.rotate(img, angle=angle)
    res_img = cv2.resize(rot_img, (w, h))

    return res_img


def add_annotation(current_annot: cv2.Mat, added_primitive: PrimitiveObject) -> cv2.Mat:
    """Create new annotation overlapping the old one.

    Args:
        current_annot (cv2.Mat): RGBA annotation
        added_primitive (PrimitiveObject): last added primitive

    Returns:
        cv2.Mat: new annotation
    """
    alpha = added_primitive.rgba_img[:, :, 3]
    mask = (alpha > 0).astype(np.uint8)
    r, g, b = added_primitive.class_props.color
    annot = cv2.merge([mask * r, mask * g, mask * b, mask * 255])
    new_annot = np.where(mask[:, :, None], annot, current_annot)

    return new_annot.astype(np.uint8)


def add_primitive(back: cv2.Mat, primitive: PrimitiveObject) -> cv2.Mat:
    """Add primitive to background."""
    assert back.shape[:2] == primitive.rgba_img.shape[:2]

    mask = (primitive.rgba_img[:, :, 3] > 0).astype(np.uint8)
    mask = mask[:, :, None]
    prim = primitive.rgba_img[:, :, :3]
    return back * (1 - mask) + prim * mask


def random_crop(img: cv2.Mat, shape: Tuple[int, int], gen: Generator) -> cv2.Mat:
    """Сuts a random piece of a given shape."""
    h, w = img.shape[:2]
    nh, nw = shape
    y = gen.integers(0, h - nh)
    x = gen.integers(0, w - nw)
    return img[y : y + nh, x : x + nw]


def random_sample(arr: List[AnyT], gen: Generator) -> AnyT:
    ind = gen.integers(0, len(arr))
    return arr[ind]


def random_border(img: cv2.Mat, size: Tuple[int, int], gen: Generator) -> cv2.Mat:
    h, w = size
    cur_h, cur_w = img.shape[:2]
    dh = h - cur_h
    dw = w - cur_w
    top = gen.integers(0, dh + 1)
    btm = dh - top
    left = gen.integers(0, dw + 1)
    rgh = dw - left

    return cv2.copyMakeBorder(
        img,
        top=top,
        bottom=btm,
        left=left,
        right=rgh,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def random_change_prim(
    primitive: cv2.Mat,
    gen_params: GenParams,
    final_size: Tuple[int, int],
) -> cv2.Mat:
    """Rotate and resize primitive; 
    returns a primitive whose (transparent) 
    background is stretched to the given size."""
    nh, nw = final_size
    angle = gen_params.generator.integers(*gen_params.rotate)
    size_factor = gen_params.generator.uniform(*gen_params.size_factor)
    primitive = rotate_and_resize(primitive, angle, size_factor)
    primitive = random_border(primitive, (nh, nw), gen=gen_params.generator)
    return primitive


def random_class_random_image_load(
    classes_path: List[str],
    cls_colors: Dict[str, Tuple[int, int, int]],
    gen: Generator,
) -> Tuple[cv2.Mat, ClassProps]:
    prim_cls = Path(random_sample(classes_path, gen))
    name = prim_cls.name
    cp = ClassProps(name, color=cls_colors[name])
    imgs = os.listdir(prim_cls)
    img_name = random_sample(imgs, gen)
    return cv2.imread(str(prim_cls / img_name), cv2.IMREAD_UNCHANGED), cp


def generate_pic(
    classes_path: List[str],
    back_paths: List[str],
    gen_params: GenParams,
    cls_colors: Dict[str, Tuple[int, int, int]],
) -> Tuple[cv2.Mat, cv2.Mat]:
    """Generate new picture and ground truth annotation."""
    bg_path = random_sample(back_paths, gen=gen_params.generator)
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)

    nh = gen_params.generator.integers(*gen_params.h_limits)
    nw = gen_params.generator.integers(*gen_params.w_limits)

    bg = random_crop(bg, (nh, nw), gen_params.generator)
    annot = np.zeros((nh, nw, 4), dtype=np.uint8)
    prim_num = gen_params.generator.integers(1, max(2, gen_params.prim_limit + 1))

    for _ in range(prim_num):
        primitive, class_props = random_class_random_image_load(
            classes_path, cls_colors, gen=gen_params.generator
        )
        primitive = imutils.resize(primitive, width=nw)
        primitive = random_change_prim(primitive, gen_params, (nh, nw))
        pobj = PrimitiveObject(primitive, class_props)
        annot = add_annotation(annot, pobj)
        bg = add_primitive(bg, pobj)

    if gen_params.gaus_blur is not None:
        bg = cv2.GaussianBlur(bg, gen_params.gaus_blur, cv2.BORDER_DEFAULT)
    
    # add light
    cx = gen_params.generator.integers(50, nw - 50)
    cy = gen_params.generator.integers(50, nw - 50)
    rot = gen_params.generator.integers(*gen_params.rotate)
    inten = gen_params.generator.random() * 0.4 + 0.5
    bg = add_gausian_light(bg, (cx, cy), nw / 2, nh // 2, rot, inten, 255)

    return bg, annot
