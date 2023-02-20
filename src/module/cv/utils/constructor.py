from typing import Tuple
import cv2
import numpy as np


def add_gausian_light(
    img: cv2.Mat,
    center: Tuple[int, int],
    x_sigma: float,
    y_sigma: float,
    rotation: int,
    max_intensity: float,
) -> cv2.Mat:
    """Add light Gaussian spot to image.

    Args:
        img (cv2.Mat): raw image
        center (Tuple[int, int]): spot center
        x_sigma (float): gaussian sigma for x
        y_sigma (float): gaussian sigma for y
        rotation (int): degree of spot rotation
        max_intensity (float): max spot intensity in center (from 0 to 1)

    Returns:
        cv2.Mat: _description_
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
    light = np.ones_like(img) * 255
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
