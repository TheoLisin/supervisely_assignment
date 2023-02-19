import os
import cv2
import numpy as np
from typing import List, Union
from pathlib import Path

PathT = Union[str, Path]


def add_annotation_on_image(
    image_path: PathT,
    annot_path: PathT,
    alpha: float = 0.5,
    border_thickness: int = 1,
) -> cv2.Mat:
    """Add annotation over image.

    Args:
        image_path (PathT): path to image
        annot_path (PathT): path to annotation (.png image)
        alpha (float, optional): transparency factor. Defaults to 0.5.
        border_thickness (int, optional): border thickness (px). Defaults to 1.

    Returns:
        cv2.Mat: image with annotation.
    """
    background = cv2.imread(str(image_path))
    annot = cv2.imread(str(annot_path), cv2.IMREAD_UNCHANGED)

    annot_gs = cv2.cvtColor(annot, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(annot_gs, 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    background = np.array(background)
    annot = np.array(annot)

    transparent_mask = np.sum(annot, axis=-1) > 0
    transparent_mask = transparent_mask.astype(int) * alpha
    transparent_mask = transparent_mask[:, :, None]

    background = background * (1 - transparent_mask) + annot * transparent_mask

    background = cv2.drawContours(background, contours, -1, (1, 1, 1), border_thickness)

    return np.uint8(background)


def annotate_image_sequence(
    img_path: PathT,
    annot_path: PathT,
    alpha: float = 0.5,
    border_thickness: int = 1,
) -> List[cv2.Mat]:
    """Annotate every image in img_path folder. Return list sorted by image name.

    Args:
        img_path (PathT): folder with raw images.
        annot_path (PathT): folder with annotatins.
            Name of image and annotation must be equal.
        alpha (float, optional): transparency factor. Defaults to 0.5.
        border_thickness (int, optional): border thickness (px). Defaults to 1.

    Returns:
        List[cv2.Mat]: List of annotated images sorted by name.
    """
    img_path = Path(img_path)
    annot_path = Path(annot_path)
    images = sorted(os.listdir(img_path))
    annots = sorted(os.listdir(annot_path))
    img_with_annot = []

    for img, annot in zip(images, annots):
        img_with_annot.append(
            add_annotation_on_image(
                img_path / img,
                annot_path / annot,
                alpha,
                border_thickness,
            ),
        )

    return img_with_annot
