import cv2
from numpy.random import Generator
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ClassProps(object):
    name: str
    color: Tuple[int, int, int]

    def _validate_color(self):
        for cp in self.color:
            if not (0 <= cp <= 255):
                raise ValueError("Class color is incorrect.")

    def __post_init__(self):
        self._validate_color()


@dataclass
class PrimitiveObject(object):
    rgba_img: cv2.Mat  # rgba image: 4 channels
    class_props: ClassProps


@dataclass
class GenParams(object):
    generator: Generator
    w_limits: Tuple[int, int]
    h_limits: Tuple[int, int]
    prim_limit: int
    size_factor: Tuple[float, float]
    rotate: Tuple[int, int]
    gaus_blur: Optional[Tuple[int, int]]