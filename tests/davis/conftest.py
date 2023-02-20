import cv2
import pytest
import numpy as np


@pytest.fixture
def samples():
    """Create sequence of frames with different shapes."""
    spls = []
    for _ in range(10):
        spls.append(np.random.randint(0, 255, size=(100, 50, 3), dtype=np.uint8))

    for _ in range(10):
        spls.append(np.random.randint(0, 255, size=(50, 100, 3), dtype=np.uint8))

    for _ in range(10):
        spls.append(np.random.randint(0, 255, size=(30, 30, 3), dtype=np.uint8))

    return spls
