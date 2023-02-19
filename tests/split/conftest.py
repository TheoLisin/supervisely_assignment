import numpy as np
import pytest


@pytest.fixture
def img_sample() -> np.ndarray:
    return np.random.randint(0, 255, size=(1000, 1500, 3))
