import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def lena_path(assets: Path) -> Path:
    return assets / "lena.jpg"


@pytest.fixture
def img_sample() -> np.ndarray:
    return np.random.randint(0, 255, size=(1000, 1500, 3))
