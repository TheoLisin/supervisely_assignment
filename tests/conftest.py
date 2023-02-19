import pytest
from pathlib import Path


@pytest.fixture
def assets() -> Path:
    """Assets path"""
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="function")
def tmp_path(tmp_path_factory) -> Path:
    """Temporary path for any data"""
    tmp_dir = tmp_path_factory.mktemp("tmp")
    return tmp_dir


@pytest.fixture
def lena_path(assets: Path) -> Path:
    return assets / "lena.jpg"


@pytest.fixture
def gslena_path(assets: Path) -> Path:
    return assets / "gslena.jpg"
