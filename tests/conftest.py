import pytest
import shutil
from pathlib import Path
from typing import Generator


@pytest.fixture
def assets() -> Path:
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="function")
def tmp_path(tmp_path_factory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("tmp")
    # tmp_dir.mkdir()
    return tmp_dir
