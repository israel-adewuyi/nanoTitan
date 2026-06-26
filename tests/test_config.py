from glob import glob

import pytest

from src.config import load_config


@pytest.mark.parametrize("path", glob("configs/*.toml"))
def test_checked_in_configs_load(path):
    load_config(path)
