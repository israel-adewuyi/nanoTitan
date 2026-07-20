from glob import glob
from pathlib import Path

import pytest

from src.config import load_config


@pytest.mark.parametrize(
    "path", (path for path in glob("configs/*.toml") if not Path(path).name.startswith("sweep_"))
)
def test_checked_in_configs_load(path):
    config = load_config(path)

    assert config.runtime.num_expert == config.model.num_experts
