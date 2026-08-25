from glob import glob
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import DataConfig, RuntimeConfig, load_config


@pytest.mark.parametrize(
    "path", (path for path in glob("configs/*.toml") if not Path(path).name.startswith("sweep_"))
)
def test_checked_in_configs_load(path):
    config = load_config(path)

    assert config.runtime.num_expert == config.model.num_experts


def test_activation_checkpointing_defaults_to_disabled():
    config = RuntimeConfig()

    assert config.activation_checkpointing is False


def test_data_config_accepts_optional_local_dataset_path(tmp_path):
    path = tmp_path / "cached.pt"

    config = DataConfig(dataset_name="source", dataset_path=path)

    assert config.dataset_path == path


def test_single_stage_pipeline_requires_one_microbatch():
    with pytest.raises(
        ValidationError,
        match="runtime.num_microbatches must be 1 when runtime.pp_size is 1",
    ):
        RuntimeConfig(pp_size=1, num_microbatches=2)


def test_single_stage_pipeline_disables_activation_checkpointing():
    with pytest.raises(
        ValidationError,
        match="runtime.activation_checkpointing must be false when runtime.pp_size is 1",
    ):
        RuntimeConfig(pp_size=1, activation_checkpointing=True)


def test_multi_stage_pipeline_allows_multiple_microbatches():
    config = RuntimeConfig(pp_size=2, num_microbatches=4, activation_checkpointing=True)

    assert config.num_microbatches == 4
    assert config.activation_checkpointing is True
