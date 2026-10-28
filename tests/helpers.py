from types import SimpleNamespace
from unittest.mock import patch

from src.config import ModelConfig, RuntimeConfig
from src.parallel_dims import ParallelContext


def make_single_rank_context(model_config: ModelConfig) -> ParallelContext:
    """Build real ownership metadata for local models without creating process groups."""
    config = SimpleNamespace(model=model_config, runtime=RuntimeConfig())
    with (
        patch("src.parallel_dims.get_world_size", return_value=1),
        patch("src.parallel_dims.get_rank", return_value=0),
        patch("src.parallel_dims.get_local_rank", return_value=0),
        patch("src.parallel_dims.dist.new_group", return_value=None),
    ):
        return ParallelContext(config)
