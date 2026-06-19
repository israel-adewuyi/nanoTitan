from src.config import RuntimeConfig


def get_layer_bounds(cfg: RuntimeConfig, pp_rank: int):
    """
    Each PP stage maps to a subset of layers.
    We compute the layer bounds for each PP stage.
    """
    per_rank_layers = cfg.model.n_layers // cfg.runtime.pp_size
    start_idx = pp_rank * per_rank_layers
    end_idx = (pp_rank + 1) * per_rank_layers
    return (start_idx, end_idx)
