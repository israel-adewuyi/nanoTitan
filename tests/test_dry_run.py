import logging
import sys

import torch

from src import train
from src.config import AppConfig, load_config


def test_dry_run_counts_full_model_without_allocating_weights_or_starting_training(
    monkeypatch, caplog
):
    raw = load_config("configs/dist.toml").model_dump()
    raw["model"].update(
        vocab_size=16,
        d_model=8,
        d_head=4,
        n_heads=2,
        n_layers=2,
        max_seq_len=8,
        ffn_in=16,
        num_experts=4,
        top_k=2,
        moe_backend="cuda",
    )
    raw["runtime"].update(dp_size=3, pp_size=2, ep_size=2, num_microbatches=2)
    cfg = AppConfig.model_validate(raw)
    original = cfg.model_dump()
    monkeypatch.setattr(train, "load_run_config", lambda path: cfg)
    monkeypatch.setattr(sys, "argv", ["train", "unused.toml", "--dry-run"])
    # Preserve pytest's log capture when main configures logging.
    monkeypatch.setattr(train, "setup_logging", lambda level: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not start training or use CUDA")

    for name in ("init_distributed", "PackedTokenDataset", "setup_optimizer"):
        monkeypatch.setattr(train, name, forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    monkeypatch.setattr(train.NanoTitanModel, "forward", forbidden)

    construct = train.NanoTitanModel.from_specs
    models = []

    def construct_on_meta(model_cfg, spec):
        model = construct(model_cfg, spec)
        assert all(p.is_meta for p in model.parameters())
        assert all(b.is_meta for b in model.buffers())
        models.append(model)
        return model

    monkeypatch.setattr(train.NanoTitanModel, "from_specs", construct_on_meta)
    with caplog.at_level(logging.INFO):
        train.main()

    assert len(models) == 1
    # Two embeddings plus two layers; each layer has 4 SwiGLU experts, 2 active.
    assert "3,936 total, 2,400 active" in caplog.text
    assert "config validation and model construction passed" in caplog.text
    assert "were not checked" in caplog.text
    assert cfg.model_dump() == original
