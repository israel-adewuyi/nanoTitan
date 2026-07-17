from pathlib import Path

from scripts.sweep import build_runs, dump_toml
from src.config import AppConfig


def test_checked_in_sweep_builds_valid_named_runs():
    _, runs = build_runs(Path("configs/sweep_hparams.toml"))

    assert len(runs) == 18
    assert runs[0][0].endswith("optim-lr-0.01__model-dtype-bfloat16")
    assert len({name for name, _ in runs}) == len(runs)

    for name, config in runs:
        round_trip = AppConfig.model_validate(config)
        assert round_trip.run_name == name


def test_generated_toml_round_trips(tmp_path):
    _, runs = build_runs(Path("configs/sweep_hparams.toml"))
    output = tmp_path / "generated.toml"
    output.write_text(dump_toml(runs[0][1]), encoding="utf-8")

    from src.config import load_config

    assert load_config(output).run_name == runs[0][0]
