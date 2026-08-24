from src.parallel.pp_schedules.gpipe import run_gpipe
from src.parallel.pp_schedules.one_f_one_b import run_1F1B


def get_pipeline_schedule(name: str):
    try:
        return {"gpipe": run_gpipe, "1f1b": run_1F1B}[name]
    except KeyError as exc:
        raise ValueError(f"Pipeline schedule '{name}' is not implemented") from exc
