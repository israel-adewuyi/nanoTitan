from src.parallel.pp_schedules.gpipe import run_gpipe


def get_pipeline_schedule(name: str):
    try:
        return {"gpipe": run_gpipe}[name]
    except KeyError as exc:
        raise ValueError(f"Pipeline schedule '{name}' is not implemented") from exc
