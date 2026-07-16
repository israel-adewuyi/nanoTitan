from functools import lru_cache
from types import ModuleType


@lru_cache
def get_cuda_extension() -> ModuleType:
    try:
        import nanotitan_cuda
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The nanoTitan CUDA extension is not installed. "
            "Build it before selecting moe_backend='cuda'."
        ) from exc
    return nanotitan_cuda
