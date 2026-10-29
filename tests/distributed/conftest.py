import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="session")
def two_ranks():
    """Keep WORLD alive across the DP, PP, and EP tests in one torchrun launch."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch with torchrun --standalone --nproc-per-node=2")

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=60))

    try:
        assert dist.get_world_size() == 2
        device = torch.device("cpu")
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        yield dist.get_rank(), device
    finally:
        # Reinitializing WORLD between modules needs out-of-band synchronization.
        # Tear it down only after every distributed test has finished instead.
        if owns_process_group:
            dist.destroy_process_group()
