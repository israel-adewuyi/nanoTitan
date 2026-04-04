import os

import torch
import torch.distributed as dist


def init_distributed():
    if is_distributed() or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_distributed():
        return dist.barrier()


def cleanup():
    if is_distributed():
        dist.destroy_process_group()
