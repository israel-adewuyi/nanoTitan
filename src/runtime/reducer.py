import torch
import torch.distributed as dist

from src.model import NanoTitanModel


class ReducerV0:
    """Primitive reducer class for DDP implementation"""

    def __init__(self, model: NanoTitanModel, world_size: int):
        self.world_size = world_size
        self.params = [p for p in model.parameters() if p.requires_grad()]
        self.hook_handles = []

        for p in self.params:
            h = p.register_hook(self.reduce_grad)
            self.hook_handles.append(h)

    def reduce_grad(self, grad) -> torch.Tensor:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad = grad / self.world_size
        return grad
