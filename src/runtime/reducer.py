import torch
import torch.distributed as dist

from src.model import NanoTitanModel


class ReducerV0:
    """Primitive reducer class for DDP implementation"""

    def __init__(self, model: NanoTitanModel, world_size: int):
        self.world_size = world_size
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.hook_handles = []

        for p in self.params:
            h = p.register_hook(self.reduce_grad)
            self.hook_handles.append(h)

    def reduce_grad(self, grad) -> torch.Tensor:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad = grad / self.world_size
        return grad

    def finalize_backward(self) -> None:
        pass


class ReducerV1:
    """
    Bucketed DDP
    Somethings I am ignoring for now, but will revisit later
    1. dtype. What is the default now? Upside / downside to using any other??
    """

    def __init__(self, model: NanoTitanModel, world_size: int, bucket_size: int):
        self.world_size = world_size
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.hook_handles = []
        self.bucket_size = bucket_size * 1024 * 1024

        self.initialize_buckets()

        for p in self.params:
            h = p.register_post_accumulate_grad_hook(self.reduce_grad)
            self.hook_handles.append(h)

    def initialize_buckets(self) -> None:
        """
        This method allocates parameters to buckets in reverse order of model.parameters()
        Assumptions.
        1. The max parameter fits into a bucket.
        2. All parameters are used.
        3. The same dtype and number of bytes per param element for all parameters
        """

        self.buckets = []
        # Map parameters to bucket via id
        bucket_id = 0
        self.param_to_bucket = {}

        # Map parameters to segments of the bucket's buffer. This makes my life easier after AllReduce
        buffer_len = 0
        self.param_to_offset = {}

        # first bucket
        bucket = {"size": 0, "params": [], "work": None}

        for param in reversed(self.params):
            # get the number of bytes the param occupies in memory
            param_bytes = param.numel() * param.element_size()

            # if param fit's into the existing bucket, put it there, track bucket id and segment length
            if param_bytes + bucket["size"] <= self.bucket_size:
                bucket["params"].append(param)
                bucket["size"] += param_bytes
                self.param_to_bucket[param] = bucket_id
                self.param_to_offset[param] = (buffer_len, buffer_len + param.numel())
                buffer_len += param.numel()
            else:
                # else save the current bucket along with it's initialized buffer and ready count
                total_numel = bucket["size"] // param.element_size()
                bucket["buffer"] = torch.empty(total_numel, device=param.device)
                bucket["ready_count"] = 0
                self.buckets.append(bucket)
                bucket_id += 1
                # create new bucket.
                bucket = {"size": param_bytes, "params": [param], "work": None}
                # track my trackables
                buffer_len = 0
                self.param_to_bucket[param] = bucket_id
                self.param_to_offset[param] = (buffer_len, buffer_len + param.numel())
                buffer_len += param.numel()

        # save the last bucket
        total_numel = bucket["size"] // self.params[-1].element_size()
        bucket["buffer"] = torch.empty(total_numel, device=self.params[-1].device)
        bucket["ready_count"] = 0
        self.buckets.append(bucket)

    def reduce_grad(self, param) -> None:
        temp_bucket = self.buckets[self.param_to_bucket[param]]
        grad = param.grad
        start, end = self.param_to_offset[param]

        temp_bucket["buffer"][start:end].copy_(grad.flatten())
        temp_bucket["ready_count"] += 1

        if temp_bucket["ready_count"] == len(temp_bucket["params"]):
            work = dist.all_reduce(temp_bucket["buffer"], op=dist.ReduceOp.SUM, async_op=True)
            temp_bucket["work"] = work

    def finalize_backward(self):
        for bucket in self.buckets:
            bucket["work"].wait()
            bucket["buffer"].div_(self.world_size)

            for param in bucket["params"]:
                start, end = self.param_to_offset[param]
                param.grad.copy_(bucket["buffer"][start:end].view_as(param))

            bucket["ready_count"] = 0
            bucket["work"] = None
