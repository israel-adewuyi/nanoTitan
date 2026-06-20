import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.config import AppConfig
from src.model.model import NanoTitanModel
from src.parallel_dims import ParallelDims

logger = logging.getLogger(__name__)


class PipelineParallel:
    def __init__(self, cfg: AppConfig, dim: ParallelDims):
        self.cfg = cfg
        self.dim = dim
        self.device = f"cuda:{dim.local_rank}"

    def train_step(self, model: NanoTitanModel, batch):
        x, y = batch
        microbatch_x, microbatch_y = self.prepare_microbatch(x, y)
        self.microbatch_size = x.shape[0] // self.cfg.runtime.num_microbatches
        # self._reset_peak_memory_stats()
        losses, backward_time = [], None

        for mb_x, mb_y in zip(microbatch_x, microbatch_y, strict=False):
            if self.dim.is_pp_first_stage:
                mb_x = mb_x.to(self.device)
            else:
                mb_x = torch.empty(
                    (
                        self.microbatch_size,
                        self.cfg.model.max_seq_len,
                        self.cfg.model.d_model,
                    ),
                    device=self.device,
                )
                logger.debug(
                    f"At rank {self.dim.local_rank}!!! Receiving activations from rank {self.dim.prev_pp_rank}"
                )
                dist.recv(mb_x, self.dim.prev_pp_rank)

            mb_x = model(mb_x)

            if self.dim.is_pp_last_stage:
                # compute loss here
                mb_y = mb_y.to(self.device)
                loss = F.cross_entropy(mb_x.reshape(-1, mb_x.size(-1)), mb_y.reshape(-1))
                losses.append(loss)
            else:
                logger.debug(
                    f"Sending activations from rank {self.dim.local_rank} to rank {self.dim.next_pp_rank}"
                )
                dist.send(mb_x, self.dim.next_pp_rank)
                losses.append(None)

    def prepare_microbatch(self, x, y) -> None:
        batch_size = x.shape[0]
        num_microbatches = self.cfg.runtime.num_microbatches

        assert batch_size % num_microbatches == 0
        microbatch_x = x.chunk(chunks=num_microbatches)
        microbatch_y = y.chunk(chunks=num_microbatches)

        return microbatch_x, microbatch_y
