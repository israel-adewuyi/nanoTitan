import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.config import AppConfig
from src.runtime.base import ScalarMetric
from src.model.model import NanoTitanModel
from src.parallel_dims import ParallelDims
from src.runtime.reducer import ReducerV1

logger = logging.getLogger(__name__)


class PipelineParallel:
    def __init__(self, cfg: AppConfig, dim: ParallelDims, reducer: ReducerV1):
        self.cfg = cfg
        self.dim = dim
        self.device = f"cuda:{dim.local_rank}"
        self.reducer = reducer
        self.stage_inputs = []
        self.stage_outputs = []

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
                dist.recv(mb_x, src=self.dim.prev_pp_rank, group=self.dim.pp_group)
                mb_x.requires_grad_()
                self.stage_inputs.append(mb_x)

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
                self.stage_outputs.append(mb_x)

                dist.send(mb_x, dst=self.dim.next_pp_rank, group=self.dim.pp_group)
                losses.append(None)

        # Backward pass
        for microbatch_idx in reversed(range(self.cfg.runtime.num_microbatches)):
            self.reducer.backward_grad_sync = microbatch_idx == 0
            loss = (
                losses[microbatch_idx] / self.cfg.runtime.num_microbatches
                if self.dim.is_pp_last_stage
                else None
            )
            self.backward(loss, model, microbatch_idx - 1)

        self.reducer.prepare_missing_grad()
        self.finalize_backward()

        metrics = {
            "train/loss": ScalarMetric(
                sum(losses).item() if self.dim.is_pp_last_stage else 0.0,
                reduce="sum",
            ),
        }

        return metrics

    def backward(self, loss, model, microbatch_idx):
        if self.dim.is_pp_last_stage:
            loss.backward()
        else:
            out_acts = self.stage_outputs[microbatch_idx]
            out_acts_grad = torch.empty(
                (
                    self.microbatch_size,
                    self.cfg.model.max_seq_len,
                    self.cfg.model.d_model,
                ),
                device=self.device,
            )
            dist.recv(out_acts_grad, src=self.dim.next_pp_rank, group=self.dim.pp_group)
            out_acts.backward(out_acts_grad)

        # incoming.backward()
        if not self.dim.is_pp_first_stage:
            dist.send(
                self.stage_inputs[microbatch_idx].grad,
                dst=self.dim.prev_pp_rank,
                group=self.dim.pp_group,
            )

    def finalize_backward(self):
        self.stage_inputs = []
        self.stage_outputs = []
        self.reducer.finalize_backward()

    def prepare_microbatch(self, x, y) -> None:
        batch_size = x.shape[0]
        num_microbatches = self.cfg.runtime.num_microbatches

        assert batch_size % num_microbatches == 0
        microbatch_x = x.chunk(chunks=num_microbatches)
        microbatch_y = y.chunk(chunks=num_microbatches)

        return microbatch_x, microbatch_y
