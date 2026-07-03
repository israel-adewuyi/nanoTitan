import logging
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import record_function

from src.config import AppConfig
from src.metrics import HistogramMetric, ScalarMetric
from src.model.model import NanoTitanModel
from src.parallel.reducer import ReducerV1
from src.parallel_dims import ParallelDims

logger = logging.getLogger(__name__)


class PipelineParallel:
    def __init__(self, cfg: AppConfig, dim: ParallelDims, reducer: ReducerV1):
        self.cfg = cfg
        self.dim = dim
        self.device = f"cuda:{dim.local_rank}"
        self.reducer = reducer
        self.stage_inputs = []
        self.stage_outputs = []

    def synchronize_device(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def train_step(self, model: NanoTitanModel, batch, optimizer):
        x, y = batch
        microbatch_x, microbatch_y = self.prepare_microbatch(x, y)
        self.microbatch_size = x.shape[0] // self.cfg.runtime.num_microbatches

        self.synchronize_device()
        step_start_time = time.perf_counter()
        # self._reset_peak_memory_stats()

        optimizer.zero_grad()
        ce_losses, moe_aux_losses, moe_stats_list = [], [], []
        with record_function("forward_pass"):
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
                        dtype=self.cfg.model.dtype,
                    )
                    logger.debug(
                        f"At rank {self.dim.local_rank}!!! Receiving activations from rank {self.dim.prev_pp_rank}"
                    )
                    dist.recv(mb_x, src=self.dim.prev_pp_rank, group=self.dim.pp_group)
                    logger.debug(
                        f"At rank {self.dim.local_rank}!!! Activations received from rank {self.dim.prev_pp_rank}"
                    )
                    mb_x.requires_grad_()
                    self.stage_inputs.append(mb_x)

                mb_x, moe_stats = model(mb_x)

                moe_aux_loss = (
                    torch.stack([s.aux_loss for s in moe_stats]).mean() / self.dim.pp_size
                )
                moe_aux_losses.append(moe_aux_loss)
                moe_stats_list.append(moe_stats)

                if self.dim.is_pp_last_stage:
                    # compute loss here
                    mb_y = mb_y.to(self.device)
                    loss = F.cross_entropy(mb_x.reshape(-1, mb_x.size(-1)), mb_y.reshape(-1))
                    ce_losses.append(loss)
                else:
                    self.stage_outputs.append(mb_x)

                    dist.send(mb_x, dst=self.dim.next_pp_rank, group=self.dim.pp_group)
                    ce_losses.append(None)

        self.synchronize_device()
        forward_time = time.perf_counter() - step_start_time

        with record_function("backward_pass"):
            # Backward pass
            logger.debug(f"[RANK {self.dim.local_rank}] Beginning bwd pass")
            for microbatch_idx in reversed(range(self.cfg.runtime.num_microbatches)):
                self.reducer.backward_grad_sync = microbatch_idx == 0
                ce_loss = (
                    ce_losses[microbatch_idx] / self.cfg.runtime.num_microbatches
                    if self.dim.is_pp_last_stage
                    else None
                )
                moe_aux_loss = moe_aux_losses[microbatch_idx] / self.cfg.runtime.num_microbatches
                self.backward(ce_loss, moe_aux_loss, model, microbatch_idx)

            self.reducer.prepare_missing_grad()
            self.finalize_backward()

        with record_function("optimizer_step"):
            optimizer.step()
        self.synchronize_device()
        step_time = time.perf_counter() - step_start_time

        metrics = {
            "train/ce_loss": ScalarMetric(
                (sum(ce_losses) / self.cfg.runtime.num_microbatches).item()
                if self.dim.is_pp_last_stage
                else 0.0,
                reduce="sum",
            ),
            "train/lb_loss": ScalarMetric(
                (sum(moe_aux_losses) / self.cfg.runtime.num_microbatches).item()
                if self.dim.is_pp_last_stage
                else 0.0,
                reduce="sum",
            ),
            "train/total_loss": ScalarMetric(
                (sum(ce_losses) + sum(moe_aux_losses) / self.cfg.runtime.num_microbatches).item()
                if self.dim.is_pp_last_stage
                else 0.0,
                reduce="sum",
            ),
            "time/step_time": ScalarMetric(step_time, reduce="max"),
            "time/forward_time": ScalarMetric(forward_time, reduce="max"),
        }
        metrics.update(self._moe_route_fraction_metrics(model, moe_stats_list))

        return metrics

    def _moe_route_fraction_metrics(
        self, model: NanoTitanModel, moe_stats_list
    ) -> dict[str, ScalarMetric | HistogramMetric]:
        metrics = {}
        local_layer_fracs = {}

        if moe_stats_list and moe_stats_list[0]:
            num_local_layers = len(moe_stats_list[0])
            for local_layer_idx in range(num_local_layers):
                counts = torch.stack(
                    [
                        mb_stats[local_layer_idx].tokens_per_expert.detach().float()
                        for mb_stats in moe_stats_list
                    ],
                    dim=0,
                ).sum(dim=0)
                frac = counts / counts.sum().clamp_min(1.0)
                global_layer_idx = model.spec.layer_start + local_layer_idx
                local_layer_fracs[global_layer_idx] = frac.detach().cpu()

        for layer_idx in range(self.cfg.model.n_layers):
            frac = local_layer_fracs.get(layer_idx)
            hist_value = torch.zeros(self.cfg.model.num_experts, dtype=torch.float32)
            if frac is not None:
                hist_value = frac / self.dim.dp_size
            metrics[f"moe/layer_{layer_idx:02d}/route_frac_dist"] = HistogramMetric(
                hist_value, reduce="sum"
            )
            for expert_idx in range(self.cfg.model.num_experts):
                value = 0.0 if frac is None else frac[expert_idx].item() / self.dim.dp_size
                metrics[f"moe/layer_{layer_idx:02d}/expert_{expert_idx:02d}/route_frac"] = (
                    ScalarMetric(value, reduce="sum")
                )

        return metrics

    def backward(self, ce_loss, moe_aux_loss, model, microbatch_idx):
        if self.dim.is_pp_last_stage:
            torch.autograd.backward(
                [ce_loss, moe_aux_loss],
            )
        else:
            out_acts = self.stage_outputs[microbatch_idx]
            out_acts_grad = torch.empty(
                (
                    self.microbatch_size,
                    self.cfg.model.max_seq_len,
                    self.cfg.model.d_model,
                ),
                dtype=self.cfg.model.dtype,
                device=self.device,
            )
            dist.recv(out_acts_grad, src=self.dim.next_pp_rank, group=self.dim.pp_group)
            torch.autograd.backward(
                [out_acts, moe_aux_loss], [out_acts_grad, torch.ones_like(moe_aux_loss)]
            )

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
