import logging
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import record_function

from src.config import AppConfig
from src.metrics import HistogramMetric, ScalarMetric
from src.model.model import NanoTitanModel
from src.parallel.pp_schedules import get_pipeline_schedule
from src.parallel_dims import ParallelDims
from src.utils import clip_gradients, compute_grad_norm

logger = logging.getLogger(__name__)


@dataclass
class MicrobatchState:
    input: torch.Tensor
    output: torch.Tensor
    ce_loss: torch.Tensor | None
    aux_loss: torch.Tensor


class PipelineParallel:
    def __init__(self, cfg: AppConfig, dim: ParallelDims, reducers: dict):
        self.cfg = cfg
        self.dim = dim
        self.device = f"cuda:{dim.local_rank}"
        self.reducers = reducers
        self.run_schedule = get_pipeline_schedule(cfg.runtime.pipeline_schedule)
        self.microbatch_states = {}

    def synchronize_device(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def train_step(self, model: NanoTitanModel, batch, optimizer):
        x, y = batch
        microbatch_x, microbatch_y = self.prepare_microbatch(x, y)
        self.microbatch_size = x.shape[0] // self.cfg.runtime.num_microbatches
        self.microbatch_states.clear()
        self.ce_losses = []
        self.moe_aux_losses = []
        self.moe_route_counts = []

        self.synchronize_device()
        self.step_start_time = time.perf_counter()

        optimizer.zero_grad()
        self.run_schedule(self, model, microbatch_x, microbatch_y)

        for reducer in self.reducers.values():
            reducer.prepare_missing_grad()
        self.finalize_backward()

        grad_norm = compute_grad_norm(model, self.dim)
        clip_gradients(model, grad_norm, self.cfg.trainer)

        with record_function("optimizer_step"):
            optimizer.step()
        self.synchronize_device()
        step_time = time.perf_counter() - self.step_start_time

        ce_losses = [loss for loss in self.ce_losses if loss is not None]

        metrics = {
            "train/ce_loss": ScalarMetric(
                (sum(ce_losses) / self.cfg.runtime.num_microbatches).item()
                if self.dim.is_pp_last_stage
                else 0.0,
                reduce="sum",
            ),
            "train/lb_loss": ScalarMetric(
                (sum(self.moe_aux_losses) / self.cfg.runtime.num_microbatches).item(),
                reduce="sum",
            ),
            "train/grad_norm": ScalarMetric(grad_norm.item(), reduce="none"),
            "time/step_time": ScalarMetric(step_time, reduce="max"),
            "time/forward_time": ScalarMetric(self.forward_time, reduce="max"),
        }
        metrics.update(self._moe_route_fraction_metrics(model, self.moe_route_counts))

        return metrics

    @torch.inference_mode()
    def val_step(self, model: NanoTitanModel, batch) -> tuple[float, int]:
        x, y = batch
        microbatch_x, microbatch_y = self.prepare_microbatch(x, y)
        self.microbatch_size = x.shape[0] // self.cfg.runtime.num_microbatches

        self.synchronize_device()
        loss_sum = 0.0
        token_count = 0

        with record_function("validation_forward_pass"):
            for microbatch_id, (micro_x, micro_y) in enumerate(
                zip(microbatch_x, microbatch_y, strict=True)
            ):
                stage_input = (
                    micro_x.to(self.device)
                    if self.dim.is_pp_first_stage
                    else self.recv_forward(microbatch_id)
                )
                stage_output, _ = model(stage_input)

                if self.dim.is_pp_last_stage:
                    target = micro_y.to(self.device)
                    loss_sum += F.cross_entropy(
                        stage_output.reshape(-1, stage_output.size(-1)),
                        target.reshape(-1),
                        reduction="sum",
                    ).item()
                    token_count += target.numel()

                if not self.dim.is_pp_last_stage:
                    self.send_forward(microbatch_id, stage_output)

        self.synchronize_device()
        return loss_sum, token_count

    def record_forward_complete(self):
        self.synchronize_device()
        self.forward_time = time.perf_counter() - self.step_start_time

    def forward_microbatch(self, microbatch_id, model, stage_input, target=None):
        if not self.dim.is_pp_first_stage:
            stage_input.requires_grad_()

        stage_output, moe_stats = model(stage_input)
        aux_loss = torch.stack([stats.aux_loss for stats in moe_stats]).mean() / self.dim.pp_size
        ce_loss = None
        if self.dim.is_pp_last_stage:
            target = target.to(self.device)
            ce_loss = F.cross_entropy(
                stage_output.reshape(-1, stage_output.size(-1)), target.reshape(-1)
            )

        self.microbatch_states[microbatch_id] = MicrobatchState(
            input=stage_input,
            output=stage_output,
            ce_loss=ce_loss,
            aux_loss=aux_loss,
        )
        self.ce_losses.append(None if ce_loss is None else ce_loss.detach())
        self.moe_aux_losses.append(aux_loss.detach())
        self.moe_route_counts.append([stats.tokens_per_expert.detach() for stats in moe_stats])
        return stage_output

    def backward_microbatch(self, microbatch_id, output_grad=None, sync_gradients=False):
        for reducer in self.reducers.values():
            reducer.backward_grad_sync = sync_gradients

        state = self.microbatch_states.pop(microbatch_id)
        aux_loss = state.aux_loss / self.cfg.runtime.num_microbatches
        if self.dim.is_pp_last_stage:
            torch.autograd.backward([state.ce_loss / self.cfg.runtime.num_microbatches, aux_loss])
        else:
            torch.autograd.backward(
                [state.output, aux_loss], [output_grad, torch.ones_like(aux_loss)]
            )

        return None if self.dim.is_pp_first_stage else state.input.grad

    def _activation_buffer(self) -> torch.Tensor:
        return torch.empty(
            (
                self.microbatch_size,
                self.cfg.model.max_seq_len,
                self.cfg.model.d_model,
            ),
            dtype=self.cfg.model.dtype,
            device=self.device,
        )

    def recv_forward(self, microbatch_id):
        stage_input = self._activation_buffer()
        logger.debug("Receiving forward microbatch %s", microbatch_id)
        dist.recv(stage_input, src=self.dim.prev_pp_rank, group=self.dim.pp_group)
        return stage_input

    def send_forward(self, microbatch_id, stage_output):
        logger.debug("Sending forward microbatch %s", microbatch_id)
        dist.send(stage_output, dst=self.dim.next_pp_rank, group=self.dim.pp_group)

    def recv_backward(self, microbatch_id):
        output_grad = self._activation_buffer()
        logger.debug("Receiving backward microbatch %s", microbatch_id)
        dist.recv(output_grad, src=self.dim.next_pp_rank, group=self.dim.pp_group)
        return output_grad

    def send_backward(self, microbatch_id, input_grad):
        logger.debug("Sending backward microbatch %s", microbatch_id)
        dist.send(input_grad, dst=self.dim.prev_pp_rank, group=self.dim.pp_group)

    def _moe_route_fraction_metrics(
        self, model: NanoTitanModel, moe_route_counts
    ) -> dict[str, ScalarMetric | HistogramMetric]:
        metrics = {}
        local_layer_fracs = {}

        if moe_route_counts and moe_route_counts[0]:
            num_local_layers = len(moe_route_counts[0])
            for local_layer_idx in range(num_local_layers):
                counts = torch.stack(
                    [
                        microbatch_counts[local_layer_idx].float()
                        for microbatch_counts in moe_route_counts
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
                hist_value = frac / self.dim.data_world_size
            metrics[f"moe/layer_{layer_idx:02d}/route_frac_dist"] = HistogramMetric(
                hist_value, reduce="sum"
            )

        return metrics

    def finalize_backward(self):
        self.microbatch_states.clear()
        for reducer in self.reducers.values():
            reducer.finalize_backward()

    def prepare_microbatch(self, x, y) -> None:
        batch_size = x.shape[0]
        num_microbatches = self.cfg.runtime.num_microbatches

        assert batch_size % num_microbatches == 0
        microbatch_x = x.chunk(chunks=num_microbatches)
        microbatch_y = y.chunk(chunks=num_microbatches)

        return microbatch_x, microbatch_y
