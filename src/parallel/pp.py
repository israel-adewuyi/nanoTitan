import logging
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint

from src.config import AppConfig
from src.metrics import HistogramMetric, ScalarMetric
from src.model.model import NanoTitanModel
from src.model.utils import max_violation
from src.parallel.pp_schedules import get_pipeline_schedule
from src.parallel_dims import ParallelDims
from src.utils import clip_gradients, compute_grad_norm

logger = logging.getLogger(__name__)


@dataclass
class MicrobatchState:
    input: torch.Tensor
    output: torch.Tensor
    ce_loss: torch.Tensor | None


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
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self.run_schedule(self, model, microbatch_x, microbatch_y)

        for reducer in self.reducers.values():
            reducer.prepare_missing_grad()
        self.finalize_backward()

        peak_allocated_gib = (
            torch.cuda.max_memory_allocated(self.device) / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )

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
            "train/grad_norm": ScalarMetric(grad_norm.item(), reduce="none"),
            "time/step_time": ScalarMetric(step_time, reduce="max"),
            "time/forward_completion_time": ScalarMetric(
                self.forward_completion_time, reduce="max"
            ),
            "memory/pp_first_peak_allocated_gib": ScalarMetric(
                peak_allocated_gib if self.dim.is_pp_first_stage else 0.0,
                reduce="max",
            ),
            "memory/pp_middle_peak_allocated_gib": ScalarMetric(
                peak_allocated_gib
                if not self.dim.is_pp_first_stage and not self.dim.is_pp_last_stage
                else 0.0,
                reduce="max",
            ),
            "memory/pp_last_peak_allocated_gib": ScalarMetric(
                peak_allocated_gib if self.dim.is_pp_last_stage else 0.0,
                reduce="max",
            ),
        }
        metrics.update(self._moe_metrics(model, self.moe_route_counts))

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

    def record_forward_completion(self):
        self.synchronize_device()
        self.forward_completion_time = time.perf_counter() - self.step_start_time

    def forward_microbatch(self, microbatch_id, model, stage_input, target=None):
        if not self.dim.is_pp_first_stage:
            stage_input.requires_grad_()

        if self.cfg.runtime.activation_checkpointing:
            stage_output, moe_stats = checkpoint(
                model, stage_input, use_reentrant=False, preserve_rng_state=True
            )
        else:
            stage_output, moe_stats = model(stage_input)

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
        )
        self.ce_losses.append(None if ce_loss is None else ce_loss.detach())
        self.moe_route_counts.append([stats.tokens_per_expert.detach() for stats in moe_stats])
        return stage_output

    def backward_microbatch(self, microbatch_id, output_grad=None, sync_gradients=False):
        for reducer in self.reducers.values():
            reducer.backward_grad_sync = sync_gradients

        state = self.microbatch_states.pop(microbatch_id)
        if self.dim.is_pp_last_stage:
            torch.autograd.backward([state.ce_loss / self.cfg.runtime.num_microbatches])
        else:
            torch.autograd.backward([state.output], [output_grad])

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

    def send_forward_recv_backward(self, stage_output):
        output_grad = self._activation_buffer()
        ops = [
            dist.P2POp(dist.isend, stage_output, self.dim.next_pp_rank, self.dim.pp_group),
            dist.P2POp(dist.irecv, output_grad, self.dim.next_pp_rank, self.dim.pp_group),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        return output_grad

    def recv_forward_send_backward(self, input_grad):
        stage_input = self._activation_buffer()
        ops = [
            dist.P2POp(dist.isend, input_grad, self.dim.prev_pp_rank, self.dim.pp_group),
            dist.P2POp(dist.irecv, stage_input, self.dim.prev_pp_rank, self.dim.pp_group),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        return stage_input

    def _moe_metrics(
        self, model: NanoTitanModel, moe_route_counts
    ) -> dict[str, ScalarMetric | HistogramMetric]:
        metrics = {}
        local_layer_fracs = {}
        max_vio_sum = 0.0

        if moe_route_counts and moe_route_counts[0]:
            # Raw top-k counts, before capacity rerouting: [local layers, experts].
            layer_counts = torch.stack(
                [torch.stack(microbatch_counts) for microbatch_counts in moe_route_counts]
            ).sum(dim=0)
            if self.dim.data_world_size > 1:
                dist.all_reduce(layer_counts, op=dist.ReduceOp.SUM, group=self.dim.shared_dp_group)

            # Max is nonlinear: pool the training batch before measuring each layer.
            max_vio_sum = max_violation(layer_counts).sum().item()
            for local_layer_idx, counts in enumerate(layer_counts.float()):
                frac = counts / counts.sum().clamp_min(1.0)
                global_layer_idx = model.spec.layer_start + local_layer_idx
                local_layer_fracs[global_layer_idx] = frac.detach().cpu()

        # World SUM combines PP stages; DP/EP ranks hold duplicate pooled counts.
        metrics["moe/max_vio"] = ScalarMetric(
            max_vio_sum / (self.cfg.model.n_layers * self.dim.data_world_size),
            reduce="sum",
        )

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
