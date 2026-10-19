import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class PipelineComms:
    def __init__(
        self,
        prev_pp_rank: int,
        pp_group: ProcessGroup,
        next_pp_rank: int,
    ):
        self.prev_pp_rank = prev_pp_rank
        self.next_pp_rank = next_pp_rank
        self.pp_group = pp_group

    def recv_forward(self, microbatch_id: int, stage_input: torch.Tensor) -> torch.Tensor:
        logger.debug("Receiving forward microbatch %s", microbatch_id)
        dist.recv(stage_input, src=self.prev_pp_rank, group=self.pp_group)
        return stage_input

    def send_forward(self, microbatch_id: int, stage_output: torch.Tensor):
        logger.debug("Sending forward microbatch %s", microbatch_id)
        dist.send(stage_output, dst=self.next_pp_rank, group=self.pp_group)

    def recv_backward(self, microbatch_id: int, output_grad: torch.Tensor):
        logger.debug("Receiving backward microbatch %s", microbatch_id)
        dist.recv(output_grad, src=self.next_pp_rank, group=self.pp_group)
        return output_grad

    def send_backward(self, microbatch_id: int, input_grad: torch.Tensor):
        logger.debug("Sending backward microbatch %s", microbatch_id)
        dist.send(input_grad, dst=self.prev_pp_rank, group=self.pp_group)

    def _send_recv(
        self, outgoing_tensor: torch.Tensor, incoming_tensor: torch.Tensor, dest_rank: int
    ) -> torch.Tensor:
        ops = [
            dist.P2POp(dist.isend, outgoing_tensor, dest_rank, self.pp_group),
            dist.P2POp(dist.irecv, incoming_tensor, dest_rank, self.pp_group),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        return incoming_tensor

    def send_forward_recv_backward(
        self, stage_output: torch.Tensor, output_grad_buffer: torch.Tensor
    ):
        return self._send_recv(stage_output, output_grad_buffer, self.next_pp_rank)

    def recv_forward_send_backward(
        self, input_grad: torch.Tensor, stage_input_buffer: torch.Tensor
    ):
        return self._send_recv(input_grad, stage_input_buffer, self.prev_pp_rank)
