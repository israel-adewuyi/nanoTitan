import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

from src.config import ModelConfig
from src.model.moe_ops import torch_backend_all_to_all
from src.model.utils import ModelShardSpec

logger = logging.getLogger(__name__)


class TorchMoEBackend:
    def __init__(
        self, cfg: ModelConfig, experts: nn.ModuleList, router: nn.Linear, spec: ModelShardSpec
    ):
        self.cfg = cfg
        self.spec = spec
        self.router = router
        self.experts = experts

    def forward(
        self,
        x: torch.Tensor,
    ):
        batch, seq_len, d_model = x.shape
        # flatten residual stream tokens into a 2D tensor of shape (num_tokens, d_model)
        flat_tokens = x.reshape(-1, d_model)

        # get the expert logits
        router_dtype = self.router.weight.dtype
        with record_function("moe/router"):
            expert_logits = self.router(flat_tokens.to(router_dtype))

        with record_function("moe/topK"):
            expert_probs = expert_logits.softmax(dim=-1)
            topk_weights, topk_expert_idx = torch.topk(expert_probs, dim=-1, k=self.cfg.top_k)
        expert_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        assert expert_weights.dtype == torch.float32, "Expert topk weights should be in fp32"

        tokens_per_expert = torch.bincount(
            topk_expert_idx.reshape(-1), minlength=self.cfg.num_experts
        )
        expert_offsets = torch.empty(self.cfg.num_experts + 1, dtype=torch.long, device=x.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)

        total_assignments = flat_tokens.shape[0] * self.cfg.top_k
        packed_X = torch.empty(
            (total_assignments, d_model), dtype=flat_tokens.dtype, device=x.device
        )
        packed_token_ids = torch.empty(total_assignments, dtype=torch.long, device=x.device)
        packed_weights = torch.empty(total_assignments, dtype=expert_weights.dtype, device=x.device)

        for expert in range(self.cfg.num_experts):
            indices = torch.argwhere(topk_expert_idx == expert)
            if indices.numel() == 0:
                continue
            start = expert_offsets[expert].item()
            end = expert_offsets[expert + 1].item()
            packed_X[start:end] = flat_tokens[indices[:, 0]]
            packed_token_ids[start:end] = indices[:, 0]
            packed_weights[start:end] = expert_weights[indices[:, 0], indices[:, 1]]

        if self.spec.ep_size == 1:
            expert_inputs = packed_X
            local_expert_offsets = expert_offsets
            permutation = None
        else:
            logging.info(expert_offsets)

            send_matrix = tokens_per_expert.view(self.spec.ep_size, self.spec.per_rank_expert)
            recv_matrix = torch.empty_like(send_matrix)

            dist.all_to_all_single(
                recv_matrix,
                send_matrix,
                input_split_sizes=[1] * self.spec.ep_size,
                output_split_sizes=[1] * self.spec.ep_size,
                group=self.spec.ep_group,
            )
            logging.info(f"Send matrix is {send_matrix}")
            logging.info(f"Recv matrix is {recv_matrix}")

            send_splits = send_matrix.sum(dim=1)
            recv_splits = recv_matrix.sum(dim=1)
            logging.info(f"Send count: {send_splits}")
            logging.info(f"Receiver count: {recv_splits}")

            receiver_X = torch_backend_all_to_all(
                packed_X,
                input_splits=send_splits.tolist(),
                output_splits=recv_splits.tolist(),
                group=self.spec.ep_group,
            )

            logging.info(f"Shape of received X is : {receiver_X.shape}")

            src_offsets = torch.zeros(
                (self.spec.ep_size, self.spec.per_rank_expert),
                dtype=torch.int32,
                device=receiver_X.device,
            )
            permutation = torch.zeros(
                (receiver_X.shape[0]), dtype=torch.int32, device=receiver_X.device
            )

            prev = 0
            for src in range(self.spec.ep_size):
                for expert in range(self.spec.per_rank_expert):
                    src_offsets[src][expert] = prev
                    prev += recv_matrix[src][expert]

            cur_idx = 0
            for expert in range(self.spec.per_rank_expert):
                for src in range(self.spec.ep_size):
                    count = recv_matrix[src][expert]
                    src_start = src_offsets[src][expert]
                    for idx in range(count):
                        permutation[cur_idx + idx] = src_start + idx
                    cur_idx += count

            local_expert_offsets = torch.empty(
                self.spec.per_rank_expert + 1, dtype=torch.long, device=receiver_X.device
            )
            local_expert_offsets[0] = 0
            local_expert_offsets[1:] = torch.cumsum(recv_matrix.sum(dim=0), dim=0)

            expert_inputs = receiver_X[permutation]

        packed_outputs = self.experts(expert_inputs, local_expert_offsets)

        if self.spec.ep_size == 1:
            returned_outputs = packed_outputs
        else:
            source_X = torch.empty_like(packed_outputs)
            source_X[permutation.long()] = packed_outputs

            returned_outputs = torch_backend_all_to_all(
                source_X,
                input_splits=recv_splits.tolist(),
                output_splits=send_splits.tolist(),
                group=self.spec.ep_group,
            )

        pool = torch.zeros_like(flat_tokens)
        weighted_outputs = (returned_outputs * packed_weights.unsqueeze(1)).to(pool.dtype)
        pool.index_add_(0, packed_token_ids, weighted_outputs)

        return (pool.reshape(batch, seq_len, d_model), tokens_per_expert)
