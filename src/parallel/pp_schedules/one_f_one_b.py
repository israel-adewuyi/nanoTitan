from torch.profiler import record_function


def run_1F1B(pipeline, model, microbatch_x, microbatch_y):
    num_microbatches = len(microbatch_x)
    pp_group_size = pipeline.dim.pp_size
    pp_rank = pipeline.dim.pp_rank
    warmup_steps = min(num_microbatches, (pp_group_size - pp_rank))
    fwd_idx, bwd_idx = 0, 0

    with record_function("1f1b/warmup"):
        while fwd_idx < warmup_steps:
            stage_input = (
                microbatch_x[fwd_idx].to(pipeline.device)
                if pipeline.dim.is_pp_first_stage
                else pipeline.recv_forward(fwd_idx)
            )
            stage_output = pipeline.forward_microbatch(
                fwd_idx, model, stage_input, microbatch_y[fwd_idx]
            )
            if not pipeline.dim.is_pp_last_stage and fwd_idx < warmup_steps - 1:
                pipeline.send_forward(fwd_idx, stage_output)
            fwd_idx += 1

    if fwd_idx == num_microbatches:
        pipeline.record_forward_completion()

    with record_function("1f1b/steady"):
        while True:
            output_grad = (
                None
                if pipeline.dim.is_pp_last_stage
                else pipeline.send_forward_recv_backward(stage_output)
            )
            input_grad = pipeline.backward_microbatch(
                bwd_idx,
                output_grad,
                sync_gradients=bwd_idx == num_microbatches - 1,
            )
            bwd_idx += 1

            if fwd_idx == num_microbatches:
                if not pipeline.dim.is_pp_first_stage:
                    pipeline.send_backward(bwd_idx - 1, input_grad)
                break
            stage_input = (
                microbatch_x[fwd_idx].to(pipeline.device)
                if pipeline.dim.is_pp_first_stage
                else pipeline.recv_forward_send_backward(input_grad)
            )
            stage_output = pipeline.forward_microbatch(
                fwd_idx, model, stage_input, microbatch_y[fwd_idx]
            )
            fwd_idx += 1
            if fwd_idx == num_microbatches:
                pipeline.record_forward_completion()

    with record_function("1f1b/cooldown"):
        while bwd_idx < num_microbatches:
            output_grad = None if pipeline.dim.is_pp_last_stage else pipeline.recv_backward(bwd_idx)
            input_grad = pipeline.backward_microbatch(
                bwd_idx,
                output_grad,
                sync_gradients=bwd_idx == num_microbatches - 1,
            )
            if not pipeline.dim.is_pp_first_stage:
                pipeline.send_backward(bwd_idx, input_grad)
            bwd_idx += 1
