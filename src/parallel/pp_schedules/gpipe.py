from torch.profiler import record_function


def run_gpipe(pipeline, model, microbatch_x, microbatch_y):
    with record_function("forward_pass"):
        for microbatch_id, (x, y) in enumerate(zip(microbatch_x, microbatch_y, strict=False)):
            stage_input = (
                x.to(pipeline.device)
                if pipeline.dim.is_pp_first_stage
                else pipeline.recv_forward(microbatch_id)
            )
            stage_output = pipeline.forward_microbatch(microbatch_id, model, stage_input, y)
            if not pipeline.dim.is_pp_last_stage:
                pipeline.send_forward(microbatch_id, stage_output)

    pipeline.record_forward_completion()

    with record_function("backward_pass"):
        for microbatch_id in reversed(range(len(microbatch_x))):
            output_grad = (
                None if pipeline.dim.is_pp_last_stage else pipeline.recv_backward(microbatch_id)
            )
            input_grad = pipeline.backward_microbatch(
                microbatch_id,
                output_grad,
                sync_gradients=microbatch_id == 0,
            )
            if not pipeline.dim.is_pp_first_stage:
                pipeline.send_backward(microbatch_id, input_grad)
