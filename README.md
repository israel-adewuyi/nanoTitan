# Benchmark Results

## Model Specs

| Spec             |       Value |
| ---------------- | ----------: |
| Total parameters |         92M |
| Layers           |          16 |
| Hidden size      |         512 |
| Attention heads  |          16 |
| Head dimension   |          32 |
| FFN hidden size  |       2,048 |
| Vocabulary size  |      50,257 |
| Sequence length  |         768 |
| Optimizer        |        Adam |
| Dataset          | TinyStories |
| Seed             |          42 |
| Learning rate    |        5e-4 |

## Benchmark Table

To compute the throughput and timing metrics, we use first 10% of steps as warmup and average over the next 80% of values (Step 350 - 3149 for DDP, Steps 600 - 5399 for Single GPU runs).

| Mode                    | GPU | Total batch size | tokens/sec | step time | Backward time | Number of tokens | Final loss |
| ----------------------- | --: | ---------------: | ---------: | --------: | ------------: | ---------------: | ---------: |
| Bucketed AllReduce DDP  |   2 |               80 |  31,928.99 |    1.9243 |        1.2623 |      215M tokens |     1.7621 |
| Per-param AllReduce DDP |   2 |               80 |  31,448.13 |    1.9537 |        1.2920 |      215M tokens |     1.7621 |
| Reference DDP (Pytorch) |   2 |               80 |  31,888.37 |    1.9267 |        1.2625 |      215M tokens |     1.7621 |
| Single GPU              |   1 |               40 |  16,045.61 |    1.9145 |        1.2500 |      184M tokens |     1.7392 |

## Plots

Loss plot
![Loss plot](assets/ddp/loss.png)
Grad Norm plot
![!Grad Norm plot](assets/ddp/grad_norm.png)

# ToDO

- [x] DDP
- [x] Benchmark DDP
- [x] Pipeline Parallelism (PP)
- [ ] Benchmark PP
- [ ] Tensor Parallelism (TP)
- [ ] Benchmark TP
- [ ] FSDP
- [ ] Bechmark FSDP
