import os
import statistics
import time
from datetime import timedelta
import torch
import torch.distributed as dist

import deep_ep


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60),
                           device_id=torch.device("cuda", local_rank))
    group = dist.new_group(list(range(dist.get_world_size())))
    rank = dist.get_rank()

    tokens = int(os.getenv("TOKENS", 4096))
    hidden = int(os.getenv("HIDDEN", 2048))
    experts = int(os.getenv("EXPERTS", 64))
    topk = int(os.getenv("TOPK", 2))
    buffer = None
    try:
        buffer = deep_ep.Buffer(group, int(2e9), explicitly_destroy=True)
        x = torch.full((tokens, hidden), rank, dtype=torch.bfloat16, device="cuda")
        scores = torch.rand((tokens, experts), device="cuda")
        topk_idx = scores.topk(topk, dim=-1, sorted=False).indices
        num_rank, _, num_expert, is_in_rank, _ = buffer.get_dispatch_layout(topk_idx, experts)
        config = deep_ep.Config(24, 8, 256)

        dispatch = lambda: buffer.dispatch(
            x=x, num_tokens_per_rank=num_rank, is_token_in_rank=is_in_rank,
            num_tokens_per_expert=num_expert, config=config)
        for _ in range(10):
            dispatch()
        torch.cuda.synchronize()
        dist.barrier()

        times = []
        wall_times = []
        for _ in range(30):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
            dispatch()
            wall_times.append((time.perf_counter() - t0) * 1e6)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3)

        # All ranks finish before rank 0 prints or tears down the communicator.
        dist.barrier()
        if rank == 0:
            print(f"cuda_us median={statistics.median(times):.2f} p10={sorted(times)[3]:.2f} p90={sorted(times)[26]:.2f}")
            print(f"wall_us median={statistics.median(wall_times):.2f} p10={sorted(wall_times)[3]:.2f} p90={sorted(wall_times)[26]:.2f}")
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
