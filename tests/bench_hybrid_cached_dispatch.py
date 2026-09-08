"""A/B cached dispatch with the same runtime, inputs, handle and CUDA kernels.

PYTHONPATH=. python tests/bench_hybrid_cached_dispatch.py
Baseline Python source is read from the repository HEAD; no checkout is changed.
"""
import statistics
import subprocess
import types
import inspect

import torch
import torch.distributed as dist
import deep_ep
import hybrid_ep_cpp
import deep_ep.hybrid_ep_buffer as current_module
from utils import bench, init_dist


def main(rank, source):
    _, _, group = init_dist(rank, 4)
    torch.manual_seed(1234 + rank)
    original = types.ModuleType('baseline_python')
    exec(compile(source, 'HEAD:deep_ep/hybrid_ep_buffer.py', 'exec'), original.__dict__)
    buffer = deep_ep.HybridEPBuffer(group, 2048, 4096, 64, load_cached_kernels=True)
    baseline = original.HybridEPBuffer.__new__(original.HybridEPBuffer)
    baseline.__dict__ = buffer.__dict__
    cached_python = types.ModuleType('cached_python')
    exec(compile(inspect.getsource(current_module), 'cached_python', 'exec'), cached_python.__dict__)
    cached_python.dense_indices_to_probs = original.dense_indices_to_probs
    cached_only = cached_python.HybridEPBuffer.__new__(cached_python.HybridEPBuffer)
    cached_only.__dict__ = buffer.__dict__
    hidden = torch.randn(4096, 2048, dtype=torch.bfloat16, device='cuda')
    idx = torch.rand(4096, 256, device='cuda').topk(8, dim=1).indices
    weights = torch.rand(4096, 8, dtype=torch.float32, device='cuda')
    probs = torch.zeros(4096, 256, dtype=torch.float32, device='cuda')
    probs.scatter_(1, idx, weights)
    *_, handle = buffer.dispatch(hidden, topk_idx=idx, probs=probs, num_of_experts=256)
    for mode in ('dense_probs', 'topk_weights'):
        args = dict(hidden=hidden, topk_idx=idx, num_of_experts=256, handle=handle)
        args.update(probs=probs) if mode == 'dense_probs' else args.update(topk_weights=weights)
        ref = baseline.dispatch(**args)
        out = buffer.dispatch(**args)
        torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)
        start, end = rank * 64, (rank + 1) * 64
        torch.testing.assert_close(out[1][:, start:end], ref[1][:, start:end], rtol=0, atol=0)
        values = {'baseline': [], 'cached_only': [], 'current': []}
        for trial in range(5):
            variants = [('baseline', baseline), ('cached_only', cached_only), ('current', buffer)]
            if trial % 2:
                variants.reverse()
            for name, variant in variants:
                dist.barrier()
                elapsed = bench(lambda: variant.dispatch(**args), num_warmups=10, num_tests=30)[0]
                slowest = torch.tensor(elapsed, dtype=torch.float64, device='cuda')
                dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
                values[name].append(slowest.item() * 1e6)
        if rank == 0:
            print(mode, {k: round(statistics.median(v), 2) for k, v in values.items()},
                  'us; median of five alternating trials, slowest rank', flush=True)
    if rank == 0:
        print('runtime:', hybrid_ep_cpp.__file__, flush=True)
    dist.barrier()
    del baseline, cached_only, buffer
    dist.destroy_process_group()


if __name__ == '__main__':
    print('baseline revision:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), flush=True)
    source = subprocess.check_output(['git', 'show', 'HEAD:deep_ep/hybrid_ep_buffer.py'], text=True)
    torch.multiprocessing.spawn(main, args=(source,), nprocs=4)
