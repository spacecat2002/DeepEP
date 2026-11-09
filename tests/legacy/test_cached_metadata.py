"""Run with PYTHONPATH=. python tests/legacy/test_cached_metadata.py.

DEEPEP_TEST_EXTENSION selects a separately built extension for A/B runs.
"""
import argparse
import importlib.util
import itertools
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

if os.getenv('DEEPEP_TEST_EXTENSION'):
    spec = importlib.util.spec_from_file_location('deep_ep._C', os.environ['DEEPEP_TEST_EXTENSION'])
    module = importlib.util.module_from_spec(spec)
    sys.modules['deep_ep._C'] = module
    spec.loader.exec_module(module)

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import per_token_cast_to_fp8


def worker(local_rank, args):
    rank, world, group = init_dist(local_rank, args.ranks)
    buffer = deep_ep.Buffer(group, int(2e9), 0, explicitly_destroy=True)
    config = deep_ep.Config(args.sms, args.chunk, 256)

    def measure(fn, graph=False):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    fn()
            fn = g.replay
        samples = []
        for _ in range(5):
            dist.barrier(group)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1e6 / args.iters / (20 if graph else 1)
            value = torch.tensor(elapsed, dtype=torch.float64, device='cuda')
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
            samples.append(value.item())
        return {'median_us': statistics.median(samples), 'samples_us': samples}

    for tokens in args.tokens:
        x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device='cuda')
        idx = torch.rand(tokens, 256, device='cuda').topk(8, dim=-1).indices.to(deep_ep.topk_idx_t)
        weights = torch.rand(tokens, 8, dtype=torch.float32, device='cuda')

        def fresh(capacity=0, route=None):
            route = idx if route is None else route
            nr, _, ne, mask, _ = buffer.get_dispatch_layout(route, 256)
            return buffer.dispatch(x, num_tokens_per_rank=nr, num_tokens_per_expert=ne,
                                   is_token_in_rank=mask, topk_idx=route, topk_weights=weights,
                                   num_worst_tokens=capacity, config=config)

        for _ in range(3):
            idx.copy_(torch.rand(tokens, 256, device='cuda').topk(8, dim=-1).indices)
            ref, ri, rw, _, handle, _ = fresh()
            ref = ref.clone()
            out, oi, ow, counts, _, _ = fresh(tokens * world)
            assert not counts
            assert torch.equal(out[:len(ref)], ref)
            assert torch.equal(oi[:len(ref)], ri)
            assert torch.equal(ow[:len(ref)], rw)
            assert (oi[len(ref):] == -1).all()
            combined, combined_weights, _ = buffer.combine(ref, handle, topk_weights=rw, config=config)
            torch.testing.assert_close(combined.float(), x.float() * handle[4].sum(1, keepdim=True), rtol=0.02, atol=0.03)
            torch.testing.assert_close(combined_weights, weights)

        raw = buffer.runtime.intranode_dispatch(
            x, None, None, None, None, handle[4], None, len(handle[3]), handle[0], handle[1],
            1, 0, config, None, False, False)
        metadata_bytes = sum(t.numel() * t.element_size() for t in raw[7:10] if t is not None)
        if args.expect_no_metadata:
            assert raw[7:10] == (None, None, None)
        if rank == 0:
            print(json.dumps(dict(tokens=tokens, metadata_bytes=metadata_bytes)), flush=True)
        del raw
        for dtype, data in [('bf16', x), ('fp8', per_token_cast_to_fp8(x))]:
            reference = buffer.dispatch(data, handle=handle, config=config)[0]
            reference = tuple(t.clone() for t in reference) if isinstance(reference, tuple) else reference.clone()
            for async_mode, allocate in [(False, False), (True, False), (True, True)]:
                for _ in range(4):
                    before = [t.clone() for t in handle]
                    out, _, _, _, new_handle, event = buffer.dispatch(
                        data, handle=handle, config=config, async_finish=async_mode,
                        previous_event=buffer.capture() if async_mode else None,
                        allocate_on_comm_stream=allocate)
                    if async_mode:
                        event.current_stream_wait()
                    assert new_handle is None
                    for old, saved in zip(handle, before):
                        assert torch.equal(old, saved)
                    for actual, expected in zip(out if isinstance(out, tuple) else (out,),
                                                reference if isinstance(reference, tuple) else (reference,)):
                        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                    if dtype == 'bf16':
                        combined, _, _ = buffer.combine(out, handle, config=config)
                        torch.testing.assert_close(combined.float(), x.float() * handle[4].sum(1, keepdim=True), rtol=0.02, atol=0.03)
            for graph in [False, True]:
                timing = measure(lambda: buffer.dispatch(data, handle=handle, config=config), graph)
                if rank == 0:
                    print(json.dumps(dict(tokens=tokens, dtype=dtype, mode='cached', graph=graph, **timing)), flush=True)
        routes = [torch.rand(tokens, 256, device='cuda').topk(8, dim=-1).indices.to(deep_ep.topk_idx_t) for _ in range(3)]
        for mode, capacity in [('fresh', 0), ('capacity', tokens * world)]:
            route_iter = itertools.cycle(routes)
            timing = measure(lambda: fresh(capacity, next(route_iter)))
            if rank == 0:
                print(json.dumps(dict(tokens=tokens, dtype='bf16', mode=mode, graph=False, **timing)), flush=True)
        if rank == 0:
            print(f'PASS tokens={tokens}', flush=True)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--tokens', type=int, nargs='+', default=[128, 1024, 4096])
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--sms', type=int, default=24)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--expect-no-metadata', action='store_true')
    args = parser.parse_args()
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.ranks)
