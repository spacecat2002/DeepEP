"""Fixed-input dispatch sweep; extension selection matches test_cached_metadata."""
import argparse
import json
import statistics
import time

from test_cached_metadata import deep_ep, dist, init_dist, per_token_cast_to_fp8, torch


def worker(local_rank, args):
    rank, world, group = init_dist(local_rank, args.ranks)
    buffer = deep_ep.Buffer(group, int(2e9), 0, explicitly_destroy=True)
    for tokens in args.tokens:
        x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device='cuda')
        idx = torch.rand(tokens, 256, device='cuda').topk(8, dim=-1).indices.to(deep_ep.topk_idx_t)
        weights = torch.rand(tokens, 8, dtype=torch.float32, device='cuda')
        all_x = [torch.empty_like(x) for _ in range(world)]
        all_idx = [torch.empty_like(idx) for _ in range(world)]
        dist.all_gather(all_x, x, group=group)
        dist.all_gather(all_idx, idx, group=group)
        expected = torch.cat([payload[((route // (256 // world)) == rank).any(1)]
                              for payload, route in zip(all_x, all_idx)])
        nr, _, ne, mask, _ = buffer.get_dispatch_layout(idx, 256)
        for dtype, data in [('bf16', x), ('fp8', per_token_cast_to_fp8(x))]:
            for sms in args.sms:
                for chunk in args.chunks:
                    config = deep_ep.Config(sms, chunk, 256)
                    def fresh():
                        return buffer.dispatch(data, num_tokens_per_rank=nr, num_tokens_per_expert=ne,
                                               is_token_in_rank=mask, topk_idx=idx, topk_weights=weights, config=config)
                    ref, ri, rw, _, handle, _ = fresh()
                    if dtype == 'bf16':
                        assert torch.equal(ref, expected)
                        combined, combined_weights, _ = buffer.combine(ref, handle, topk_weights=rw, config=config)
                        torch.testing.assert_close(combined.float(), x.float() * mask.sum(1, keepdim=True), rtol=0.02, atol=0.03)
                        torch.testing.assert_close(combined_weights, weights)
                    ref = tuple(t.clone() for t in ref) if isinstance(ref, tuple) else ref.clone()
                    cached = lambda: buffer.dispatch(data, handle=handle, config=config)
                    actual = cached()[0]
                    for a, b in zip(actual if isinstance(actual, tuple) else (actual,),
                                    ref if isinstance(ref, tuple) else (ref,)):
                        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for mode, fn in [('cached', cached), ('fresh', fresh)]:
                        for _ in range(10):
                            fn()
                        samples = []
                        for _ in range(3):
                            dist.barrier(group)
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            for _ in range(args.iters):
                                fn()
                            torch.cuda.synchronize()
                            elapsed = torch.tensor((time.perf_counter() - start) * 1e6 / args.iters,
                                                   dtype=torch.float64, device='cuda')
                            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
                            samples.append(elapsed.item())
                        if rank == 0:
                            print(json.dumps(dict(tokens=tokens, dtype=dtype, sms=sms, chunk=chunk,
                                                  mode=mode, median_us=statistics.median(samples), samples_us=samples)), flush=True)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, nargs='+', default=[128, 1024, 4096])
    p.add_argument('--sms', type=int, nargs='+', default=[24, 64])
    p.add_argument('--chunks', type=int, nargs='+', default=[6, 8, 16, 24, 32])
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--ranks', type=int, default=4)
    p.add_argument('--iters', type=int, default=50)
    args = p.parse_args()
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.ranks)
