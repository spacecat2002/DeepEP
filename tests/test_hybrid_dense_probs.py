"""PYTHONPATH=. python tests/test_hybrid_dense_probs.py"""
import torch
import hybrid_ep_cpp
from deep_ep.hybrid_ep_buffer import dense_indices_to_probs


def reference(idx, weights, experts):
    valid = (idx >= 0) & (idx < experts)
    result = torch.zeros(idx.size(0), experts, dtype=torch.float32, device=idx.device)
    result.scatter_add_(1, torch.where(valid, idx, 0).long(),
                        torch.where(valid, weights, 0).float())
    return result


def main():
    torch.manual_seed(2026)
    # Include duplicate experts, invalid IDs, tail rows, empty K/T and fallbacks.
    for dtype in (torch.int32, torch.int64):
        for experts in (1, 64, 256, 1024, 1025):
            for tokens, topk in ((0, 8), (1, 0), (33, 8), (129, 32), (3, 33)):
                idx = torch.randint(-2, experts + 2, (tokens, topk), dtype=dtype, device='cuda')
                weights = torch.randn(tokens, topk, device='cuda', dtype=torch.float32)
                result = dense_indices_to_probs(idx, weights, tokens, experts)
                torch.testing.assert_close(result, reference(idx, weights, experts), atol=1e-6, rtol=1e-6)
    idx = torch.tensor([[0, 0, -1, 256, 255, 32, 32, -2]], device='cuda')
    weights = torch.tensor([[1., 2., float('nan'), float('inf'), -4., 5., 6., 1.]], device='cuda')
    torch.testing.assert_close(dense_indices_to_probs(idx, weights, 1, 256),
                               reference(idx, weights, 256), atol=0, rtol=0)
    for w in (weights.bfloat16(), weights[:, ::2]):
        i = idx if w.size(1) == 8 else idx[:, ::2]
        torch.testing.assert_close(dense_indices_to_probs(i, w, 1, 256), reference(i, w, 256))
    grad_weights = torch.ones_like(weights, requires_grad=True)
    dense_indices_to_probs(idx, grad_weights, 1, 256).sum().backward()
    torch.testing.assert_close(grad_weights.grad, ((idx >= 0) & (idx < 256)).float())
    # The allocation/launch must use the caller's stream and support graph replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dense_indices_to_probs(idx, weights, 1, 256)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = dense_indices_to_probs(idx, weights, 1, 256)
    weights.add_(1)
    graph.replay()
    torch.testing.assert_close(captured, reference(idx, weights, 256), atol=0, rtol=0)
    try:
        hybrid_ep_cpp.dense_topk_probs(idx, weights[:, :2], 256)
    except RuntimeError:
        pass
    else:
        raise AssertionError('extension accepted mismatched shapes')
    print('dense probs: CUDA types, masks, duplicates, empty shapes, fallbacks, graph replay PASS')


if __name__ == '__main__':
    main()
