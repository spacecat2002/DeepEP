"""Run with PYTHONPATH=. python tests/test_hybrid_cached_routing.py."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import deep_ep.hybrid_ep_buffer as module


def check_cached_routing():
    buffer = module.HybridEPBuffer.__new__(module.HybridEPBuffer)
    buffer.configurer = SimpleNamespace(
        buffer_config=SimpleNamespace(num_of_experts_per_rank=64)
    )
    buffer.num_of_hybrid_ep_ranks_per_nvlink_domain = 4
    buffer.num_of_nodes = 1
    idx = torch.tensor([[0, 32, -1, 255], [64, 64, 256, -2]])
    weights = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    kwargs = dict(topk_idx=idx, topk_weights=weights, num_of_tokens=2,
                  num_of_experts=256, probs=None, routing_map=None)
    expected = torch.zeros(2, 256)
    expected[0, 0], expected[0, 32], expected[0, 255] = 1., 2., 4.
    expected[1, 64] = 11.
    _, routing, probs, _ = buffer._prepare_routing_data(**kwargs, cached=True)
    assert routing is None
    torch.testing.assert_close(probs, expected, rtol=0, atol=0)
    # Explicit maps retain precedence even if conflicting indices are supplied.
    routing_map = torch.zeros(2, 256, dtype=torch.bool)
    _, routing, probs, _ = buffer._prepare_routing_data(
        **dict(kwargs, routing_map=routing_map), cached=True
    )
    assert routing is routing_map and probs is None
    for bad in (dict(num_of_experts=128), dict(probs=torch.zeros(2, 128))):
        try:
            buffer._prepare_routing_data(**dict(kwargs, **bad), cached=True)
        except AssertionError:
            pass
        else:
            raise AssertionError("cached routing bypassed shape/layout validation")


def check_count_readiness():
    buffer = module.HybridEPBuffer.__new__(module.HybridEPBuffer)
    fields = dict(sparse_to_dense_map=None, rdma_to_attn_map=None,
                  attn_to_rdma_map=None, num_dispatched_tokens_tensor=None,
                  local_expert_routing_map=None, num_of_tokens_per_rank=2,
                  config=None)
    ready = False

    def synchronize():
        nonlocal ready
        ready = True

    def dispatch(**kwargs):
        assert ready, "CPU pinned count read before metadata completion"
        return kwargs['hidden'], None, None

    buffer.update_template_config = Mock(return_value=None)
    buffer.runtime = SimpleNamespace(
        metadata_preprocessing=Mock(return_value=SimpleNamespace(**fields)),
        dispatch=dispatch,
    )
    stream = SimpleNamespace(synchronize=Mock(side_effect=synchronize))
    with patch.object(module.torch.cuda, 'current_stream', return_value=stream), \
         patch.object(module.hybrid_ep_cpp, 'HandleImpl', SimpleNamespace):
        hidden = torch.zeros(2, 16)
        *_, handle = buffer.dispatch(hidden, routing_map=torch.zeros(2, 256, dtype=torch.bool))
        assert stream.synchronize.call_count == 1
        buffer.dispatch(hidden, handle=handle)
        assert stream.synchronize.call_count == 1, "cached handle unnecessarily waited"
        assert buffer.runtime.metadata_preprocessing.call_count == 1


if __name__ == '__main__':
    check_cached_routing()
    check_count_readiness()
    print('cached routing semantics and count readiness: PASS')
