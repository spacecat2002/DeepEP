# Legacy cached dispatch metadata A/B

Date: 2026-09-10. Base: `01dc3aaac82068020353dce2c302e38153c0bfaa`.
Both builds include the pre-existing layout.cu changes and 32-bit top-k
configuration in compiled.cuh. This is not a comparison with pristine HEAD.

## Environment and method

- 4 NVIDIA L20A GPUs, compute capability 10.0, NV18 links between every pair.
- PyTorch 2.11.0+cu130; CUDA compiler 13.2; driver 580.167.08.
- `TORCH_CUDA_ARCH_LIST=10.0`, default aggressive PTX instructions disabled.
- EP4, hidden=7168, experts=256, top-k=8, Config(24, 8, 256).
- Uniform random top-k routes, seeded per rank. BF16 and FP8 cached payloads.
- Five samples, 100 calls per sample, 10 warmups. Each sample measures wall
  time through GPU synchronization and takes the maximum across ranks.
  Reported values are medians. Graph replay contains 20 dispatches; its time
  is divided by 20. These are amortized call times, not profiler kernel times.
- No explicit L2 flush. No other GPU workload was present at initial inspection.
- Order: baseline, optimized, baseline repeat. Tables use the baseline repeat
  and optimized run with the same final test script. First baseline used a
  fixed route for non-cached timing and is not used in that comparison.

## Cached results

Time in microseconds; reduction is `(baseline - optimized) / baseline`.

| Tokens/rank | Payload | Eager baseline | Eager optimized | Reduction | Graph baseline | Graph optimized | Reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| 128 | BF16 | 48.870 | 48.567 | 0.62% | 43.601 | 43.338 | 0.61% |
| 128 | FP8 | 44.336 | 44.148 | 0.42% | 39.292 | 38.967 | 0.83% |
| 1024 | BF16 | 198.216 | 196.700 | 0.76% | 194.157 | 192.679 | 0.76% |
| 1024 | FP8 | 174.754 | 173.745 | 0.58% | 169.711 | 168.837 | 0.52% |
| 4096 | BF16 | 693.391 | 689.781 | 0.52% | 689.044 | 685.126 | 0.57% |
| 4096 | FP8 | 599.206 | 596.001 | 0.53% | 594.620 | 591.352 | 0.55% |

The three cached metadata outputs now return None. Rank 0 logical tensor
bytes eliminated per dispatch: 4,128 / 31,464 / 125,736 for the three sizes.
This is not a measurement of allocator reserved memory or physical allocation
traffic; PyTorch may reuse cached allocations. Graph replay does not repeat
host allocations, so its result also captures the changed device work.

## Changing routes with top-k weights

The following BF16 times include layout and dispatch. Three pre-generated
routes rotate on successive calls; random route generation is excluded.
Capacity uses `num_worst_tokens=tokens*world_size`.

| Tokens/rank | Fresh baseline | Fresh optimized | Capacity baseline | Capacity optimized |
|---|---:|---:|---:|---:|
| 128 | 70.696 | 70.791 | 66.833 | 66.945 |
| 1024 | 226.101 | 226.192 | 222.158 | 222.330 |
| 4096 | 754.971 | 755.974 | 749.824 | 751.586 |

Non-cached differences are 0.04%-0.23% slower in this run. No material
regression is established by these measurements. Capacity saves about
3.8-4.4 us in the optimized run, but allocates a worst-case output and returns
an empty per-expert count list. Downstream processing must handle padding.

## Correctness and scope

The dedicated test passed for baseline and optimized EP4 at 128/1024/4096:
BF16/FP8 equality, repeated original-handle reuse, handle immutability across
cached dispatch, combine after cached dispatch, asynchronous operation with
and without communication-stream allocation, CUDA graphs, changing routes,
top-k weights round trip, capacity prefix equality and -1 padded indices.
The optimized build additionally asserts three None outputs at the C++ API.
The original test_intranode.py passed all 24 correctness combinations at
EP4/1024/7168, including previous events, sync/async, BF16/FP8, top-k weights,
capacity outputs and combine, followed by its tuning loops.
The optimized dedicated test also passed EP2 with 1 and 129 tokens per rank
(`--ranks 2 --tokens 1 129 --iters 20 --expect-no-metadata`), covering empty
channels and uneven channel task ranges. Output: `/tmp/deepev1-ep2-test.log`.

The measured cached benefit is small, around 0.4%-0.8%, with visible variation
in some samples. It is not a model inference speedup. Cached dispatch requires
unchanged routing and does not accept top-k indices/weights. No combine rewrite,
multi-node validation, sanitizer run or production inference trace is included.

## Reproduction and artifacts

```sh
env TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=8 /home/admin/workspace/sglang/.venv/bin/python setup.py build_ext --inplace
env PYTHONPATH=. /home/admin/workspace/sglang/.venv/bin/python tests/legacy/test_cached_metadata.py --expect-no-metadata
env PYTHONPATH=. /home/admin/workspace/sglang/.venv/bin/python tests/legacy/test_intranode.py --num-processes 4 --num-tokens 1024 --hidden 7168
```

To select a preserved build, set `DEEPEP_TEST_EXTENSION` to its absolute path.
Omit `--expect-no-metadata` for the baseline. Current-machine artifacts:

- `/tmp/deepev1-baseline.so`, SHA256 `878de0e43485850d241e0349b6ef043fd31a4278acdc96a58a6ec25649291dc8`
- `/tmp/deepev1-optimized.so`, SHA256 `a3eab1ecb37ab27a3d641fb2a8bd84b9912d13b99203fd5dae4afd9243710512`
- `/tmp/deepev1-baseline-repeat.log`, `/tmp/deepev1-optimized-test.log`: all samples.
- `/tmp/deepev1-original-suite.log`: original test output.
- `/tmp/deepev1-baseline-build.log`, `/tmp/deepev1-optimized-build.log`: build output.

Temporary artifacts are not tracked by git and may be removed by system cleanup.
