# Dispatch experiments on NVLink EP4

Environment and starting tree: see cached_metadata_test.md. All measurements
below use SM=64, hidden=7168, experts=256, top-k=8, uniform random routing.
Reference includes the cached metadata patch and the user's layout/32-bit
index changes. It is not pristine upstream DeepEP.

## Config sweep

bench_dispatch_sweep.py tests SM=24/64 and chunks=6/8/16/24/32. Each case
uses 10 warmups and three samples of 50 calls, with device synchronization
around each sample, maximum elapsed time across ranks, then median.
Fresh includes weights and notify/dispatch, but excludes layout and routing
generation. Routes are fixed during timing. This is amortized eager latency,
not isolated GPU kernel duration or inference latency.

SM=64 reference repeat, microseconds:

| Tokens | Payload | Mode | Chunk 6 | Best | Best chunk | Speedup |
|---:|---|---|---:|---:|---:|---:|
| 1024 | BF16 | cached | 121.15 | 112.33 | 8 | 1.079 |
| 1024 | BF16 | fresh | 140.45 | 131.17 | 8 | 1.071 |
| 1024 | FP8 | cached | 103.92 | 79.14 | 24 | 1.313 |
| 1024 | FP8 | fresh | 122.83 | 97.31 | 24 | 1.262 |
| 4096 | BF16 | cached | 365.70 | 320.01 | 16 | 1.143 |
| 4096 | BF16 | fresh | 384.44 | 342.44 | 16 | 1.123 |
| 4096 | FP8 | cached | 316.94 | 185.55 | 24 | 1.708 |
| 4096 | FP8 | fresh | 338.06 | 209.89 | 24 | 1.611 |

At 128 tokens configuration benefits were at most about 1.6% in the repeat.
No universal default was changed: the best chunk depends on size and dtype.

## Kernel experiments

Only DEEPEP_TEST_VARIANT=REMOTE_DIRECT remains available. Unset it for
normal behavior. The ineffective TMA and LOCAL_DIRECT implementations and
build options were removed after evaluation; their historical results follow.

- TMA_PIPELINE: receiver stages four quarter-token transfers through two
  shared-memory regions, without increasing shared-memory allocation.
  Correctness passed at SM64/chunk24/tokens128,1024,4096. The sweep showed
  no useful improvement over the tuned reference.
- LOCAL_DIRECT: local-rank hidden payload goes directly into recv_x; metadata
  and remote payload still use the ring. Correctness passed; best sweep
  differences were small, generally around 0-2%.
- REMOTE_DIRECT: hidden payload goes directly to a reserved shared output
  region on every destination. Metadata and completion still use the ring.
  Reads destination rank-prefix data from the destination GPU. An initial
  version incorrectly read the source's prefix and failed the independent
  reference check; that version is not a valid performance result.
- SENDER_TMA: hidden loads overlap metadata processing, then TMA stores write
  to the normal ring. REMOTE_TMA combines this with REMOTE_DIRECT.
- REMOTE_TMA_FULL: 384 threads and 16 KiB/warp, still 192 KiB/block,
  transfer a full token per TMA load/store. Experimental EP2/EP4 only;
  12 warps do not divide evenly across EP8. Tested hidden size is 7168.

REMOTE_DIRECT is an experimental borrowed-output API: its output is overwritten
by the next dispatch and invalid after buffer destruction. It reserves output
at byte offset 1 GiB, with capacity and ring-overlap checks. It must not be
enabled for applications expecting normal tensor ownership or arbitrary
overlapping dispatches. It is not a production implementation of SwiftEP.

REMOTE_DIRECT passed independent all-gather BF16 output comparisons and
combine/weight checks in the sweep, plus the dedicated BF16/FP8, asynchronous,
capacity, changing-route and graph checks at SM64/chunk24/tokens128,1024,4096.
References are cloned before another dispatch to avoid aliasing false positives.

Its 100-iteration repeat at 4096 tokens/chunk32 measured 287.54 us cached and
313.10 us fresh for BF16. Fresh speedup is 1.228 versus reference chunk6 and
1.094 versus tuned reference. FP8 fresh was 215.41 us, worse than the tuned
209.89 us reference. Buffer fusion alone is not a universal win.

## Sender TMA ablation

SM64 fresh (weights included), best measured chunk for each variant, us.
Reference uses chunks6/8/16/24/32; sender TMA and half-token remote TMA
use chunks16/24/32; full-token remote TMA uses chunks8/16/24/32.
These are best-of-sweep results, not paired same-chunk speedups.

| Tokens | Payload | Tuned reference | Direct stores | Sender TMA, ring | Sender TMA, direct | Full-token TMA, direct |
|---:|---|---:|---:|---:|---:|---:|
| 128 | BF16 | 53.68 | 52.91 | 53.21 | 53.08 | 55.77 |
| 128 | FP8 | 50.62 | 51.33 | 51.29 | 52.04 | 54.53 |
| 1024 | BF16 | 131.17 | 110.96 | 126.18 | 110.79 | 111.71 |
| 1024 | FP8 | 97.31 | 89.58 | 96.62 | 91.49 | 104.86 |
| 4096 | BF16 | 342.44 | 312.65 | 323.76 | 316.74 | 320.32 |
| 4096 | FP8 | 209.89 | 213.09 | 214.61 | 243.88 | 276.07 |

The tested sender-TMA variants did not consistently improve on tuned direct
stores. Full-token transfers changed both transaction granularity and warp
parallelism, so the results cannot isolate either factor. No universal 1.2x
gain over the tuned reference has been demonstrated. Gains over chunk6 are
real for the measured cases but include configuration tuning.

All sender-TMA sweeps passed independent BF16 output/weights/combine checks
and FP8 cached/fresh equality. REMOTE_TMA additionally passed the dedicated
async, communication-stream allocation, graph, capacity and changing-route
checks at SM64/chunk24/tokens128,1024,4096. REMOTE_TMA_FULL passed the
same dedicated checks at all three sizes. No sanitizer, production trace,
TBO throughput or arbitrary output-lifetime validation is claimed.

Additional logs: /tmp/deepev1-sender-tma-sweep.log,
/tmp/deepev1-remote-tma-sweep.log, /tmp/deepev1-full-tma-sweep.log,
/tmp/deepev1-remote-tma-check.log, /tmp/deepev1-full-tma-check.log.
The full-token experiment is no longer buildable from the current source.

## Reproduction

```sh
env TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=8 DEEPEP_TEST_VARIANT=REMOTE_DIRECT /home/admin/workspace/sglang/.venv/bin/python setup.py build_ext --inplace
env PYTHONPATH=. /home/admin/workspace/sglang/.venv/bin/python tests/legacy/bench_dispatch_sweep.py --sms 64
env PYTHONPATH=. /home/admin/workspace/sglang/.venv/bin/python tests/legacy/test_cached_metadata.py --sms 64 --chunk 24 --expect-no-metadata
```

DEEPEP_TEST_EXTENSION selects a preserved extension in a fresh Python process.
Logs on this machine: /tmp/deepev1-sweep-reference.log,
/tmp/deepev1-reference-repeat.log, /tmp/deepev1-pipeline-sweep.log,
/tmp/deepev1-direct-sweep.log, /tmp/deepev1-remote-fixed-sweep.log,
/tmp/deepev1-remote-repeat.log. These temporary artifacts are not git-tracked.
