#include <deep_ep/common/exception.cuh>

#include "compiled.cuh"
#include "launch.cuh"

namespace deep_ep::legacy {

namespace layout {

template <int kNumThreads>
__global__ void get_dispatch_layout_atomic(const topk_idx_t* topk_idx,
                                           int* num_tokens_per_rank,
                                           int* num_tokens_per_rdma_rank,
                                           int* num_tokens_per_expert,
                                           bool* is_token_in_rank,
                                           int num_tokens,
                                           int num_topk,
                                           int num_ranks,
                                           int num_experts) {
    const auto thread_idx = static_cast<int>(blockIdx.x) * kNumThreads + static_cast<int>(threadIdx.x);
    const auto thread_stride = static_cast<int>(gridDim.x) * kNumThreads;
    const auto num_experts_per_rank = num_experts / num_ranks;

    for (int token_idx = thread_idx; token_idx < num_tokens; token_idx += thread_stride) {
        const auto token_topk = topk_idx + token_idx * num_topk;
        for (int i = 0; i < num_topk; ++i) {
            const auto expert_idx = static_cast<int>(token_topk[i]);
            if (expert_idx < 0 or expert_idx >= num_experts)
                continue;

            atomicAdd(num_tokens_per_expert + expert_idx, 1);
            const auto rank_idx = expert_idx / num_experts_per_rank;
            bool first_rank = true, first_rdma_rank = true;
            for (int j = 0; j < i; ++j) {
                const auto previous_expert_idx = static_cast<int>(token_topk[j]);
                if (previous_expert_idx < 0 or previous_expert_idx >= num_experts)
                    continue;
                const auto previous_rank_idx = previous_expert_idx / num_experts_per_rank;
                first_rank &= previous_rank_idx != rank_idx;
                first_rdma_rank &= previous_rank_idx / LEGACY_NUM_MAX_NVL_PEERS != rank_idx / LEGACY_NUM_MAX_NVL_PEERS;
            }
            if (first_rank) {
                is_token_in_rank[token_idx * num_ranks + rank_idx] = true;
                atomicAdd(num_tokens_per_rank + rank_idx, 1);
            }
            if (num_tokens_per_rdma_rank != nullptr and first_rdma_rank)
                atomicAdd(num_tokens_per_rdma_rank + rank_idx / LEGACY_NUM_MAX_NVL_PEERS, 1);
        }
    }
}

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(const topk_idx_t* topk_idx,
                                    int* num_tokens_per_rank,
                                    int* num_tokens_per_rdma_rank,
                                    int* num_tokens_per_expert,
                                    bool* is_token_in_rank,
                                    int num_tokens,
                                    int num_topk,
                                    int num_ranks,
                                    int num_experts) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    if (expert_begin_idx < expert_end_idx) {
        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0 and num_ranks > LEGACY_NUM_MAX_NVL_PEERS);

    // Count rank statistics
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / LEGACY_NUM_MAX_NVL_PEERS;
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx = rank_begin_idx / LEGACY_NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / LEGACY_NUM_MAX_NVL_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto expert_begin = rank_begin_idx * num_expert_per_rank;
        auto expert_end = rank_end_idx * num_expert_per_rank;

        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
            #pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / LEGACY_NUM_MAX_NVL_PEERS]++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

            #pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream) {
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    if (num_ranks > LEGACY_NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(num_tokens_per_rdma_rank == nullptr or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);
        const auto num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(num_tokens_per_rank, 0, num_ranks * sizeof(int), stream));
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(num_tokens_per_expert, 0, num_experts * sizeof(int), stream));
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(is_token_in_rank, 0, static_cast<size_t>(num_tokens) * num_ranks * sizeof(bool), stream));
        if (num_tokens_per_rdma_rank != nullptr)
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(num_tokens_per_rdma_rank, 0, num_rdma_ranks * sizeof(int), stream));

        const auto num_sms = min(32, max(1, (num_tokens + kNumThreads - 1) / kNumThreads));
        SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
        LAUNCH_KERNEL(&cfg,
                      (get_dispatch_layout_atomic<kNumThreads>),
                      topk_idx,
                      num_tokens_per_rank,
                      num_tokens_per_rdma_rank,
                      num_tokens_per_expert,
                      is_token_in_rank,
                      num_tokens,
                      num_topk,
                      num_ranks,
                      num_experts);
        return;
    }
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    EP_STATIC_ASSERT(kNumRanksPerSM % LEGACY_NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg,
                  (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
                  topk_idx,
                  num_tokens_per_rank,
                  num_tokens_per_rdma_rank,
                  num_tokens_per_expert,
                  is_token_in_rank,
                  num_tokens,
                  num_topk,
                  num_ranks,
                  num_experts);
}

}  // namespace layout

}  // namespace deep_ep::legacy
