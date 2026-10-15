#include <cstdlib>
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace deep_ep {

namespace layout {

// One pass over TopK, with shared histograms. The original implementation
// rereads the complete router tensor for every four-expert slice.
__global__ void ep4_layout_single_pass(const int64_t* ids, int* rank_counts,
                                      int* expert_counts, bool* token_in_rank,
                                      int tokens, int experts) {
    __shared__ int histogram[512], counts[4];
    for (int i = threadIdx.x; i < experts; i += blockDim.x) histogram[i] = 0;
    if (threadIdx.x < 4) counts[threadIdx.x] = 0;
    __syncthreads();
    int local_counts[4] = {};
    const int per_rank = experts / 4;
    for (int token = blockIdx.x * blockDim.x + threadIdx.x;
         token < tokens; token += gridDim.x * blockDim.x) {
        unsigned mask = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int e = static_cast<int>(ids[static_cast<int64_t>(token) * 8 + k]);
            if (e >= 0 && e < experts) {
                atomicAdd(histogram + e, 1);
                mask |= 1u << (e / per_rank);
            }
        }
        #pragma unroll
        for (int rank = 0; rank < 4; ++rank) {
            const bool present = mask & (1u << rank);
            token_in_rank[static_cast<int64_t>(token) * 4 + rank] = present;
            local_counts[rank] += present;
        }
    }
    #pragma unroll
    for (int rank = 0; rank < 4; ++rank) atomicAdd(counts + rank, local_counts[rank]);
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        atomicAdd(expert_counts + e, histogram[e]);
    if (threadIdx.x < 4) atomicAdd(rank_counts + threadIdx.x, counts[threadIdx.x]);
}

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__
void get_dispatch_layout(const int64_t* topk_idx,
                                    int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                                    int* num_tokens_per_expert, bool* is_token_in_rank,
                                    int num_tokens, int num_topk, int num_ranks, int num_experts) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    if (expert_begin_idx < expert_end_idx) {
        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++ i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++ j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++ num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++ i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

    // Count rank statistics
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto expert_begin = rank_begin_idx * num_expert_per_rank;
        auto expert_end = rank_end_idx * num_expert_per_rank;

        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++ i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++ i)
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
                    is_in_rank[rank_idx] ++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS] ++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++ j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

            #pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++ j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++ i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++ i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(const int64_t* topk_idx,
                         int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank,
                         int num_tokens, int num_topk, int num_ranks, int num_experts,
                         cudaStream_t stream) {
    const char* fast_env = std::getenv("DEEPEP_SINGLE_PASS_LAYOUT");
    if (fast_env && fast_env[0] == '1' && fast_env[1] == '\0' &&
        num_ranks == 4 && num_topk == 8 && num_experts <= 512 && !num_tokens_per_rdma_rank) {
        CUDA_CHECK(cudaMemsetAsync(num_tokens_per_rank, 0, 4 * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(num_tokens_per_expert, 0, num_experts * sizeof(int), stream));
        SETUP_LAUNCH_CONFIG(20, 256, stream);
        LAUNCH_KERNEL(&cfg, ep4_layout_single_pass, topk_idx, num_tokens_per_rank,
            num_tokens_per_expert, is_token_in_rank, num_tokens, num_experts);
        return;
    }
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
                  topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank,
                  num_tokens, num_topk, num_ranks, num_experts);
}

} // namespace layout

} // namespace deep_ep
