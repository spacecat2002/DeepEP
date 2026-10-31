#include "configs.cuh"
#include "buffer.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "api.cuh"

namespace deep_ep {

namespace intranode {

inline bool env_flag_default(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return value[0] == '1' && value[1] == '\0';
}

template <int kNumRanks>
__device__ __forceinline__ void load_rank_bits(const bool* row, uint64_t (&bits)[2]) {
    if constexpr (kNumRanks == 2)
        bits[0] = __ldg(reinterpret_cast<const uint16_t*>(row));
    else if constexpr (kNumRanks == 4)
        bits[0] = __ldg(reinterpret_cast<const uint32_t*>(row));
    else if constexpr (kNumRanks == 8)
        bits[0] = __ldg(reinterpret_cast<const uint64_t*>(row));
    else {
        bits[0] = __ldg(reinterpret_cast<const uint64_t*>(row));
        bits[1] = __ldg(reinterpret_cast<const uint64_t*>(row + 8));
    }
}

template<int kNumRanks>
__global__ void
notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped,
                const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                int num_tokens, int num_channels, const bool* is_token_in_rank, int* channel_prefix_matrix,
                int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment,
                void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;

    if (sm_id == 0) {
        // Barrier first
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {
            per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
            per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];
        }
        // Publish expert counts with coalesced workers instead of four
        // threads serially issuing every peer's expert-count stores.
        for (int expert = thread_id; expert < num_experts; expert += num_threads) {
            const int destination = expert / num_experts_per_rank;
            const int local_expert = expert % num_experts_per_rank;
            auto destination_counts = static_cast<int*>(buffer_ptrs[destination]) + kNumRanks * kNumRanks;
            destination_counts[rank * num_experts_per_rank + local_expert] = num_tokens_per_expert[expert];
        }

        // Wait for all ranks to be finished
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
        if (thread_id < kNumRanks) {
            #pragma unroll
            for (int i = 1; i < kNumRanks; ++ i)
                local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
            if (thread_id == rank)
                *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++ i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }
        // Finish expert reads before copy/queue clearing.
        __syncthreads();

        // Copy rank size prefix matrix to another tensor
        #pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

        // Extra memset for later communication queue
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_buffer[i] = 0;

        // Barrier
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        int channel_id = (sm_id - 1) * num_warps + warp_id;
        if (channel_id < num_channels) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            int counts[kNumRanks] = {0};

            // Iterate over tokens once, accumulating all destination ranks.
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                uint64_t bits[2];
                load_rank_bits<kNumRanks>(is_token_in_rank + i * kNumRanks, bits);
                #pragma unroll
                for (int dst_rank = 0; dst_rank < kNumRanks; ++dst_rank)
                    counts[dst_rank] += (bits[dst_rank / 8] >> ((dst_rank % 8) * 8)) & 0xff;
            }

            #pragma unroll
            for (int dst_rank = 0; dst_rank < kNumRanks; ++dst_rank)
                counts[dst_rank] = warp_reduce_sum(counts[dst_rank]);

            if (elect_one_sync()) {
                #pragma unroll
                for (int dst_rank = 0; dst_rank < kNumRanks; ++dst_rank)
                    channel_prefix_matrix[dst_rank * num_channels + channel_id] = counts[dst_rank];
            }
        }
    }
}

template <int kNumRanks>
__global__ void
prefix_sum(int* channel_prefix_matrix, int num_channels) {
    auto rank = static_cast<int>(threadIdx.x);
    if (rank < kNumRanks) {
        auto prefix_row = channel_prefix_matrix + rank * num_channels;
        #pragma unroll
        for (int i = 1; i < num_channels; ++i)
            prefix_row[i] += prefix_row[i - 1];
    }
}

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, const bool* is_token_in_rank, int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment,
                     void** buffer_ptrs, int** barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks) { \
    auto kernel = notify_dispatch<ranks>; \
    LAUNCH_KERNEL(&cfg, kernel, \
        num_tokens_per_rank, moe_recv_counter_mapped, \
        num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts, \
        num_tokens, num_channels, is_token_in_rank, channel_prefix_matrix, \
        rank_prefix_matrix_copy, num_memset_int, expert_alignment, \
        buffer_ptrs, barrier_signal_ptrs, rank); \
    LAUNCH_KERNEL(&prefix_cfg, prefix_sum<ranks>, channel_prefix_matrix, num_channels); \
    } break

    constexpr int kNumThreads = 128;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

    SETUP_LAUNCH_CONFIG(1 + (num_channels + kNumThreads / 32 - 1) / (kNumThreads / 32), kNumThreads, stream);
    cudaLaunchConfig_t prefix_cfg = {dim3(1), dim3(kNumThreads), 0, stream, nullptr, 0};
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

__global__ void deepgemm_init_layout(const int* num_recv_tokens_per_expert,
                                     int* expert_start_loc, int* m_indices) {
    const int expert = static_cast<int>(blockIdx.x);
    int start = 0;
    for (int i = 0; i < expert; ++i)
        start += num_recv_tokens_per_expert[i];
    const int count = num_recv_tokens_per_expert[expert];
    if (threadIdx.x == 0)
        expert_start_loc[expert] = start;
    for (int i = static_cast<int>(threadIdx.x); i < count;
         i += static_cast<int>(blockDim.x))
        m_indices[start + i] = expert;
}

__global__ void deepgemm_scatter_bf16(const int4* recv_x,
                                      const int64_t* recv_topk_idx,
                                      int4* output, int64_t* output_index,
                                      int* expert_start_loc, int hidden_int4,
                                      int num_topk) {
    const int token = static_cast<int>(blockIdx.x);
    __shared__ int dst;
    for (int k = 0; k < num_topk; ++k) {
        if (threadIdx.x == 0) {
            const int64_t expert = recv_topk_idx[token * num_topk + k];
            dst = expert >= 0 ? atomicAdd(expert_start_loc + expert, 1) : -1;
            output_index[token * num_topk + k] = dst;
        }
        __syncthreads();
        if (dst >= 0) {
            for (int i = static_cast<int>(threadIdx.x); i < hidden_int4;
                 i += static_cast<int>(blockDim.x))
                output[dst * hidden_int4 + i] = recv_x[token * hidden_int4 + i];
        }
        __syncthreads();
    }
}

// Owned expert-major epilogue: IPC scratch is consumed before dispatch completion.
void deepgemm_permute_bf16(const void* recv_x, const int64_t* recv_topk_idx,
                           const int* num_recv_tokens_per_expert,
                           void* output, int* m_indices, int64_t* output_index,
                           int* expert_start_loc, int num_tokens, int hidden,
                           int num_topk, int num_experts, cudaStream_t stream) {
    EP_HOST_ASSERT(hidden * 2 % sizeof(int4) == 0);
    CUDA_CHECK(cudaMemsetAsync(output_index, 0xff,
                               static_cast<size_t>(num_tokens) * num_topk * sizeof(int64_t),
                               stream));
    deepgemm_init_layout<<<num_experts, 128, 0, stream>>>(
        num_recv_tokens_per_expert, expert_start_loc, m_indices);
    if (num_tokens > 0) {
        deepgemm_scatter_bf16<<<num_tokens, 256, 0, stream>>>(
            static_cast<const int4*>(recv_x), recv_topk_idx,
            static_cast<int4*>(output), output_index, expert_start_loc,
            hidden * 2 / sizeof(int4), num_topk);
    }
}

template<int kNumRanks>
__global__ void
cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int,
                       void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
    // A simplified version for cached handles
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    // Copy and clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = static_cast<int*>(buffer_ptrs[rank]);
    #pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
        ptr[i] = rank_prefix_matrix[i];
    #pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
        ptr[kNumRanks * kNumRanks + i] = 0;

    // Barrier after cleaning
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int,
                            void** buffer_ptrs, int** barrier_signal_ptrs,
                            int rank, int num_ranks, cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks) \
    LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, \
        rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 128, stream);
    SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp, bool kUseTMA = true, bool kLocalBypass = false, int kScanMode = 0, int kRouteMode = 0>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch(int4* recv_x, float* recv_x_scales, float* recv_x_sf_scale_for_nvfp4, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,
         int* send_head, const int4* x, const float* x_scales, const float* sf_scale_for_nvfp4, const int64_t* topk_idx, const float* topk_weights,
         const bool* is_token_in_rank, const int* channel_prefix_matrix,
         int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales, int num_sf_scales_for_nvfp4,
         int scale_token_stride, int scale_hidden_stride, int sf_scale_for_nvfp4_token_stride, int sf_scale_for_nvfp4_hidden_stride,
         void** buffer_ptrs, int rank,
         int num_max_send_tokens, int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    // Clear padding before sender/receiver branches can return early.
    // Valid payload rows are disjoint from this tail, including split streams.
    if (num_worst_tokens > 0 && kRouteMode != 1) {
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_sms * kNumThreads;
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;
    }

    // Several warps are response for a single rank
    int num_threads_per_rank = kNumThreads / kNumRanks;
    const auto num_channels = num_sms / 2;
    int responsible_rank = thread_id / num_threads_per_rank;
    int rank_thread_begin = responsible_rank * num_threads_per_rank;
    // The remote-only receiver has no self FIFO to drain. Reassign its
    // otherwise idle warp group evenly to the three actual remote sources.
    if constexpr (kRouteMode == 2 && kNumRanks == 4) {
        if (!is_sender) {
            num_threads_per_rank = kNumThreads / (kNumRanks - 1);
            const int source_group = thread_id / num_threads_per_rank;
            responsible_rank = source_group + (source_group >= rank);
            rank_thread_begin = source_group * num_threads_per_rank;
        }
    }
    __shared__ int sender_total_rows;
    __shared__ int sender_warp_counts[kNumRanks];
    __shared__ int sender_warp_offsets[kNumRanks];
    if constexpr (kScanMode >= 4) {
        if (is_sender) {
            // Compute the per-destination warp allocation once per sender
            // block. Repeating this prefix arithmetic in all 768 threads was
            // pure fixed overhead; every thread only needs the final mapping.
            if (thread_id == 0) {
                int total_rows = 0;
                for (int destination = 0; destination < kNumRanks; ++destination) {
                    const int* prefix = channel_prefix_matrix + destination * num_channels;
                    if ((kRouteMode != 1 || destination == rank) &&
                        (kRouteMode != 2 || destination != rank))
                        total_rows += prefix[sm_id / 2] - (sm_id / 2 ? prefix[sm_id / 2 - 1] : 0);
                }
                sender_total_rows = total_rows;
                for (int destination = 0; destination < kNumRanks; ++destination) {
                    sender_warp_offsets[destination] = destination;
                    sender_warp_counts[destination] = 1;
                }
                if (total_rows > 0) {
                // Split kernels allocate workers only to destinations they execute.
                constexpr int active_ranks = kRouteMode == 1 ? 1 :
                    kRouteMode == 2 ? kNumRanks - 1 : kNumRanks;
                constexpr int extra_warps = kNumThreads / 32 - active_ranks;
                int64_t remainders[kNumRanks];
                int remaining_warps = kNumThreads / 32;
                #pragma unroll
                for (int destination = 0; destination < kNumRanks; ++destination) {
                    const int* prefix = channel_prefix_matrix + destination * num_channels;
                    int rows = prefix[sm_id / 2] - (sm_id / 2 ? prefix[sm_id / 2 - 1] : 0);
                    if ((kRouteMode == 1 && destination != rank) ||
                        (kRouteMode == 2 && destination == rank)) rows = 0;
                    const int64_t scaled_rows = static_cast<int64_t>(rows) * extra_warps;
                    const bool active = (kRouteMode != 1 || destination == rank) &&
                        (kRouteMode != 2 || destination != rank);
                    sender_warp_counts[destination] = active + static_cast<int>(scaled_rows / total_rows);
                    remainders[destination] = scaled_rows % total_rows;
                    remaining_warps -= sender_warp_counts[destination];
                }
                // Largest remainder preserves equal shares for nearly equal
                // peers. Prefix rounding can otherwise turn 6/6/6/6 into
                // 5/6/6/7 and delay an almost equally loaded destination.
                #pragma unroll
                for (int destination = 0; destination < kNumRanks; ++destination) {
                    int priority = 0;
                    #pragma unroll
                    for (int other = 0; other < kNumRanks; ++other)
                        priority += remainders[other] > remainders[destination] ||
                            (remainders[other] == remainders[destination] && other < destination);
                    sender_warp_counts[destination] += priority < remaining_warps;
                }
                int first_warp = 0;
                for (int destination = 0; destination < kNumRanks; ++destination) {
                    sender_warp_offsets[destination] = first_warp;
                    first_warp += sender_warp_counts[destination];
                }
                }
            }
            __syncthreads();
            if (sender_total_rows > 0) {
                const int warp_id = thread_id / 32;
                #pragma unroll
                for (int destination = 0; destination < kNumRanks; ++destination) {
                    const int first_warp = sender_warp_offsets[destination];
                    const int last_warp = first_warp + sender_warp_counts[destination];
                    if (warp_id >= first_warp && warp_id < last_warp) {
                        responsible_rank = destination;
                        rank_thread_begin = first_warp * 32;
                        num_threads_per_rank = sender_warp_counts[destination] * 32;
                        break;
                    }
                }
            }
        }
    }
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    const auto responsible_channel = sm_id / 2;

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT((topk_idx == nullptr)  == (topk_weights == nullptr));
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    auto ring_slot = [&](int ordinal) {
        if constexpr (kScanMode >= 6) {
            return ordinal & (num_recv_buffer_tokens - 1);
        }
        return ordinal % num_recv_buffer_tokens;
    };

    // Calculate pointers by the specific layout
    // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) + kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? rank : responsible_rank;
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    // Channel buffer metadata
    // Senders are responsible for tails, and receivers are responsible for heads
    // Stored on the receiver side
    // The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
    // `start_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

    // Channel data buffers, stored on the receiver side
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // `topk_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(int64_t)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    // `x_scales_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales * sizeof(float)
    // `x_sf_scale_for_nvfp4_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_sf_scales_for_nvfp4 * sizeof(float)
    auto channel_x_buffers = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers = Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_idx_buffers = Buffer<int64_t>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_topk_weights_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_x_scales_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_scales, channel_rank_offset * num_recv_buffer_tokens * num_scales);
    auto channel_x_sf_scale_for_nvfp4_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_sf_scales_for_nvfp4, channel_rank_offset * num_recv_buffer_tokens * num_sf_scales_for_nvfp4);

        const int local_begin = rank > 0
            ? static_cast<int*>(buffer_ptrs[rank])[(rank - 1) * kNumRanks + rank]
            : 0;
        const bool local_destination = kLocalBypass && responsible_rank == rank;
        const int local_rank_offset = local_destination ? local_begin : 0;
        const int local_channel_offset = local_destination && responsible_channel > 0
            ? __ldg(channel_prefix_matrix + rank * num_channels + responsible_channel - 1)
            : 0;
        const int local_output_rows = local_destination
            ? static_cast<int*>(buffer_ptrs[rank])[(kNumRanks - 1) * kNumRanks + rank]
            : 0;

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    // Keep the existing per-warp shared-memory slice. A whole row avoids
    // a second TMA load/wait/store when it fits; larger rows retain two parts.
    const int tma_parts = kScanMode == 9 ? 1 : kScanMode >= 3 &&
        static_cast<int64_t>(hidden_int4) * sizeof(int4) + sizeof(uint64_t) <= kNumTMABytesPerWarp ? 1 : 2;
    const int half_hidden_int4 = hidden_int4 / tma_parts;
    const int half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    uint32_t tma_phase = 0;
    if constexpr (kUseTMA) {
        // Validate the complete warp slice before initializing its barrier.
        EP_DEVICE_ASSERT(hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
        // Local bypass still uses the same compact TMA copy when available;
        // it writes the final receive layout directly and does not touch the
        // remote queue. Initializing the per-warp barrier is therefore needed
        // for both local and remote payload copies.
        if (elect_one_sync()) {
            mbarrier_init(tma_mbarrier, 1);
            fence_barrier_init();
        }
        __syncwarp();
    }
#endif

    if (is_sender) {
        // Workers for sending
        constexpr int num_send_warps = kNumThreads / 32;
        const int num_send_warps_per_rank = num_threads_per_rank / 32;
        const auto send_thread_id = thread_id;
        const auto send_warp_id_in_rank = (send_thread_id - rank_thread_begin) / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);
        constexpr bool local_only = kRouteMode == 1;
        constexpr bool remote_only = kRouteMode == 2;
        const bool process_destination = (!local_only || local_destination) &&
            (!remote_only || !local_destination);
        if (!process_destination) return;
        const bool write_head = process_destination;
        const bool write_metadata = process_destination;
        // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (write_metadata && send_warp_id_in_rank == 0 and elect_one_sync()) {
            int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
            st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
            st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
        }
        __syncwarp();

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        // Local rows bypass the queue and are written directly to the final
        // receive layout. Publish their channel base directly so the paired
        // receiver block can return without polling an empty queue.
        if constexpr (kLocalBypass) {
            if (local_destination && kRouteMode == 1 && send_warp_id_in_rank == 0 && elect_one_sync()) {
                const int channel_begin = responsible_channel > 0
                    ? __ldg(channel_prefix_matrix + rank * num_channels + responsible_channel - 1)
                    : 0;
                st_relaxed_sys_global(
                    recv_channel_offset + responsible_rank * num_channels + responsible_channel,
                    channel_begin);
            }
        }

        auto copy_payload = [&](int64_t token_idx, int ordinal) {
            int dst_slot_idx = ring_slot(ordinal);
            const int local_output_row = local_rank_offset + local_channel_offset + ordinal;
            if (local_destination)
                EP_DEVICE_ASSERT(local_output_row >= 0 && local_output_row < local_output_rows);
            // Copy data
            auto shifted_channel_x_buffers = local_destination
                ? reinterpret_cast<int4*>(recv_x) +
                      static_cast<int64_t>(local_output_row) * hidden_int4
                : channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
            auto shifted_x = x + token_idx * hidden_int4;
#ifndef DISABLE_SM90_FEATURES
            if constexpr (kLocalBypass) {
                if constexpr (kScanMode == 9 && kUseTMA) {
                // This specialization is selected only when the full row fits
                // the existing warp slice; larger rows use the generic copy.
                tma_store_wait<0>();
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted_x, tma_mbarrier, half_hidden_bytes);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                    mbarrier_wait(tma_mbarrier, tma_phase);
                    tma_store_1d(tma_buffer, shifted_channel_x_buffers, half_hidden_bytes, false);
                }
                __syncwarp();
                } else
#endif
                {
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);
                }
            } else {
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
                                   shifted_channel_x_buffers, shifted_x,
                                   __ldg, st_na_global);
            }

            // Copy source index
            if (elect_one_sync())
                (local_destination ? recv_src_idx[local_output_row]
                                   : channel_src_idx_buffers[dst_slot_idx]) = static_cast<int>(token_idx);

            // Copy `topk_idx` and `topk_weights` with transformed index
            if (lane_id < num_topk) {
                // Top-k index
                int recv_expert_begin = responsible_rank * num_experts_per_rank, recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
                auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);
                idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
                (local_destination
                     ? recv_topk_idx[static_cast<int64_t>(local_output_row) * num_topk + lane_id]
                     : channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id]) = idx_value;

                // Top-k weights
                auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                (local_destination
                     ? recv_topk_weights[static_cast<int64_t>(local_output_row) * num_topk + lane_id]
                     : channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id]) = weight_value;
            }

            // Copy `x_scales`
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                (local_destination
                     ? recv_x_scales[static_cast<int64_t>(local_output_row) * num_scales + i]
                     : channel_x_scales_buffers[dst_slot_idx * num_scales + i]) = __ldg(x_scales + offset);
            }

            // Copy `sf_scale_for_nvfp4`
            #pragma unroll
            for (int i = lane_id; i < num_sf_scales_for_nvfp4; i += 32) {
                auto offset = token_idx * sf_scale_for_nvfp4_token_stride + i * sf_scale_for_nvfp4_hidden_stride;
                (local_destination
                     ? recv_x_sf_scale_for_nvfp4[static_cast<int64_t>(local_output_row) * num_sf_scales_for_nvfp4 + i]
                     : channel_x_sf_scale_for_nvfp4_buffers[dst_slot_idx * num_sf_scales_for_nvfp4 + i]) = __ldg(sf_scale_for_nvfp4 + offset);
            }

        };

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;
        int tile_base = token_start_idx, tile_end = token_start_idx;
        unsigned selected_mask = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx; ) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            // NOTES: the head index received by different warps may not be the same
            auto start_time = clock64();
            if (process_destination && !local_destination && elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                if constexpr (kScanMode >= 2) {
                    // All warps agree on chunk/tile boundaries, but each warp
                    // enumerates only its own payload rows within a tile.
                    if (token_idx >= tile_end) {
                        tile_base = static_cast<int>(token_idx);
                        tile_end = min(tile_base + 32, token_end_idx);
                        bool selected = tile_base + lane_id < tile_end &&
                            is_token_in_rank[(tile_base + lane_id) * kNumRanks + responsible_rank];
                        selected_mask = __ballot_sync(0xffffffff, selected);
                        if (send_warp_id_in_rank == 0 && tile_base + lane_id < tile_end) {
                            int prefix = __popc(selected_mask & ((1u << lane_id) - 1));
                            if (write_head)
                                send_head[(tile_base + lane_id) * kNumRanks + responsible_rank] =
                                    selected ? cached_channel_tail_idx + prefix : -1;
                        }
                    }
                    int available = __popc(selected_mask);
                    int take = min(available, num_max_send_tokens - chunk_token_idx);
                    unsigned before_lane = (1u << lane_id) - 1;
                    int prefix = __popc(selected_mask & before_lane);
                    bool in_chunk = (selected_mask & (1u << lane_id)) && prefix < take;
                    unsigned chunk_mask = __ballot_sync(0xffffffff, in_chunk);
                    unsigned owned_mask = __ballot_sync(0xffffffff, in_chunk &&
                        (cached_channel_tail_idx + prefix + 1) % num_send_warps_per_rank == send_warp_id_in_rank);
                    while (owned_mask) {
                        int selected_lane = __ffs(owned_mask) - 1;
                        int ordinal = cached_channel_tail_idx +
                            __popc(chunk_mask & ((1u << selected_lane) - 1));
                        copy_payload(tile_base + selected_lane, ordinal);
                        owned_mask &= owned_mask - 1;
                    }
                    cached_channel_tail_idx += take;
                    chunk_token_idx += take;
                    selected_mask &= ~chunk_mask;
                    token_idx = selected_mask ? tile_base + __ffs(selected_mask) - 1 : tile_end;
                    continue;
                }

                    if (write_head && token_idx % num_send_warps_per_rank == send_warp_id_in_rank && elect_one_sync())
                        send_head[token_idx * kNumRanks + responsible_rank] = is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;
                    if (!is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                        token_idx++;
                        continue;
                    }

                const int ordinal = cached_channel_tail_idx++;
                if (process_destination && cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank)
                    copy_payload(token_idx, ordinal);

                // Move token index
                chunk_token_idx ++, token_idx ++;
            }

            // Complete asynchronous payload stores before publishing the
            // channel tail. A read-only wait would allow premature publication.
#ifndef DISABLE_SM90_FEATURES
            if constexpr (kScanMode == 9 && kUseTMA) {
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                __syncwarp();
            }
#endif
            // Move tail index
            // NOTES: here all warps should share the same new tail
            asm volatile("bar.sync %0, %1;" :: "r"(responsible_rank), "r"(num_threads_per_rank));
            if (process_destination && !local_destination) {
                if (send_warp_id_in_rank == 0 and elect_one_sync())
                    st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
            }
        }
    } else {
        if constexpr (kRouteMode == 1)
            return;
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps = kNumThreads / 32;
        const auto recv_thread_id = thread_id;
        const auto recv_thread_id_in_rank = recv_thread_id - rank_thread_begin;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        constexpr int num_recv_warps_per_rank = kNumThreads / 32 /
            (kRouteMode == 2 && kNumRanks == 4 ? kNumRanks - 1 : kNumRanks);
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;
        if constexpr (kRouteMode == 2) {
            if constexpr (kLocalBypass) {
                if (responsible_rank == rank)
                    return;
            }
        }
        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0);
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0);
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            if (recv_warp_id_in_rank == 0)
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Self-destination rows were written directly by the sender warps into
        // the final receive tensors, so this receiver group has no queue to drain.
        if constexpr (kLocalBypass) {
            if (responsible_rank == rank)
                return;
        }

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are the same
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());

                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n", rank, responsible_channel, num_tokens_to_recv);
                    trap();
                }
            }

            // Synchronize queue tail
            asm volatile("bar.sync %0, %1;" :: "r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = ring_slot(cached_channel_head_idx + chunk_idx);
                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
                auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                if constexpr (kUseTMA) {
                    #pragma unroll
                    for (int i = 0; i < tma_parts; ++ i) {
                        tma_store_wait<0>();
                        if (elect_one_sync()) {
                            tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4, tma_mbarrier, half_hidden_bytes);
                            mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                            mbarrier_wait(tma_mbarrier, tma_phase);
                            tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4, half_hidden_bytes, false);
                        }
                    }
                    __syncwarp();
                } else
#endif
                {
                    // Valid int4 rows that cannot fit the TMA slice use the existing warp copy.
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4,
                                       ld_nc_global, st_na_global);
                }
            }

            // Copy `src_idx`
            #pragma unroll 4
            for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx; chunk_idx += 32 * num_recv_warps_per_rank)
                recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] = ld_nc_global(channel_src_idx_buffers.buffer() + ring_slot(chunk_idx));

            // Copy `topk_idx` and `topk_weights`
            #pragma unroll 4
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
                int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
                int token_idx_in_buffer = ring_slot(cached_channel_head_idx + chunk_idx);
                auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
                auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
                recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
            }

            // Copy `x_scales`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_scales, scales_idx = i % num_scales;
                int token_idx_in_buffer = ring_slot(cached_channel_head_idx + chunk_idx);
                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);
            }

            // Copy `sf_scale_for_nvfp4`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_sf_scales_for_nvfp4; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_sf_scales_for_nvfp4, sf_scales_idx = i % num_sf_scales_for_nvfp4;
                int token_idx_in_buffer = ring_slot(cached_channel_head_idx + chunk_idx);
                recv_x_sf_scale_for_nvfp4[static_cast<int64_t>(total_offset + chunk_idx) * num_sf_scales_for_nvfp4 + sf_scales_idx] =
                    ld_nc_global(channel_x_sf_scale_for_nvfp4_buffers.buffer() + token_idx_in_buffer * num_sf_scales_for_nvfp4 + sf_scales_idx);
            }

            // Move queue
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
            asm volatile("bar.sync %0, %1;" :: "r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            // Exit
            num_tokens_to_recv -= num_recv_tokens;
        }
    }

}

void dispatch(void* recv_x, float* recv_x_scales, float* recv_x_sf_scale_for_nvfp4, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,
              int* send_head, const void* x, const float* x_scales, const float* sf_scale_for_nvfp4, const int64_t* topk_idx, const float* topk_weights,
              const bool* is_token_in_rank, const int* channel_prefix_matrix,
              int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales, int num_sf_scales_for_nvfp4,
              int scale_token_stride, int scale_hidden_stride, int sf_scale_for_nvfp4_token_stride, int sf_scale_for_nvfp4_hidden_stride,
              void** buffer_ptrs, int rank, int num_ranks,
              cudaStream_t stream, int num_sms, int num_max_send_tokens, int num_recv_buffer_tokens,
              cudaStream_t local_stream) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
    const bool use_tma = hidden_int4 % 2 == 0 &&
        static_cast<int64_t>(hidden_int4 / 2) * sizeof(int4) + sizeof(uint64_t) <= kNumTMABytesPerWarp;
    const char* fifo_env = std::getenv("DEEPEP_FIFO_OPTIMIZED");
    const bool fifo_optimized = fifo_env && fifo_env[0] == '1' && fifo_env[1] == '\0' &&
        num_ranks == 4;
    // Keep the production interface to one optimized-path switch. Compile-time
    // specializations below are implementation details selected by shape.
    const bool local_bypass = fifo_optimized &&
        env_flag_default("DEEPEP_LOCAL_BYPASS", true);
    const bool compact_tma_shape = use_tma &&
        static_cast<int64_t>(hidden_int4) * sizeof(int4) + sizeof(uint64_t) <= kNumTMABytesPerWarp;
    const int scan_mode = fifo_optimized ? (compact_tma_shape ? 9 : 6) : 0;

#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(static_cast<int64_t>(num_sf_scales_for_nvfp4) * sf_scale_for_nvfp4_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks) { \
    if (local_stream != nullptr && local_bypass) { \
    auto local_kernel = scan_mode == 9 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 9, 1> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 9, 1>) : scan_mode == 6 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 6, 1> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 6, 1>) : (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 0, 1> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 0, 1>); \
    cfg.stream = local_stream; \
    SET_SHARED_MEMORY_FOR_TMA(local_kernel); \
    LAUNCH_KERNEL(&cfg, local_kernel, \
        reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_x_sf_scale_for_nvfp4, recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_offset, \
        send_head, reinterpret_cast<const int4*>(x), x_scales, sf_scale_for_nvfp4, topk_idx, topk_weights, \
        is_token_in_rank, channel_prefix_matrix, \
        num_tokens, num_worst_tokens, hidden_int4, num_topk, num_experts, num_scales, num_sf_scales_for_nvfp4, \
        scale_token_stride, scale_hidden_stride, sf_scale_for_nvfp4_token_stride, sf_scale_for_nvfp4_hidden_stride, \
        buffer_ptrs, rank, num_max_send_tokens, num_recv_buffer_tokens); \
    auto remote_kernel = scan_mode == 9 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 9, 2> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 9, 2>) : scan_mode == 6 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 6, 2> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 6, 2>) : (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 0, 2> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 0, 2>); \
    cfg.stream = stream; \
    SET_SHARED_MEMORY_FOR_TMA(remote_kernel); \
    LAUNCH_KERNEL(&cfg, remote_kernel, \
        reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_x_sf_scale_for_nvfp4, recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_offset, \
        send_head, reinterpret_cast<const int4*>(x), x_scales, sf_scale_for_nvfp4, topk_idx, topk_weights, \
        is_token_in_rank, channel_prefix_matrix, \
        num_tokens, num_worst_tokens, hidden_int4, num_topk, num_experts, num_scales, num_sf_scales_for_nvfp4, \
        scale_token_stride, scale_hidden_stride, sf_scale_for_nvfp4_token_stride, sf_scale_for_nvfp4_hidden_stride, \
        buffer_ptrs, rank, num_max_send_tokens, num_recv_buffer_tokens); \
    } else { \
    auto kernel = local_bypass \
        ? (scan_mode == 9 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 9> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 9>) \
           : scan_mode == 6 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true, 6> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true, 6>) \
           : (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, true> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, true>)) \
        : (scan_mode == 9 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, false, 9> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, false, 9>) \
           : scan_mode == 6 ? (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, false, 6> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, false, 6>) \
           : (use_tma ? dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, true, false> : dispatch<ranks, kNumThreads, kNumTMABytesPerWarp, false, false>)); \
    SET_SHARED_MEMORY_FOR_TMA(kernel); \
    LAUNCH_KERNEL(&cfg, kernel, \
        reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_x_sf_scale_for_nvfp4, recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_offset, \
        send_head, reinterpret_cast<const int4*>(x), x_scales, sf_scale_for_nvfp4, topk_idx, topk_weights, \
        is_token_in_rank, channel_prefix_matrix, \
        num_tokens, num_worst_tokens, hidden_int4, num_topk, num_experts, num_scales, num_sf_scales_for_nvfp4, \
        scale_token_stride, scale_hidden_stride, sf_scale_for_nvfp4_token_stride, sf_scale_for_nvfp4_hidden_stride, \
        buffer_ptrs, rank, \
        num_max_send_tokens, num_recv_buffer_tokens); \
    } \
    } break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
#ifndef DISABLE_SM90_FEATURES
    if (local_stream != nullptr) attr[0].val.cooperative = 0;
#endif
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

// A control-only collective: no payload allocation and no queue polling.
template<int kNumRanks, bool kParallelHeads = false>
__global__ void
cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens, int num_memset_int,
                      int** barrier_signal_ptrs, int rank) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    if (sm_id == 0) {
        // Barrier before cleaning
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        // Clean
        auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
        auto ptr = static_cast<int*>(buffer_ptrs[rank]);
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            ptr[i] = 0;

        // Barrier after cleaning
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        const auto channel_id = sm_id - 1;
        const auto thread_id = static_cast<int>(threadIdx.x);
        const auto rank_id = thread_id / 32;
        const auto lane_id = thread_id % 32;
        if (rank_id >= kNumRanks)
            return;

        int token_start_idx, token_end_idx;
        get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // NOTES: `1 << 25` is a heuristic large number
        int last_head = 1 << 25;
        #pragma unroll
        for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
            int token_idx = token_idx_tail - lane_id, expected_head = 0;
            auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
            if constexpr (kParallelHeads) {
                // Reverse token order: the closest valid lane at or below the
                // current lane is the next routed token. A ballot replaces the
                // five-step inclusive shuffle scan while preserving carry.
                const unsigned valid_heads = __ballot_sync(0xffffffff, current_head >= 0);
                const unsigned through_lane = valid_heads & (0xffffffffu >> (31 - lane_id));
                int previous_lane = -1;
                if (through_lane) previous_lane = 31 - static_cast<int>(__clz(through_lane));
                const int next_head = __shfl_sync(0xffffffff, current_head, max(0, previous_lane));
                expected_head = -(previous_lane >= 0 ? next_head : last_head) - 1;
                int carry_lane = -1;
                if (valid_heads) carry_lane = 31 - static_cast<int>(__clz(valid_heads));
                const int carry_head = __shfl_sync(0xffffffff, current_head, max(0, carry_lane));
                if (carry_lane >= 0) last_head = carry_head;
            } else for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++ i) {
                const int head = __shfl_sync(0xffffffff, current_head, i);
                if (head < 0) {
                    if (lane_id == i)
                        expected_head = -last_head - 1;
                } else {
                    last_head = head;
                }
            }
            if (current_head < 0 and token_idx >= token_start_idx)
                send_head[token_idx * kNumRanks + rank_id] = expected_head;
        }
    }
}

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels,
                           int num_recv_tokens, int num_memset_int,
                           int** barrier_signal_ptrs, int rank, int num_ranks,
                           cudaStream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks) { \
    auto kernel = parallel_heads ? cached_notify_combine<ranks, true> : cached_notify_combine<ranks, false>; \
    LAUNCH_KERNEL(&cfg, kernel, \
        buffer_ptrs, send_head, num_channels, num_recv_tokens, num_memset_int, barrier_signal_ptrs, rank); \
    } break

    const char* fifo_setting = std::getenv("DEEPEP_FIFO_OPTIMIZED");
    const bool parallel_heads = fifo_setting && fifo_setting[0] == '1' && fifo_setting[1] == '\0' &&
        num_ranks == 4 && num_channels == 10;
    const int num_threads = std::max(128, 32 * num_ranks);
    EP_HOST_ASSERT(num_ranks <= num_threads);
    EP_HOST_ASSERT(num_threads <= 1024);
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
    SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
    SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template<typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp,
         bool kGlobalStore = false,
         bool kBalancedSender = false, bool kLocalBypass = false,
         bool kSingleSourceFast = false, bool kNoBiasFast = false, int kRouteMode = 0>
__global__ void __launch_bounds__(kNumThreads, 1)
combine(dtype_t* recv_x, float* recv_topk_weights,
        const dtype_t* x, const float* topk_weights,
        const dtype_t* bias_0, const dtype_t* bias_1,
        const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix,
        int* send_head, int num_tokens, int num_recv_tokens, int hidden, int num_topk,
        void** buffer_ptrs, int rank,
        int num_max_send_tokens, int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x) * (kRouteMode ? 2 : 1);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto sm_id = static_cast<int>(blockIdx.x) * (kRouteMode ? 2 : 1) + (kRouteMode == 2), lane_id = get_lane_id();
    const auto num_channels = num_sms / 2;
    const bool is_sender = sm_id % 2 == 0;
    if constexpr (kRouteMode == 1) { if (!is_sender) return; }
    if constexpr (kRouteMode == 2) { if (is_sender) return; }
    const int responsible_channel = sm_id / 2;
    const bool recv_buffer_is_power_of_two =
        (num_recv_buffer_tokens & (num_recv_buffer_tokens - 1)) == 0;
    auto ring_slot = [&](int ordinal) {
        if constexpr (kGlobalStore || kLocalBypass)
            return ordinal & (num_recv_buffer_tokens - 1);
        return recv_buffer_is_power_of_two
            ? ordinal & (num_recv_buffer_tokens - 1)
            : ordinal % num_recv_buffer_tokens;
    };
    EP_DEVICE_ASSERT(num_topk <= 32);

    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
    int hidden_int4_aligned = align_down(hidden_int4, 32);
    auto x_int4 = reinterpret_cast<const int4*>(x);
    auto bias_0_int4 = reinterpret_cast<const int4*>(bias_0);
    auto bias_1_int4 = reinterpret_cast<const int4*>(bias_1);
    auto recv_int4 = reinterpret_cast<int4*>(recv_x);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
#endif

    if (is_sender) {
        // Workers for sending
        // Several warps are responsible for a single rank
        int num_send_warps_per_rank = (kNumThreads / 32) / kNumRanks;
        constexpr int num_send_warps = kNumThreads / 32;
        int num_threads_per_rank = num_send_warps_per_rank * 32;
        const auto send_thread_id = thread_id;
        const auto send_warp_id = send_thread_id / 32;
        int send_rank_id = (responsible_channel + send_warp_id) % kNumRanks;
        int send_warp_id_in_rank = send_warp_id / kNumRanks;
        if constexpr (kRouteMode == 1 && kNumRanks == 4) {
            // Spend all sender warps on the three remote peers.
            num_send_warps_per_rank = num_send_warps / (kNumRanks - 1);
            num_threads_per_rank = num_send_warps_per_rank * 32;
            const int peer = (responsible_channel + send_warp_id) % (kNumRanks - 1);
            send_rank_id = peer + (peer >= rank);
            send_warp_id_in_rank = send_warp_id / (kNumRanks - 1);
        }
        EP_STATIC_ASSERT(num_send_warps * 32 == kNumThreads, "Invalid warp count");

        if constexpr (kBalancedSender) {
            int rows[kNumRanks];
            int total_rows = 0;
            #pragma unroll
            for (int peer = 0; peer < kNumRanks; ++peer) {
                const int begin = peer ? rank_prefix_matrix[(peer - 1) * kNumRanks + rank] : 0;
                const int count = rank_prefix_matrix[peer * kNumRanks + rank] - begin;
                const int channel_begin = channel_prefix_matrix[peer * num_channels + responsible_channel];
                const int channel_end = responsible_channel + 1 == num_channels ? count :
                    channel_prefix_matrix[peer * num_channels + responsible_channel + 1];
                rows[peer] = channel_end - channel_begin;
                total_rows += rows[peer];
            }
            if (total_rows > 0) {
                constexpr int extra = num_send_warps - kNumRanks;
                int warps[kNumRanks];
                int64_t remainders[kNumRanks];
                int remaining = num_send_warps;
                #pragma unroll
                for (int peer = 0; peer < kNumRanks; ++peer) {
                    const int64_t scaled = static_cast<int64_t>(rows[peer]) * extra;
                    warps[peer] = 1 + static_cast<int>(scaled / total_rows);
                    remainders[peer] = scaled % total_rows;
                    remaining -= warps[peer];
                }
                #pragma unroll
                for (int peer = 0; peer < kNumRanks; ++peer) {
                    int priority = 0;
                    #pragma unroll
                    for (int other = 0; other < kNumRanks; ++other)
                        priority += remainders[other] > remainders[peer] ||
                            (remainders[other] == remainders[peer] && other < peer);
                    warps[peer] += priority < remaining;
                }
                int first = 0;
                #pragma unroll
                for (int peer = 0; peer < kNumRanks; ++peer) {
                    if (send_warp_id >= first && send_warp_id < first + warps[peer]) {
                        send_rank_id = peer;
                        send_warp_id_in_rank = send_warp_id - first;
                        num_send_warps_per_rank = warps[peer];
                        num_threads_per_rank = warps[peer] * 32;
                    }
                    first += warps[peer];
                }
            }
        }
        const bool local_destination = kLocalBypass && send_rank_id == rank;
        // Local rows are read directly by the receiver during reduction. The
        // local sender group has no queue, payload, or tail update to perform.
        if (local_destination)
            return;

        // Calculate pointers by the specific layout
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]));
        auto num_channels_total = num_channels * kNumRanks;
        auto channel_rank_offset = responsible_channel * kNumRanks + rank;

        // Channel meta data
        // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_x_buffers = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        auto channel_src_idx_buffers = Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
        auto channel_topk_weights_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);

        // Get tasks
        // NOTES: `channel_offset` is already shifted
        int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
        int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
        int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
        int num_channel_tokens = (responsible_channel == num_channels - 1 ? num_rank_tokens : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) - channel_offset;
        int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;

        // Iterate over all tokens and send by chunks
        int current_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx; ) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            auto start_time = clock64();
            int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
            if (!local_destination) {
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();
            }

            // Send by chunk
            #pragma unroll
            for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
                // Get an empty slot
                int dst_slot_idx = ring_slot(current_channel_tail_idx + i);

                // Copy data
                if (!local_destination) {
                    auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;
                    UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);
                }

                // Send source index
                if (!local_destination && elect_one_sync())
                    channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);

                // Send `topk_weights`
                if (!local_destination && num_topk > 0 and lane_id < num_topk)
                    channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = __ldg(topk_weights + (token_idx + i) * num_topk + lane_id);
            }
            token_idx += num_round_tokens;
            current_channel_tail_idx += num_round_tokens;

            // Move tail index
            asm volatile("bar.sync %0, %1;" :: "r"(send_rank_id), "r"(num_threads_per_rank));
            if (!local_destination && send_warp_id_in_rank == 0 && elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);
        }
    } else {
        // Workers for receiving
        // One warp for moving the queue head, others for reduction
        constexpr int num_recv_warps = kNumThreads / 32;
        const auto recv_warp_id = thread_id / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32 and kNumThreads > 32);
        EP_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % 32 == 0);

        // Shared head, tail and retired flags for receiver warps
        __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
        __shared__ volatile int channel_tail_idx[kNumRanks];
        __shared__ volatile bool warp_retired[num_recv_warps];
        if (thread_id < num_recv_warps)
            warp_retired[thread_id] = false;
        if (lane_id < kNumRanks)
            warp_channel_head_idx[recv_warp_id][lane_id] = 0;
        if (thread_id < kNumRanks)
            channel_tail_idx[thread_id] = 0;
        asm volatile("bar.sync 0, %0;" :: "r"(kNumThreads));

        if (thread_id < 32) {
            int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + lane_id;
            int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

            // Queue head updater
            int last_head = 0;
            while (lane_id < kNumRanks && !(kLocalBypass && lane_id == rank)) {
                // Check retired
                bool retired = true;
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++ i)
                    retired = retired and warp_retired[i];
                if (retired)
                    break;

                // Update queue tail
                channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

                // Update minimum head
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++ i) if (not warp_retired[i])
                    min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
                if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
                    st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
            }
        } else {
            // Receivers
            // Channel metadata
            // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
            Buffer<int4> channel_x_buffers[kNumRanks];
            Buffer<float> channel_topk_weights_buffers[kNumRanks];

            // Calculate pointers by the specific layout
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++ i) {
                auto channel_rank_offset = responsible_channel * kNumRanks + i;
                auto num_channels_total = num_channels * kNumRanks;
                // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
                auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * sizeof(int));

                // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
                channel_x_buffers[i] = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

                // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
                ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channels_total * num_recv_buffer_tokens * sizeof(int));

                // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
                channel_topk_weights_buffers[i] = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
            }

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);
            int local_row_base = 0;
            if constexpr (kLocalBypass) {
                const int local_rank_offset = rank > 0
                    ? __ldg(rank_prefix_matrix + (rank - 1) * kNumRanks + rank)
                    : 0;
                const int local_channel_offset = __ldg(
                    channel_prefix_matrix + rank * num_channels + responsible_channel);
                local_row_base = local_rank_offset + local_channel_offset;
            }

            // Iterate over all tokens and combine
            for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx; token_idx += num_recv_warps - 1) {
                // Read expected head
                int expected_head = -1;
                if (lane_id < kNumRanks)
                    expected_head = __ldg(send_head + token_idx * kNumRanks + lane_id);

                auto start_time = clock64();
                while (__any_sync(0xffffffff, expected_head >= 0 and
                                  !(kLocalBypass && lane_id == rank) &&
                                  channel_tail_idx[lane_id] <= expected_head)) {
                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n", rank, responsible_channel, expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Broadcast current heads
                int num_topk_ranks = 0, topk_ranks[kNumRanks];
                int slot_indices[kNumRanks], local_row_indices[kNumRanks];
                #pragma unroll
                for (int i = 0; i < kNumRanks; ++ i) {
                    auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);
                    if (expected_head_i >= 0) {
                        if constexpr (kLocalBypass) {
                            if (i == rank) {
                                local_row_indices[num_topk_ranks] = local_row_base + expected_head_i;
                                slot_indices[num_topk_ranks] = 0;
                            } else {
                                local_row_indices[num_topk_ranks] = 0;
                                slot_indices[num_topk_ranks] = ring_slot(expected_head_i);
                            }
                        } else {
                            slot_indices[num_topk_ranks] = ring_slot(expected_head_i);
                            local_row_indices[num_topk_ranks] = 0;
                        }
                        topk_ranks[num_topk_ranks ++] = i;
                    }
                }

                // A single contribution needs no general multi-rank reduction.
                if constexpr (kSingleSourceFast) {
                    if (num_topk_ranks == 1) {
                        const int source_rank = topk_ranks[0];
                        const bool is_local = source_rank == rank;
                        const int source_row = local_row_indices[0];
                        for (int i = lane_id; i < hidden_int4; i += 32) {
                            const int4 input_value = is_local
                                ? __ldg(x_int4 + static_cast<int64_t>(source_row) * hidden_int4 + i)
                                : ld_nc_global(channel_x_buffers[source_rank].buffer() + slot_indices[0] * hidden_int4 + i);
                            if constexpr (kNoBiasFast) {
                                recv_int4[token_idx * hidden_int4 + i] = input_value;
                                continue;
                            }
                            const int4 bias_0_value = bias_0_int4 != nullptr
                                ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                            const int4 bias_1_value = bias_1_int4 != nullptr
                                ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                            auto input_dtypes = reinterpret_cast<const dtype_t*>(&input_value);
                            auto bias_0_dtypes = reinterpret_cast<const dtype_t*>(&bias_0_value);
                            auto bias_1_dtypes = reinterpret_cast<const dtype_t*>(&bias_1_value);
                            int4 output_value;
                            auto output_dtypes = reinterpret_cast<dtype_t*>(&output_value);
                            #pragma unroll
                            for (int j = 0; j < kDtypePerInt4; ++j)
                                output_dtypes[j] = static_cast<dtype_t>(
                                    static_cast<float>(bias_0_dtypes[j]) +
                                    static_cast<float>(bias_1_dtypes[j]) +
                                    static_cast<float>(input_dtypes[j]));
                            recv_int4[token_idx * hidden_int4 + i] = output_value;
                        }
                        if (lane_id < num_topk)
                            recv_topk_weights[token_idx * num_topk + lane_id] =
                                is_local
                                    ? __ldg(topk_weights + static_cast<int64_t>(source_row) * num_topk + lane_id)
                                    : ld_nc_global(channel_topk_weights_buffers[source_rank].buffer() + slot_indices[0] * num_topk + lane_id);
                        if (lane_id < kNumRanks)
                            warp_channel_head_idx[recv_warp_id][lane_id] =
                                (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
                        continue;
                    }
                }

                // Wait shared memory release
#ifndef DISABLE_SM90_FEATURES
                if constexpr (!kGlobalStore) {
                    tma_store_wait<0>();
                    __syncwarp();
                }
#endif

                // Reduce data with pipeline
                constexpr int kNumStages = 8;
                EP_STATIC_ASSERT(kNumStages * 32 * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");
                #pragma unroll
                for (int i = lane_id; i < hidden_int4; i += 32) {
                    // Read buffers
                    int4 recv_value_int4[kNumRanks];
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++ j) {
                        if constexpr (kLocalBypass) {
                            recv_value_int4[j] = topk_ranks[j] == rank
                                ? __ldg(x_int4 + static_cast<int64_t>(local_row_indices[j]) * hidden_int4 + i)
                                : ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);
                        } else {
                            recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);
                        }
                    }

                    float values[kDtypePerInt4];
                    if constexpr (kNoBiasFast) {
                        #pragma unroll
                        for (int j = 0; j < kDtypePerInt4; ++j)
                            values[j] = 0.0f;
                    } else {
                        // Read and reduce bias only when the caller supplied it.
                        int4 bias_0_value_int4 = bias_0_int4 != nullptr
                            ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                        int4 bias_1_value_int4 = bias_1_int4 != nullptr
                            ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                        auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                        auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                        #pragma unroll
                        for (int j = 0; j < kDtypePerInt4; ++j)
                            values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
                    }

                    // Reduce all-to-all results
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++ j) {
                        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                        #pragma unroll
                        for (int k = 0; k < kDtypePerInt4; ++ k)
                            values[k] += static_cast<float>(recv_value_dtypes[k]);
                    }

                    // Cast back to `dtype_t`
                    int4 out_int4;
                    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++ j)
                        out_dtypes[j] = static_cast<dtype_t>(values[j]);

#ifndef DISABLE_SM90_FEATURES
                    if constexpr (kGlobalStore) {
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;
                    } else if (i < hidden_int4_aligned) {
                        // Wait TMA arrival
                        tma_store_wait<kNumStages - 1>();
                        __syncwarp();

                        // Write into TMA buffer
                        auto tma_stage_idx = (i / 32) % kNumStages;
                        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;

                        // Issue TMA
                        tma_store_fence();
                        __syncwarp();
                        if (elect_one_sync()) {
                            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));
                            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                                        recv_int4 + token_idx * hidden_int4 + i, tma_bytes, false);
                        }
                        __syncwarp();
                    } else {
#endif
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;
#ifndef DISABLE_SM90_FEATURES
                    }
#endif
                }

                // Reduce `topk_weights`
                if (lane_id < num_topk) {
                    float value = 0;
                    #pragma unroll
                    for (int i = 0; i < num_topk_ranks; ++ i) {
                        if constexpr (kLocalBypass) {
                            value += topk_ranks[i] == rank
                                ? __ldg(topk_weights + static_cast<int64_t>(local_row_indices[i]) * num_topk + lane_id)
                                : ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                        } else {
                            value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                        }
                    }
                    recv_topk_weights[token_idx * num_topk + lane_id] = value;
                }

                // Update head
                if (lane_id < kNumRanks)
                    warp_channel_head_idx[recv_warp_id][lane_id] = (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                warp_retired[recv_warp_id] = true;
        }
    }
}

void combine(cudaDataType_t type,
             void* recv_x, float* recv_topk_weights,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix,
             int* send_head, int num_tokens, int num_recv_tokens, int hidden, int num_topk,
             void** buffer_ptrs, int rank, int num_ranks,
             cudaStream_t stream, int num_sms,
             int num_max_send_tokens, int num_recv_buffer_tokens,
             cudaStream_t local_stream) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 4096;
    const char* fifo_setting = std::getenv("DEEPEP_FIFO_OPTIMIZED");
    const bool fifo_optimized = fifo_setting && fifo_setting[0] == '1' && fifo_setting[1] == '\0' &&
        num_ranks == 4;
    const bool global_store = (num_ranks == 2 || num_ranks == 4) &&
        fifo_optimized;
    const bool balanced_sender = fifo_optimized &&
        env_flag_default("DEEPEP_COMBINE_BALANCED_SEND", false);
    const bool local_bypass = num_ranks == 4 && fifo_optimized &&
        env_flag_default("DEEPEP_COMBINE_LOCAL_BYPASS", true);
    const bool no_bias_fast = local_bypass && bias_0 == nullptr && bias_1 == nullptr;
    // Increase receiver latency hiding within the same ten-CTA grid for the
    // measured QMSum shape. Other shapes/configurations keep their old default.
    // The queue must also cover the lag between 31 interleaved consumers;
    // otherwise a sender waiting for a whole chunk can block their progress.
    const bool wide_recv_safe = num_recv_buffer_tokens - num_max_send_tokens >= 1024 / 32 - 2;
    const int recv_threads = wide_recv_safe && env_flag_default("DEEPEP_COMBINE_RECV_1024",
        no_bias_fast && hidden == 3072 && num_sms == 20) ? 1024 : kNumThreads;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

#define COMBINE_LAUNCH_CASE(dtype, ranks) { \
    auto kernel = global_store ? combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, true> : combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, false>; \
    if constexpr (ranks == 4) { \
        if (balanced_sender) { \
            kernel = global_store ? combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, true, true> : combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, false, true>; \
        } \
        if (local_bypass && !balanced_sender) { \
            if (no_bias_fast) kernel = global_store ? combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, true, false, true, true, true> : \
                combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, false, false, true, true, true>; \
            else kernel = global_store ? combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, true, false, true, true> : \
                combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, false, false, true, true>; \
        } \
    } \
    if (local_stream != nullptr && local_bypass) { \
        auto local_kernel = no_bias_fast ? combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, true, false, true, true, true, 2> : \
            combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, true, false, true, true, false, 2>; \
        if (recv_threads == 1024) local_kernel = no_bias_fast ? \
            combine<dtype, ranks, 1024, kNumTMABytesPerWarp, true, false, true, true, true, 2> : \
            combine<dtype, ranks, 1024, kNumTMABytesPerWarp, true, false, true, true, false, 2>; \
        /* Load both kernels before the sender can block on FIFO progress. */ \
        SET_SHARED_MEMORY_FOR_TMA(local_kernel); \
        kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, true, false, true, false, false, 1>; \
    SET_SHARED_MEMORY_FOR_TMA(kernel); \
    cfg.dynamicSmemBytes = 0; \
    LAUNCH_KERNEL(&cfg, kernel, \
        reinterpret_cast<dtype*>(recv_x), recv_topk_weights, \
        reinterpret_cast<const dtype*>(x), topk_weights,   \
        reinterpret_cast<const dtype*>(bias_0), reinterpret_cast<const dtype*>(bias_1), \
        src_idx, rank_prefix_matrix, channel_prefix_matrix, \
        send_head, num_tokens, num_recv_tokens, hidden, num_topk, \
        buffer_ptrs, rank, \
        num_max_send_tokens, num_recv_buffer_tokens); \
        cfg.stream = local_stream; \
        kernel = local_kernel; \
        cfg.blockDim.x = recv_threads; \
    LAUNCH_KERNEL(&cfg, kernel, \
        reinterpret_cast<dtype*>(recv_x), recv_topk_weights, \
        reinterpret_cast<const dtype*>(x), topk_weights,   \
        reinterpret_cast<const dtype*>(bias_0), reinterpret_cast<const dtype*>(bias_1), \
        src_idx, rank_prefix_matrix, channel_prefix_matrix, \
        send_head, num_tokens, num_recv_tokens, hidden, num_topk, \
        buffer_ptrs, rank, \
        num_max_send_tokens, num_recv_buffer_tokens); \
        cfg.stream = stream; \
    } else { \
    SET_SHARED_MEMORY_FOR_TMA(kernel); \
    LAUNCH_KERNEL(&cfg, kernel, \
        reinterpret_cast<dtype*>(recv_x), recv_topk_weights, \
        reinterpret_cast<const dtype*>(x), topk_weights,   \
        reinterpret_cast<const dtype*>(bias_0), reinterpret_cast<const dtype*>(bias_1), \
        src_idx, rank_prefix_matrix, channel_prefix_matrix, \
        send_head, num_tokens, num_recv_tokens, hidden, num_topk, \
        buffer_ptrs, rank, \
        num_max_send_tokens, num_recv_buffer_tokens); \
    } } \
    break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype) SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving
    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
#ifndef DISABLE_SM90_FEATURES
    if (local_stream != nullptr) {
        // No grid synchronization or paired sender/receiver block cluster is
        // needed for split kernels. The two grids together retain num_sms CTAs.
        attr[0].val.cooperative = 0;
        attr[1].val.clusterDim.x = 1;
        cfg.gridDim.x = num_sms / 2;
    }
#endif
    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

// Push each remote contribution into its origin's shared buffer. The origin
// reduces local HBM reads; local contributions bypass staging entirely.
} // namespace intranode

} // namespace deep_ep
