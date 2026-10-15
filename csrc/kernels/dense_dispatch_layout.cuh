#pragma once
#include <cstddef>

#ifdef __CUDACC__
#define DENSE_HD __host__ __device__
#else
#define DENSE_HD
#endif
namespace deep_ep::intranode {
struct DenseDispatchLayout {
    size_t channel, x, source, topk, weights, scales, sf, bytes;
    DENSE_HD static size_t align16(size_t bytes) { return (bytes + 15) / 16 * 16; }
    DENSE_HD DenseDispatchLayout(size_t rows, size_t hidden_bytes, int ranks, int channels,
                                int num_topk, int num_scales, int num_sf) {
        channel = align16(ranks * ranks * sizeof(int));
        x = align16(channel + ranks * channels * sizeof(int));
        source = align16(x + rows * hidden_bytes);
        topk = align16(source + rows * sizeof(int));
        weights = align16(topk + rows * num_topk * sizeof(long long));
        scales = align16(weights + rows * num_topk * sizeof(float));
        sf = align16(scales + rows * num_scales * sizeof(float));
        bytes = align16(sf + rows * num_sf * sizeof(float));
    }
};
}
#undef DENSE_HD
