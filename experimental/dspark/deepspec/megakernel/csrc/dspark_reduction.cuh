// Exact 256-lane descending-stride sum tree used by normalization phases.
#pragma once

#include <cuda_runtime.h>

namespace dspark_reduction {

template <bool kWarpFinish>
__device__ __forceinline__ void sum_256(float* shared) {
  // All 256 inputs have already been published by the caller's CTA barrier.
  if constexpr (kWarpFinish) {
    if (threadIdx.x < 32) {
      const int lane = static_cast<int>(threadIdx.x);
      // These are the original delta=128,64,32 tree nodes. Do not replace
      // them with a conventional warp-first reduction: that reassociates.
      const float a = shared[lane] + shared[lane + 128];
      const float b = shared[lane + 64] + shared[lane + 192];
      const float c = shared[lane + 32] + shared[lane + 160];
      const float d = shared[lane + 96] + shared[lane + 224];
      float value = (a + b) + (c + d);
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        const float other = __shfl_down_sync(0xffffffffu, value, delta);
        if (lane < delta) {
          value += other;
        }
      }
      if (lane == 0) {
        shared[0] = value;
      }
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int delta = 128; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] += shared[threadIdx.x + delta];
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void sum_256(float* shared) {
#ifdef DSPARK_WARP_TREE_REDUCTIONS
  sum_256<true>(shared);
#else
  sum_256<false>(shared);
#endif
}

// Keep the original descending-stride comparison tree, including its behavior
// for ties and NaNs. Warp 0 evaluates the first three levels in registers.
struct ArgmaxPair {
  float value;
  int index;
};

__device__ __forceinline__ ArgmaxPair choose_argmax(ArgmaxPair a,
                                                    ArgmaxPair b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

template <bool kWarpFinish>
__device__ __forceinline__ void argmax_256(float* values, int* indices) {
  if constexpr (kWarpFinish) {
    if (threadIdx.x < 32) {
      const int lane = static_cast<int>(threadIdx.x);
      const auto read = [&](int offset) {
        return ArgmaxPair{values[lane + offset], indices[lane + offset]};
      };
      const auto a = choose_argmax(read(0), read(128));
      const auto b = choose_argmax(read(64), read(192));
      const auto c = choose_argmax(read(32), read(160));
      const auto d = choose_argmax(read(96), read(224));
      auto result = choose_argmax(choose_argmax(a, b), choose_argmax(c, d));
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        ArgmaxPair other{__shfl_down_sync(0xffffffffu, result.value, delta),
                         __shfl_down_sync(0xffffffffu, result.index, delta)};
        if (lane < delta) result = choose_argmax(result, other);
      }
      if (lane == 0) {
        values[0] = result.value;
        indices[0] = result.index;
      }
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int delta = 128; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        const int lane = static_cast<int>(threadIdx.x);
        const auto result =
            choose_argmax({values[lane], indices[lane]},
                          {values[lane + delta], indices[lane + delta]});
        values[lane] = result.value;
        indices[lane] = result.index;
      }
      __syncthreads();
    }
  }
}

}  // namespace dspark_reduction
