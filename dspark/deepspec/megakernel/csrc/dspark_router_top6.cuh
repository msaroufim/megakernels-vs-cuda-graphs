#pragma once

#include <cuda_runtime.h>

#include <cfloat>

#include "dspark_reduction.cuh"

namespace dspark_router {

// Caller provides 256 threads. The finite-value selection is a lexicographic
// maximum (score, inverse expert index), so its grouping is immaterial. A NaN
// takes the original comparison tree, whose left-operand behavior matters.
__device__ __forceinline__ bool top6_single_warp(const float* scores,
                                                 const float* bias,
                                                 int* indices, float* weights) {
  const int thread = static_cast<int>(threadIdx.x);
  const float adjusted = scores[thread] + bias[thread];
  if (__syncthreads_or(isnan(adjusted))) return false;
  if (thread < 32) {
    const int lane = thread;
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      values[i] = scores[lane + i * 32] + bias[lane + i * 32];
    for (int selected = 0; selected < 6; ++selected) {
      dspark_reduction::ArgmaxPair best{values[0], lane};
#pragma unroll
      for (int i = 1; i < 8; ++i)
        best =
            dspark_reduction::choose_argmax(best, {values[i], lane + i * 32});
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        dspark_reduction::ArgmaxPair other{
            __shfl_down_sync(0xffffffffu, best.value, delta),
            __shfl_down_sync(0xffffffffu, best.index, delta)};
        if (lane < delta) best = dspark_reduction::choose_argmax(best, other);
      }
      const int chosen = __shfl_sync(0xffffffffu, best.index, 0);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (lane + i * 32 == chosen) values[i] = -FLT_MAX;
      if (lane == 0) indices[selected] = chosen;
    }
    if (lane == 0) {
      float sum = 0.0f;
      for (int selected = 0; selected < 6; ++selected) {
        const float value = scores[indices[selected]];
        weights[selected] = value;
        sum += value;
      }
      for (int selected = 0; selected < 6; ++selected)
        weights[selected] = 1.5f * weights[selected] / sum;
    }
  }
  __syncthreads();
  return true;
}

}  // namespace dspark_router
