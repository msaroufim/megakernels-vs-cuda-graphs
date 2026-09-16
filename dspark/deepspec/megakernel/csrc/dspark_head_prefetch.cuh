#pragma once
// Exact head mHC projection: stream the four FP32 weight rows through a
// two-slot ring, retaining each thread's ascending FMA chain and the original
// 256-lane reduction trees. Five independent trees share each CTA barrier.
#include <cuda_bf16.h>
#include <cutlass/arch/memory_sm80.h>
namespace dspark_head_prefetch {
constexpr int kStreamsOffset = 16384;
// Keep the staged input above both the five reduction trees and the caller's
// BF16 head-hidden scratch at bytes [1040, 9232).
constexpr int kWeightsOffset = 49152;
constexpr int kStageK = 2048;
constexpr int kWeightSlotBytes = 4 * kStageK * 4;
constexpr int kSharedBytes = kWeightsOffset + 2 * kWeightSlotBytes;
template <class Stage>
__device__ __forceinline__ const __nv_bfloat16* project(Stage stage, int row,
                                                        float* shared) {
  const int lane = threadIdx.x;
  auto* bytes = reinterpret_cast<unsigned char*>(shared);
  auto* streams = reinterpret_cast<__nv_bfloat16*>(bytes + kStreamsOffset);
  auto* weights = reinterpret_cast<float*>(bytes + kWeightsOffset);
#pragma unroll
  for (int packet = lane; packet < 2048; packet += 256) {
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
        reinterpret_cast<uint4*>(streams) + packet,
        reinterpret_cast<const uint4*>(stage.streams + row * 16384) + packet);
  }
  const auto load_weights = [&](int tile) {
#pragma unroll
    for (int packet = lane; packet < 2048; packet += 256) {
      const int stream = packet / 512;
      const int column = (packet % 512) * 4;
      cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
          reinterpret_cast<uint4*>(weights + (tile & 1) * 8192) + packet,
          reinterpret_cast<const uint4*>(stage.hc_fn + stream * 16384 +
                                         tile * 2048 + column));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  load_weights(0);
  load_weights(1);
  float square = 0.0f;
  float dot[4] = {0, 0, 0, 0};
#pragma unroll 1
  for (int tile = 0; tile < 8; tile++) {
    if (tile == 7)
      asm volatile("cp.async.wait_group 0;\n" ::);
    else
      asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    // Every lane finishes reading a slot before any lane reuses it below.
#pragma unroll
    for (int step = 0; step < 8; step++) {
      const int column = tile * 2048 + lane + step * 256;
      const float value = __bfloat162float(streams[column]);
      square = fmaf(value, value, square);
#pragma unroll
      for (int stream = 0; stream < 4; stream++)
        dot[stream] =
            fmaf(value,
                 weights[(tile & 1) * 8192 + stream * 2048 + lane + step * 256],
                 dot[stream]);
    }
    __syncthreads();
    if (tile + 2 < 8) load_weights(tile + 2);
  }
  shared[lane] = square;
#pragma unroll
  for (int stream = 0; stream < 4; stream++)
    shared[(stream + 1) * 256 + lane] = dot[stream];
  __syncthreads();
#pragma unroll
  for (int delta = 128; delta; delta /= 2) {
    if (lane < delta) {
#pragma unroll
      for (int value = 0; value < 5; value++)
        shared[value * 256 + lane] += shared[value * 256 + lane + delta];
    }
    __syncthreads();
  }
  if (lane == 0) {
    const float inverse_rms = rsqrtf(shared[0] / 16384 + 1.0e-6f);
#pragma unroll
    for (int stream = 0; stream < 4; stream++) {
      const float mix = shared[(stream + 1) * 256] * inverse_rms;
      const float logit = mix * stage.hc_scale[0] + stage.hc_base[stream];
      shared[256 + stream] = 1.0f / (1.0f + expf(-logit)) + 1.0e-6f;
    }
  }
  __syncthreads();
  return streams;
}
}  // namespace dspark_head_prefetch
