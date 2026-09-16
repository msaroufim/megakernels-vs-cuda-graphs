// Seven-warp HC normalization, concurrent with warp zero's Sinkhorn loop.
#pragma once
#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_conversion.h>

#include "dspark_quant_scale.cuh"

namespace dspark_hc_overlap {
// Called only by threads 32..255. Virtual lane IDs retain the original
// 256-lane square chains and descending-stride reduction tree exactly.
template <class Stage>
__device__ __forceinline__ void reduce_norm(const Stage& stage, float* shared) {
  const int worker = static_cast<int>(threadIdx.x) - 32;
  auto* reduced = reinterpret_cast<__nv_bfloat16*>(shared + 256);
  float* pre = shared + 2304;
  if (worker < 4) {
    const float value =
        stage.mixes[worker] * stage.scale[0] + stage.base[worker];
    pre[worker] = 1.0f / (1.0f + expf(-value)) + 1.0e-6f;
  }
  // This also publishes the cp.async input copies completed by each worker.
  cutlass::arch::NamedBarrier::sync(224, 1);
  for (int virtual_lane = worker; virtual_lane < 256; virtual_lane += 224) {
    float square = 0.0f;
#pragma unroll 4
    for (int column = virtual_lane; column < 4096; column += 256) {
      float value = 0.0f;
#pragma unroll
      for (int stream = 0; stream < 4; ++stream) {
        value = fmaf(pre[stream],
                     __bfloat162float(stage.streams[stream * 4096 + column]),
                     value);
      }
      const __nv_bfloat16 stored = __float2bfloat16_rn(value);
      reduced[column] = stored;
      value = __bfloat162float(stored);
      square = fmaf(value, value, square);
    }
    shared[virtual_lane] = square;
  }
  cutlass::arch::NamedBarrier::sync(224, 1);
#pragma unroll
  for (int delta = 128; delta > 0; delta /= 2) {
    if (worker < delta) shared[worker] += shared[worker + delta];
    cutlass::arch::NamedBarrier::sync(224, 1);
  }
  const float inverse_rms = rsqrtf(shared[0] / 4096 + 1.0e-6f);
  const int warp = worker / 32;
  const int lane = worker % 32;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  for (int block = warp; block < 32; block += 7) {
    float values[4];
    float amax = 0.0f;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const int column = block * 128 + index * 32 + lane;
      const __nv_bfloat16 normalized =
          __float2bfloat16_rn(__bfloat162float(reduced[column]) * inverse_rms *
                              stage.norm_weight[column]);
      stage.normalized[column] = normalized;
      values[index] = __bfloat162float(normalized);
      amax = fmaxf(amax, fabsf(values[index]));
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
    amax = fmaxf(amax, 1.0e-4f);
    const int exponent = dspark_quant_scale::exponent(amax);
    if (lane == 0) {
      const int encoded = exponent + 127;
      stage.activation_scales[block] = static_cast<uint8_t>(
          encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
    }
    const float scale = dspark_quant_scale::scale(exponent);
#pragma unroll
    for (int index = 0; index < 4; ++index)
      stage.quantized[block * 128 + index * 32 + lane] =
          convert(values[index] / scale);
  }
}
}  // namespace dspark_hc_overlap
