#pragma once

#include "dspark_epi_phase.cuh"

namespace dspark_swiglu {

// One warp owns one complete 128-value quantization block. Preserve the
// BF16 intermediate and the reference scale formula; only max is regrouped.
template <bool Routed>
__device__ inline void execute(const dspark_epi::SwigluArgs& args) {
  constexpr int kRows =
      dspark_epi::kDraftRows * (Routed ? dspark_epi::kActivatedExperts : 1);
  constexpr int kElements = kRows * dspark_epi::kIntermediate;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  const int block_begin = static_cast<int>(args.begin) * 4;
  const int block_end = static_cast<int>(args.end) * 4;
  for (int block = block_begin + warp; block < block_end; block += 8) {
    const int base = block * 128;
    if (base >= kElements) continue;
    const int row = base / dspark_epi::kIntermediate;
    const int feature = base % dspark_epi::kIntermediate;
    float values[4];
    float maximum = 0.0f;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int column = lane + part * 32;
      const int source = row * 2 * dspark_epi::kIntermediate + feature + column;
      float gate = __bfloat162float(args.w13[source]);
      float up = __bfloat162float(args.w13[source + dspark_epi::kIntermediate]);
      gate = fminf(gate, dspark_epi::kSwiGluLimit);
      up =
          fminf(fmaxf(up, -dspark_epi::kSwiGluLimit), dspark_epi::kSwiGluLimit);
      float value = gate / (1.0f + expf(-gate)) * up;
      if constexpr (Routed) value *= args.route_weights[row];
      const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
      args.swiglu[base + column] = rounded;
      values[part] = __bfloat162float(rounded);
      maximum = fmaxf(maximum, fabsf(values[part]));
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    maximum = fmaxf(maximum, 1.0e-4f);
    const int exponent = static_cast<int>(ceilf(log2f(maximum / 448.0f)));
    const float scale = ldexpf(1.0f, exponent);
    if (lane == 0) {
      const int encoded = exponent + 127;
      args.swiglu_scales[block] = static_cast<uint8_t>(
          encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
    }
#pragma unroll
    for (int part = 0; part < 4; ++part)
      args.swiglu_quantized[base + lane + part * 32] =
          convert(values[part] / scale);
  }
}

}  // namespace dspark_swiglu
