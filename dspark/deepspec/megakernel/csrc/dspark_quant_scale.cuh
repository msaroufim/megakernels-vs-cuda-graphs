// E8M0 scale construction for finite BF16 amax after the 1e-4 floor.
#pragma once
#include <cuda_runtime.h>

namespace dspark_quant_scale {
__device__ __forceinline__ int exponent(float amax) {
#ifdef DSPARK_BF16_QUANT_BITS
  // 448 = 1.75 * 2^8. BF16 mantissas give an exact comparison with 1.75;
  // ceil(log2(amax / 448)) increases only when that threshold is exceeded.
  // Exhaustively checked against the retained CUDA path for all finite BF16
  // magnitudes, including the floor and exact power-of-two boundaries.
  if (isfinite(amax)) {
    const unsigned bits = __float_as_uint(amax);
    return static_cast<int>((bits >> 23) & 255) - 135 +
           ((bits & 0x7fffff) > 0x600000);
  }
#endif
  return static_cast<int>(ceilf(log2f(amax / 448.0f)));
}
__device__ __forceinline__ float scale(int exponent) {
#ifdef DSPARK_BF16_QUANT_BITS
  if (exponent >= -126 && exponent <= 127)
    return __uint_as_float(static_cast<unsigned>(exponent + 127) << 23);
#endif
  return ldexpf(1.0f, exponent);
}
}  // namespace dspark_quant_scale
