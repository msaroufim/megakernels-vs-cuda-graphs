#pragma once

#include <cuda_bf16.h>

#include <cstdint>

// PTX m16n8k16 fragment mapping: four FP32 accumulators per lane.
// QK uses position x head; PV uses feature x head. N=8 holds exactly the
// eight live heads, eliminating the padded half of the WMMA16x16 path.
namespace dspark_attn_mma8_detail {
__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4],
                                     const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ld_a(uint32_t (&a)[4], const __nv_bfloat16* p,
                                     int stride) {
  int lane = threadIdx.x & 31;
  uint32_t addr =
      __cvta_generic_to_shared(p + (lane % 16) * stride + (lane / 16) * 8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}
__device__ __forceinline__ void ld_at(uint32_t (&a)[4], const __nv_bfloat16* p,
                                      int stride) {
  int lane = threadIdx.x & 31;
  uint32_t addr = __cvta_generic_to_shared(
      p + ((lane % 8) + (lane / 16) * 8) * stride + ((lane / 8) % 2) * 8);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(addr));
}
__device__ __forceinline__ void ld_b(uint32_t (&b)[2], const __nv_bfloat16* p,
                                     int stride) {
  int lane = threadIdx.x & 15;
  uint32_t addr =
      __cvta_generic_to_shared(p + (lane % 8) * stride + (lane / 8) * 8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(addr));
}

}  // namespace dspark_attn_mma8_detail
