#pragma once
#include "dspark_small_phase.cuh"
namespace dspark_small {
__device__ inline void execute_hc_mixgroup_staged_weights(const HcArgs& args, float* shared) {
  constexpr int kGroup=1, kUnrollOverride=16;
  constexpr bool kCpAsync=true;
  constexpr int kMixes = kHcMixes;
  constexpr int kFlattened = kHcFlattened;
  static_assert(kMixes % kGroup == 0, "mix group must divide the 24 mixes");
  constexpr int kGroups = kMixes / kGroup;
  // Keep total loads in flight near 8-12 so the register window stays small
  // enough not to tax the megakernel's 252-register budget.
  constexpr int kUnroll =
      kUnrollOverride > 0
          ? kUnrollOverride
          : (kGroup <= 2 ? 8 / kGroup
                         : (kGroup <= 4 ? 12 / kGroup : (kGroup <= 8 ? 2 : 1)));
  // Scratch holds the square tree plus one tree per mix, all folded by a
  // single 8-level barrier pass (independent trees, unchanged order).
  constexpr int kScratchFloats = (kGroup + 1) * kHcThreads;
  constexpr int kPackets = kFlattened * sizeof(__nv_bfloat16) / 16;  // 2048
  constexpr int kPacketsPerThread = kPackets / kHcThreads;           // 8
  static_assert(
      kScratchFloats * sizeof(float) + kFlattened * sizeof(__nv_bfloat16)
          <= kSharedBytesBudget,
      "HC mix-group staging must fit dynamic shared memory");
  static_assert(kScratchFloats * sizeof(float) + kFlattened * (sizeof(__nv_bfloat16) + sizeof(float)) <= kSharedBytesBudget, "HC staged input/weight scratch exceeds phase budget");
  float* reduction_scratch = shared;
  __nv_bfloat16* staged_streams =
      reinterpret_cast<__nv_bfloat16*>(shared + kScratchFloats);
  const int lane = static_cast<int>(threadIdx.x);
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const int row = static_cast<int>(item) / kGroups;
    const int group = static_cast<int>(item) % kGroups;
    // Weight staging is the only dataflow change. Existing input copy remains intact.
    float* staged_weights=reinterpret_cast<float*>(staged_streams+kFlattened);
    const unsigned weight_address=static_cast<unsigned>(__cvta_generic_to_shared(staged_weights));
    const float* source_weights=args.fn_weight+group*kFlattened;
#pragma unroll
      for(int packet=0;packet<16;++packet) {
        const int offset=lane+packet*256;
        const unsigned dst=weight_address+offset*16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst),"l"(reinterpret_cast<const uint4*>(source_weights)+offset));
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    
    // (a) cache the row. Pure data movement -- the BF16 bit patterns are
    // copied verbatim, so every value the chains below consume is the one
    // execute_hc_parallel would have loaded from GMEM.
    {
      const uint4* source =
          reinterpret_cast<const uint4*>(args.streams + row * kFlattened);
      uint4* destination = reinterpret_cast<uint4*>(staged_streams);
      if (kCpAsync) {
#pragma unroll
        for (int packet = 0; packet < kPacketsPerThread; ++packet) {
          const int index = lane + packet * kHcThreads;
          const unsigned address = static_cast<unsigned>(
              __cvta_generic_to_shared(destination + index));
          asm volatile(
              "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address),
              "l"(source + index));
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
      } else {
#pragma unroll
        for (int packet = 0; packet < kPacketsPerThread; ++packet) {
          const int index = lane + packet * kHcThreads;
          destination[index] = source[index];
        }
      }
    }
    __syncthreads();
    // One base pointer plus compile-time slot offsets: a kGroup-wide array of
    // pointers would burn 2 * kGroup registers for nothing (kGroup = 24 alone
    // would be 48), and slot * kFlattened * 4 <= 1.5 MB fits the LDG
    // immediate-offset field.
    const float* weight_base = staged_weights;
    float local_square = 0.0f;
    float local_dot[kGroup];
#pragma unroll
    for (int slot = 0; slot < kGroup; ++slot) {
      local_dot[slot] = 0.0f;
    }
#pragma unroll kUnroll
    for (int step = 0; step < kHcPerThread; ++step) {
      const int column = lane + step * kHcThreads;
      const float value = __bfloat162float(staged_streams[column]);
      local_square = fmaf(value, value, local_square);
#pragma unroll
      for (int slot = 0; slot < kGroup; ++slot) {
        local_dot[slot] = fmaf(
            value, weight_base[slot * kFlattened + column], local_dot[slot]);
      }
    }
    // reduction_scratch is disjoint from staged_streams and is written before
    // it is read, so the item-trailing barrier is the only one needed here.
    reduction_scratch[lane] = local_square;
#pragma unroll
    for (int slot = 0; slot < kGroup; ++slot) {
      reduction_scratch[(slot + 1) * kHcThreads + lane] = local_dot[slot];
    }
    __syncthreads();
    // One barrier pass folds all kGroup + 1 independent halving trees; each
    // tree sees the same operands in the same order as the per-mix trees it
    // replaces.
#pragma unroll
    for (int delta = kHcThreads / 2; delta > 0; delta /= 2) {
      if (lane < delta) {
#pragma unroll
        for (int slot = 0; slot < kGroup + 1; ++slot) {
          reduction_scratch[slot * kHcThreads + lane] +=
              reduction_scratch[slot * kHcThreads + lane + delta];
        }
      }
      __syncthreads();
    }
    if (lane < kGroup) {
      const float inverse_rms =
          rsqrtf(reduction_scratch[0] / kFlattened + kNormEpsilon);
      args.mixes[row * kMixes + group * kGroup + lane] =
          reduction_scratch[(lane + 1) * kHcThreads] * inverse_rms;
    }
    __syncthreads();
  }
}
}
