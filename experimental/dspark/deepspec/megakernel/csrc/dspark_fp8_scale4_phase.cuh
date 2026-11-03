// Dense FP8 block-scaled QB candidate. Offline A permutation and E8M0
// scale replication preserve operands. The full-K tensor-core accumulation
// changes FP32 rounding relative to the per-128-K software scale chain;
// this path therefore requires numerical validation, not just byte checks.
#pragma once
#include <cute/arch/copy_sm90_tma.hpp>

#include "dspark_proj_phase.cuh"
namespace dspark_fp8_scale4 {
template <int Stages>
struct Storage {
  alignas(128) uint8_t a[Stages][16384];
  alignas(128) uint8_t b[Stages][1024];
  alignas(128) uint32_t sfa[Stages][128], sfb[Stages][128];
  alignas(16) uint64_t full[Stages], scales[Stages], free[Stages];
  alignas(16) uint64_t acc_full[2], acc_free[2];
  alignas(16) float partial[2][128 * 8];
  uint32_t tmem;
};
struct Args {
  const cutlass::float_e4m3_t* input;
  const uint8_t* input_scales;
  const cutlass::float_e4m3_t* weight;
  const uint8_t* weight_scales;
  float* partials;
  __nv_bfloat16* output;
  uint32_t begin, end;
};

template <int Stages, int K, int M, int Rows, int Splits, bool Bf16, int OutputStride = M>
__device__ inline void execute(const Args& args) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  using namespace cute;
  static_assert(K % (Splits * 128) == 0 && M % 128 == 0);
  static_assert(Rows >= 1 && Rows <= 8);
  static_assert(!Bf16 || Splits == 1);
  constexpr int Groups = K / 128;
  constexpr int ItemGroups = Groups / Splits;
  static_assert(Groups % 4 == 0 && ItemGroups % 4 == 0,
                "SF4 loads require complete aligned groups of four K128 blocks");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& s = *reinterpret_cast<Storage<Stages>*>(dynamic_shared_memory);
  static_assert(sizeof(Storage<Stages>) <= 139520);
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  dspark_shared::TcgenTiledMma mma;
  auto cta = mma.get_slice(Int<0>{});
  auto acc = mma.make_fragment_C(
      partition_shape_C(mma, make_shape(Int<128>{}, Int<8>{})));
  dspark_tmem::Allocator1Sm allocator{};
  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, Stages>(s.full, 33);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, Stages>(s.scales, 32);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, Stages>(s.free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 2>(s.acc_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 2>(s.acc_free, 128);
    allocator.allocate(32, &s.tmem);
  }
  for (int i = tid; i < Stages * 1024; i += 256)
    reinterpret_cast<uint8_t*>(s.b)[i] = 0;
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  acc.data() = s.tmem;
  uint64_t da[Stages], db[Stages];
  CUTE_UNROLL
  for (int i = 0; i < Stages; i++) {
    da[i] =
        UMMA::make_umma_desc<UMMA::Major::K>(
            make_tensor(
                make_smem_ptr(reinterpret_cast<cutlass::float_e4m3_t*>(s.a[i])),
                layout<0>(dspark_shared::TcgenSmemLayoutA{})))
            .desc_;
    db[i] =
        UMMA::make_umma_desc<UMMA::Major::K>(
            make_tensor(
                make_smem_ptr(reinterpret_cast<cutlass::float_e4m3_t*>(s.b[i])),
                layout<0>(dspark_shared::TcgenSmemLayoutB{})))
            .desc_;
  }
  const auto idesc = UMMA::make_instr_desc_block_scaled<
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
      cutlass::float_ue8m0_t, 128, 8, UMMA::Major::K, UMMA::Major::K>();
  const int count = args.end - args.begin;
  if (tid == 160 || warp == 6) {
    for (int r = 0; r < count * ItemGroups; r++) {
      int stage = r % Stages, item = args.begin + r / ItemGroups;
      int tile = item / Splits;
      int kb = (item % Splits) * ItemGroups + r % ItemGroups;
      if (r >= Stages) cute::wait_barrier(s.free[stage], (r / Stages - 1) & 1);
      if (tid == 160) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &s.full[stage], 16384);
        SM90_BULK_COPY_G2S::copy(
            reinterpret_cast<const uint8_t*>(args.weight) +
                (static_cast<int64_t>(tile) * Groups + kb) * 16384,
            &s.full[stage], s.a[stage], 16384);
      } else {
        for (int p = lane; p < 8 * Rows; p += 32) {
          int sub = p / (2 * Rows), row = (p % (2 * Rows)) / 2, half = p % 2;
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
              s.b[stage] + sub * 256 + half * 128 + row * 16,
              args.input + row * K + kb * 128 + sub * 32 + half * 16);
        }
        // Four DIFFERENT consecutive K128 E8M0 bytes, not four copies
        // of one exponent. The MMA selector below chooses kb modulo four.
        // Keep the existing barrier phase/arrival counts at every K128 step.
        if ((kb & 3) == 0) {
          const uint32_t wa = *reinterpret_cast<const uint32_t*>(
              args.weight_scales + tile * Groups + kb);
          for (int w = lane; w < 128; w += 32) {
            int row = (w % 4) * 32 + w / 4;
            s.sfa[stage][w] = wa;
            s.sfb[stage][w] = row < Rows
                ? *reinterpret_cast<const uint32_t*>(
                      args.input_scales + row * Groups + kb)
                : 0x7f7f7f7fu;
          }
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::ClusterBarrier::arrive(&s.scales[stage]);
        cutlass::arch::cpasync_barrier_arrive_noinc(&s.full[stage]);
      }
    }
  } else if (warp == 4) {
    for (int r = 0; r < count * ItemGroups; r++) {
      int stage = r % Stages, item = r / ItemGroups, kb = r % ItemGroups,
          e = item & 1;
      cute::wait_barrier(s.full[stage], (r / Stages) & 1);
      cute::wait_barrier(s.scales[stage], (r / Stages) & 1);
      if (kb == 0 && item >= 2)
        cute::wait_barrier(s.acc_free[e], (item / 2 - 1) & 1);
      if (elect_one_sync()) {
        if ((kb & 3) == 0) {
          dspark_w13::dg_utccp_32x128b_warpx4(
              dspark_w13::dg_make_sf_desc(s.sfa[stage]), s.tmem + 16);
          dspark_w13::dg_utccp_32x128b_warpx4(
              dspark_w13::dg_make_sf_desc(s.sfb[stage]), s.tmem + 20);
        }
        CUTE_UNROLL
        for (int j = 0; j < 4; j++)
          dspark_w13::dg_mma_mxf8f6f4(
              da[stage] + j * 256, db[stage] + j * 16, s.tmem + e * 8, kb || j,
              dspark_w13::dg_runtime_instr_desc(idesc, kb & 3, kb & 3), s.tmem + 16,
              s.tmem + 20);
      }
      __syncwarp();
      cutlass::arch::umma_arrive(&s.free[stage]);
      if (kb == ItemGroups - 1) cutlass::arch::umma_arrive(&s.acc_full[e]);
    }
  } else if (warp < 4) {
    auto cp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, acc);
    auto tc = cp.get_slice(tid);
    auto a0 = acc, a1 = acc;
    a1.data() = s.tmem + 8;
    auto src0 = tc.partition_S(a0), src1 = tc.partition_S(a1);
    auto pl = make_layout(make_shape(Int<128>{}, Int<8>{}),
                          make_stride(Int<1>{}, Int<128>{}));
    auto d0 = tc.partition_D(
        cta.partition_C(make_tensor(make_smem_ptr(s.partial[0]), pl)));
    auto d1 = tc.partition_D(
        cta.partition_C(make_tensor(make_smem_ptr(s.partial[1]), pl)));
    auto reg = make_tensor<float>(shape(d0));
    for (int item = 0; item < count; item++) {
      int e = item & 1;
      cute::wait_barrier(s.acc_full[e], (item / 2) & 1);
      if (e == 0)
        copy(cp, src0, reg);
      else
        copy(cp, src1, reg);
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&s.acc_free[e]);
      if (e == 0)
        copy(reg, d0);
      else
        copy(reg, d1);
      cutlass::arch::NamedBarrier::sync(128, 0);
      CUTE_UNROLL
      for (int row = 0; row < Rows; row++) {
        int logical = args.begin + item;
        int output_index = row * OutputStride + (logical / Splits) * 128 + tid;
        if constexpr (Bf16) {
          args.output[output_index] =
              __float2bfloat16_rn(s.partial[e][row * 128 + tid]);
        } else {
          args.partials[output_index * Splits + logical % Splits] =
              s.partial[e][row * 128 + tid];
        }
      }
    }
  }
  // Close every final asynchronous acknowledgement before ending this
  // invocation's barrier lifetimes. Counts are local to this execute call.
  if (tid == 128) {
    const int rounds = count * ItemGroups;
    CUTE_UNROLL
    for (int stage = 0; stage < Stages; ++stage) {
      if (stage < rounds) {
        const int last = stage + ((rounds - 1 - stage) / Stages) * Stages;
        cute::wait_barrier(s.free[stage], (last / Stages) & 1);
      }
    }
    CUTE_UNROLL
    for (int e = 0; e < 2; ++e) {
      if (e < count) {
        const int last = e + ((count - 1 - e) / 2) * 2;
        cute::wait_barrier(s.acc_free[e], (last / 2) & 1);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    CUTE_UNROLL
    for (int stage = 0; stage < Stages; ++stage) {
      cutlass::arch::ClusterBarrier::invalidate(&s.full[stage]);
      cutlass::arch::ClusterBarrier::invalidate(&s.scales[stage]);
      cutlass::arch::ClusterBarrier::invalidate(&s.free[stage]);
    }
    CUTE_UNROLL
    for (int e = 0; e < 2; ++e) {
      cutlass::arch::ClusterBarrier::invalidate(&s.acc_full[e]);
      cutlass::arch::ClusterBarrier::invalidate(&s.acc_free[e]);
    }
  }
  if (warp == 0) allocator.free(s.tmem, 32);
  __syncthreads();
#endif
}
template <int Stages>
__device__ inline void qb(const dspark_proj::QbArgs& args) {
  execute<Stages, 1024, 32768, 5, 1, true>(
      {args.q_lora_quantized, args.q_lora_scales, args.wqb, args.wqb_scales,
       nullptr, args.query_projection, args.begin, args.end});
}
}  // namespace dspark_fp8_scale4
