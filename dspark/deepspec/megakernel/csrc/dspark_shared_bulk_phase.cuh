// Shared W13 bulk staging preserves the existing per-K128 scale/FP32 order.
// Bulk FP8 staging with the original per-K128 arithmetic boundaries.
// Four K32 MMAs form each unscaled inner product; the drain applies the
// original E8M0 product and ordered FP32 fmaf chain. Four TMEM slots let
// staging, MMA, and draining overlap without changing the reduction order.
#pragma once
#include <cute/arch/copy_sm90_tma.hpp>

#include "dspark_epi_phase.cuh"
#include "dspark_proj_phase.cuh"
namespace dspark_shared_bulk {
template <int Stages>
struct Storage {
  alignas(128) uint8_t a[Stages][16384];
  alignas(128) uint8_t b[Stages][1024];
  alignas(16) uint64_t full[Stages], free[Stages];
  alignas(16) uint64_t acc_full[4], acc_free[4];
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
        cutlass::arch::ClusterBarrier, Stages>(s.free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 4>(s.acc_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 4>(s.acc_free, 128);
    allocator.allocate(32, &s.tmem);
  }
  for (int i = tid; i < Stages * 1024; i += 256)
    reinterpret_cast<uint8_t*>(s.b)[i] = 0;
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  acc.data() = s.tmem;
#if (DSPARK_DESCRIPTOR_ARITHMETIC & 2)
  const uint64_t da_base = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
      make_smem_ptr(reinterpret_cast<cutlass::float_e4m3_t*>(s.a[0])),
      layout<0>(dspark_shared::TcgenSmemLayoutA{}))).desc_;
  const uint64_t db_base = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
      make_smem_ptr(reinterpret_cast<cutlass::float_e4m3_t*>(s.b[0])),
      layout<0>(dspark_shared::TcgenSmemLayoutB{}))).desc_;
  const auto da_at = [=](int stage) { return da_base + static_cast<uint64_t>(stage) * 1024; };
  const auto db_at = [=](int stage) { return db_base + static_cast<uint64_t>(stage) * 64; };
#else
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
  const auto da_at = [&](int stage) { return da[stage]; };
  const auto db_at = [&](int stage) { return db[stage]; };
#endif
  const uint64_t idesc = UMMA::make_runtime_instr_desc<>(mma.idesc_);
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
        cutlass::arch::cpasync_barrier_arrive_noinc(&s.full[stage]);
      }
    }
  } else if (warp == 4) {
    for (int r = 0; r < count * ItemGroups; r++) {
      int stage = r % Stages, e = r % 4;
      cute::wait_barrier(s.full[stage], (r / Stages) & 1);
      if (r >= 4) cute::wait_barrier(s.acc_free[e], (r / 4 - 1) & 1);
      CUTE_UNROLL
      for (int j = 0; j < 4; j++)
        SM100_MMA_F8F6F4_SS::fma(da_at(stage) + j * 256, db_at(stage) + j * 16,
                                 s.tmem + e * 8, j ? 1u : 0u, idesc);
      cutlass::arch::umma_arrive(&s.free[stage]);
      cutlass::arch::umma_arrive(&s.acc_full[e]);
    }
  } else if (warp < 4) {
    auto cp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, acc);
    auto tc = cp.get_slice(tid);
    auto a0 = acc, a1 = acc, a2 = acc, a3 = acc;
    a1.data() = s.tmem + 8;
    a2.data() = s.tmem + 16;
    a3.data() = s.tmem + 24;
    auto src0 = tc.partition_S(a0), src1 = tc.partition_S(a1);
    auto src2 = tc.partition_S(a2), src3 = tc.partition_S(a3);
    auto pl = make_layout(make_shape(Int<128>{}, Int<8>{}),
                          make_stride(Int<1>{}, Int<128>{}));
    auto d0 = tc.partition_D(
        cta.partition_C(make_tensor(make_smem_ptr(s.partial[0]), pl)));
    auto d1 = tc.partition_D(
        cta.partition_C(make_tensor(make_smem_ptr(s.partial[1]), pl)));
    auto reg = make_tensor<float>(shape(d0));
    float result[Rows];
    for (int r = 0; r < count * ItemGroups; r++) {
      int e = r % 4, parity = r & 1, block = r % ItemGroups;
      int logical = args.begin + r / ItemGroups;
      int tile = logical / Splits, split = logical % Splits;
      if (block == 0) {
        CUTE_UNROLL
        for (int row = 0; row < Rows; row++) result[row] = 0.0f;
      }
      cute::wait_barrier(s.acc_full[e], (r / 4) & 1);
      if (e == 0)
        copy(cp, src0, reg);
      else if (e == 1)
        copy(cp, src1, reg);
      else if (e == 2)
        copy(cp, src2, reg);
      else
        copy(cp, src3, reg);
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&s.acc_free[e]);
      if (parity == 0)
        copy(reg, d0);
      else
        copy(reg, d1);
      cutlass::arch::NamedBarrier::sync(128, 0);
      int kb = split * ItemGroups + block;
      float ws =
          dspark_epi::decode_e8m0(args.weight_scales[tile * Groups + kb]);
      CUTE_UNROLL
      for (int row = 0; row < Rows; row++) {
        float xs =
            dspark_epi::decode_e8m0(args.input_scales[row * Groups + kb]);
        result[row] =
            fmaf(s.partial[parity][row * 128 + tid], xs * ws, result[row]);
        if (block == ItemGroups - 1) {
          int output_index = row * OutputStride + tile * 128 + tid;
          if constexpr (Bf16)
            args.output[output_index] = __float2bfloat16_rn(result[row]);
          else
            args.partials[output_index * Splits + split] = result[row];
        }
      }
    }
  }
  __syncthreads();
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
}  // namespace dspark_shared_bulk
