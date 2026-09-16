// Grouped WO_A with K128/K256 operand rings. Preserve the original K16
// BF16 MMA chain and packed weights; quantize each row in one warp after
// the accumulator drain, avoiding per-row cross-warp epilogue barriers.
#pragma once
#include "dspark_proj_phase.cuh"

namespace dspark_woa_streamed {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
using dspark_lm::kDraftBlock;
using dspark_lm::kHidden;
using dspark_lm::kOutputTile;
using dspark_lm::kSharedBytesBudget;
using dspark_lm::kUmmaK;
constexpr int kVocab = 8 * 1024;
using dspark_lm::LmUmmaAtom;
using dspark_lm::LmUmmaSmemLayoutA;
using dspark_lm::LmUmmaSmemLayoutB;
using dspark_lm::LmUmmaTiledMma;

template <int SlotK>
struct Storage {
  static constexpr int kUmmaStages = 512 / SlotK;
  static constexpr int kUmmaSubTiles = SlotK / 16;
  static constexpr int kUmmaSlotElements = 128 * SlotK;
  static constexpr int kUmmaPartialBuffers = 2;
  alignas(128) __nv_bfloat16 b_ring[kUmmaStages][kUmmaSubTiles * 8 * kUmmaK];
  alignas(128) __nv_bfloat16 a_ring[kUmmaStages][kUmmaSlotElements];
  alignas(16) float partial[kUmmaPartialBuffers][kOutputTile * 8];
  alignas(16) cute::uint64_t stage_full[kUmmaStages];
  alignas(16) cute::uint64_t slot_free[kUmmaStages];
  alignas(16) cute::uint64_t accumulator_full[1];
  alignas(16) cute::uint64_t accumulator_free[1];
  alignas(16) cute::uint32_t tmem_base_ptr;
};

template <int SlotK>
__device__ inline void execute(const dspark_proj::WoaArgs& args) {
  using namespace cute;
  static_assert(SlotK == 128 || SlotK == 256);
  constexpr int kUmmaStages = 512 / SlotK;
  constexpr int kUmmaSubTiles = SlotK / 16;
  constexpr int kUmmaSlotElements = 128 * SlotK;
  constexpr int kUmmaSlotsPerItem = 4096 / SlotK;
  constexpr int kUmmaPartialBuffers = 2;
  constexpr int kStageArrivals = 33;
  constexpr int kTmemColumns = 32;
  static_assert(sizeof(Storage<SlotK>) <= kSharedBytesBudget,
                "UMMA staging ring + padded B must fit dynamic shared memory");
  extern __shared__ __align__(128) unsigned char dynamic_shared_memory[];
  auto& storage = *reinterpret_cast<Storage<SlotK>*>(dynamic_shared_memory);
  LmUmmaTiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto accumulator_shape =
      partition_shape_C(tiled_mma, make_shape(Int<kOutputTile>{}, Int<8>{}));
  auto tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape);
  dspark_tmem::Allocator1Sm tmem_allocator{};
  const int warp = static_cast<int>(threadIdx.x) / 32;

  if (warp == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(storage.stage_full,
                                                    kStageArrivals);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, kUmmaStages>(storage.slot_free, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(storage.accumulator_full, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, 1>(storage.accumulator_free,
                                          kOutputTile);
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  for (int i = threadIdx.x; i < kUmmaStages * kUmmaSubTiles * 8 * kUmmaK;
       i += 256) {
    reinterpret_cast<__nv_bfloat16*>(storage.b_ring)[i] =
        __float2bfloat16_rn(0.0f);
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  tmem_accumulator.data() = storage.tmem_base_ptr;

#if (DSPARK_DESCRIPTOR_ARITHMETIC & 1)
  const uint64_t a_descriptor_base =
      UMMA::make_umma_desc<UMMA::Major::K>(
          make_tensor(
              make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[0])),
              layout<0>(LmUmmaSmemLayoutA{})))
          .desc_;
  const auto a_descriptor_at = [=](int stage) {
    return a_descriptor_base +
           static_cast<uint64_t>(stage) * kUmmaSlotElements / 8;
  };
#else
  uint64_t a_descriptor[kUmmaStages];
  CUTE_UNROLL
  for (int s = 0; s < kUmmaStages; ++s) {
    a_descriptor[s] =
        UMMA::make_umma_desc<UMMA::Major::K>(
            make_tensor(
                make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.a_ring[s])),
                layout<0>(LmUmmaSmemLayoutA{})))
            .desc_;
  }
  const auto a_descriptor_at = [&](int stage) { return a_descriptor[stage]; };
#endif
  const uint64_t b_descriptor_base =
      UMMA::make_umma_desc<UMMA::Major::K>(
          make_tensor(
              make_smem_ptr(reinterpret_cast<bfloat16_t*>(storage.b_ring[0])),
              layout<0>(LmUmmaSmemLayoutB{})))
          .desc_;
  const uint64_t instruction_descriptor =
      UMMA::make_runtime_instr_desc<>(tiled_mma.idesc_);

  const int item_count = static_cast<int>(args.end - args.begin);
  const int total_slots = item_count * kUmmaSlotsPerItem;

  if (warp >= 5) {
    if (threadIdx.x == 160 || warp == 6) {
      for (int c = 0; c < total_slots; ++c) {
        const int s = c % kUmmaStages;
        if (c >= kUmmaStages) {
          wait_barrier(storage.slot_free[s], (c / kUmmaStages - 1) & 1);
        }
        const int item = static_cast<int>(args.begin) + c / kUmmaSlotsPerItem;
        const int slot_in_item = c % kUmmaSlotsPerItem;
        if (threadIdx.x == 160) {
          constexpr int bytes = kUmmaSlotElements * sizeof(__nv_bfloat16);
          const auto* source =
              args.wo_a +
              (static_cast<int64_t>(item) * kUmmaSlotsPerItem + slot_in_item) *
                  kUmmaSlotElements;
          cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
              &storage.stage_full[s], bytes);
          cute::SM90_BULK_COPY_G2S::copy(source, &storage.stage_full[s],
                                         storage.a_ring[s], bytes);
        } else {
          const int lane = threadIdx.x % 32;
          for (int p = lane; p < kUmmaSubTiles * kDraftBlock * 2; p += 32) {
            const int j = p / (kDraftBlock * 2),
                      row = (p % (kDraftBlock * 2)) / 2, half = p % 2;
            cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Always>(
                reinterpret_cast<uint8_t*>(storage.b_ring[s]) + j * 256 +
                    half * 128 + row * 16,
                args.attention + (row * 8 + item / 8) * kHidden +
                    (slot_in_item * kUmmaSubTiles + j) * kUmmaK + half * 8);
          }
          cutlass::arch::cpasync_barrier_arrive_noinc(&storage.stage_full[s]);
        }
      }
    }
  } else if (warp == 4) {
    for (int c = 0; c < total_slots; ++c) {
      const int s = c % kUmmaStages;
      const int slot_in_item = c % kUmmaSlotsPerItem;
      const int item_local = c / kUmmaSlotsPerItem;
      wait_barrier(storage.stage_full[s], (c / kUmmaStages) & 1);
      if (slot_in_item == 0 && item_local > 0) {
        wait_barrier(storage.accumulator_free[0], (item_local - 1) & 1);
      }
      CUTE_UNROLL
      for (int j = 0; j < kUmmaSubTiles; ++j) {
        LmUmmaAtom::fma(a_descriptor_at(s) + static_cast<uint64_t>(j) * 256,
                        b_descriptor_base +
                            static_cast<uint64_t>(s * kUmmaSubTiles + j) * 16,
                        storage.tmem_base_ptr,
                        (slot_in_item == 0 && j == 0) ? 0u : 1u,
                        instruction_descriptor);
      }
      cutlass::arch::umma_arrive(&storage.slot_free[s]);
      if (slot_in_item == kUmmaSlotsPerItem - 1) {
        cutlass::arch::umma_arrive(&storage.accumulator_full[0]);
      }
    }
  } else {
    auto tmem_to_register =
        make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tmem_accumulator);
    auto thread_copy = tmem_to_register.get_slice(threadIdx.x);
    auto slot_view = tmem_accumulator;
    slot_view.data() = storage.tmem_base_ptr;
    auto thread_tmem = thread_copy.partition_S(slot_view);
    auto partial_layout =
        make_layout(make_shape(Int<kOutputTile>{}, Int<8>{}),
                    make_stride(Int<1>{}, Int<kOutputTile>{}));
    auto shared_partial_low =
        make_tensor(make_smem_ptr(storage.partial[0]), partial_layout);
    auto thread_partial_low =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_low));
    auto shared_partial_high =
        make_tensor(make_smem_ptr(storage.partial[1]), partial_layout);
    auto thread_partial_high =
        thread_copy.partition_D(cta_mma.partition_C(shared_partial_high));
    auto register_accumulator = make_tensor<float>(shape(thread_partial_low));

    for (int i = 0; i < item_count; ++i) {
      const int parity = i & 1;
      wait_barrier(storage.accumulator_full[0], parity);
      copy(tmem_to_register, thread_tmem, register_accumulator);
      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::ClusterBarrier::arrive(&storage.accumulator_free[0]);
      if (parity == 0) {
        copy(register_accumulator, thread_partial_low);
      } else {
        copy(register_accumulator, thread_partial_high);
      }
      cutlass::arch::NamedBarrier::sync(4 * 32, 0);
      const int64_t tile_base =
          (static_cast<int64_t>(args.begin) + i) * kOutputTile;
      CUTE_UNROLL
      for (int row = 0; row < kDraftBlock; ++row) {
        args.output_lora[static_cast<int64_t>(row) * kVocab + tile_base +
                         threadIdx.x] =
            __float2bfloat16_rn(storage.partial[parity % kUmmaPartialBuffers]
                                               [row * kOutputTile +
                                                static_cast<int>(threadIdx.x)]);
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
  __syncthreads();
  const int lane = threadIdx.x % 32;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  for (uint32_t item = args.begin; item < args.end; ++item) {
    for (int row = warp; row < 5; row += 8) {
      const int base = row * kVocab + item * 128;
      float values[4];
      float maximum = 0.0f;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        values[part] =
            __bfloat162float(args.output_lora[base + lane + part * 32]);
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
        args.output_lora_scales[row * 64 + item] = static_cast<uint8_t>(
            encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
      }
#pragma unroll
      for (int part = 0; part < 4; ++part)
        args.output_lora_quantized[base + lane + part * 32] =
            convert(values[part] / scale);
    }
  }
}
#endif
}  // namespace dspark_woa_streamed
