// Copyright (c) 2026 The DeepSpec Authors
// SPDX-License-Identifier: MIT
// Adapted from this project's dspark_device_graph_runtime_kernel.cu preparation
// and publication kernels, and dspark_boltin_kv.cu's pinned SGLang0.5.16 gather.
// The accompanying LICENSE contains the MIT permission and warranty terms.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dspark_proposal_io {

constexpr std::uint64_t kAbi = 0x4453494F000100A0ULL;
constexpr int kWorkers = 152;
constexpr int kThreads = 256;
constexpr int kLayers = 3;
constexpr int kRingRows = 128;
constexpr int kHeadDim = 512;
constexpr int kNopeDim = 448;
constexpr int kQuantBlock = 64;
constexpr int kPageTokens = 256;
constexpr std::int64_t kTokenDataBytes = 576;
constexpr std::int64_t kScaleBytes = 8;
constexpr std::int64_t kScaleSectionOffset = kPageTokens * kTokenDataBytes;
constexpr std::int64_t kPageBytes = 149760;

// Appended at raw_workspace + generated::kWorkspaceBytes + 16. The first152
// bytes are immutable for the owner's lifetime; arrivals is the only mutable
// field. Host admission owns extents, alignment, disjoint storage and lifetime.
struct IntegrationContext {
  const std::int64_t* bonus;
  const std::int32_t* commit_len;
  const std::int64_t* new_seq_len;
  const __nv_bfloat16* target_hidden;
  const float* freqs_real;
  const std::uint8_t* swa0;
  const std::uint8_t* swa1;
  const std::uint8_t* swa2;
  const std::int32_t* req_to_token;
  const std::int64_t* full_to_swa;
  const std::int64_t* req_pool_indices;
  std::int64_t* candidates;
  std::int32_t* status;
  std::int64_t pages;
  std::int64_t req_rows;
  std::int64_t req_cols;
  std::int64_t full_locations;
  std::int64_t allow_zero_locations;
  std::int64_t freq_rows;
  std::uint64_t arrivals;
};

static_assert(std::is_standard_layout_v<IntegrationContext>);
static_assert(sizeof(void*) == 8 && sizeof(IntegrationContext) == 160);
static_assert(alignof(IntegrationContext) == 8);
static_assert(offsetof(IntegrationContext, bonus) == 0);
static_assert(offsetof(IntegrationContext, commit_len) == 8);
static_assert(offsetof(IntegrationContext, new_seq_len) == 16);
static_assert(offsetof(IntegrationContext, target_hidden) == 24);
static_assert(offsetof(IntegrationContext, freqs_real) == 32);
static_assert(offsetof(IntegrationContext, swa0) == 40);
static_assert(offsetof(IntegrationContext, swa1) == 48);
static_assert(offsetof(IntegrationContext, swa2) == 56);
static_assert(offsetof(IntegrationContext, req_to_token) == 64);
static_assert(offsetof(IntegrationContext, full_to_swa) == 72);
static_assert(offsetof(IntegrationContext, req_pool_indices) == 80);
static_assert(offsetof(IntegrationContext, candidates) == 88);
static_assert(offsetof(IntegrationContext, status) == 96);
static_assert(offsetof(IntegrationContext, pages) == 104);
static_assert(offsetof(IntegrationContext, req_rows) == 112);
static_assert(offsetof(IntegrationContext, req_cols) == 120);
static_assert(offsetof(IntegrationContext, full_locations) == 128);
static_assert(offsetof(IntegrationContext, allow_zero_locations) == 136);
static_assert(offsetof(IntegrationContext, freq_rows) == 144);
static_assert(offsetof(IntegrationContext, arrivals) == 152);

enum RowStatus : std::int32_t {
  kOk = 0,
  kInvalidRequest = 1,
  kLogicalOutOfBounds = 2,
  kInvalidFullLocation = 3,
  kInvalidSwaLocation = 4,
  kPositionMismatch = 5,
};

__device__ __forceinline__ std::uint64_t load_acquire(const void* address) {
  std::uint64_t value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
               : "=l"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ float decode_e4m3fn(std::uint8_t raw) {
  const std::uint32_t magnitude = raw & 0x7fU;
  const std::uint32_t exponent = magnitude >> 3;
  const std::uint32_t mantissa = magnitude & 0x07U;
  float value;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else if (exponent == 0x0fU && mantissa == 0x07U) {
    value = __int_as_float(0x7fffffff);
  } else {
    value = ldexpf(static_cast<float>(8U + mantissa), static_cast<int>(exponent) - 10);
  }
  return raw & 0x80U ? -value : value;
}

__device__ __forceinline__ float decode_ue8m0(std::uint8_t raw) {
  // Preserve raw0 => FP32zero, including the original signed-zero/NaN behavior
  // of the following FP8 multiplication and BF16 conversion.
  return __uint_as_float(static_cast<std::uint32_t>(raw) << 23);
}

__device__ __forceinline__ void gather_row(
    const IntegrationContext& context,
    int row,
    std::int64_t expected_position,
    __nv_bfloat16* out) {
  const int layer = row / kRingRows;
  const int ring_offset = row % kRingRows;
  auto* out_row = out + static_cast<std::int64_t>(row) * kHeadDim;
  const std::int64_t request = context.req_pool_indices[0];
  const std::int64_t last = context.new_seq_len[0] - 1;
  const std::int64_t logical = last - ((last - ring_offset) & (kRingRows - 1));
  std::int32_t code = kOk;
  std::int64_t swa_location = 0;
  if (expected_position != last) {
    code = kPositionMismatch;
  } else if (logical < 0) {
    // Missing history is valid even when a request mapping is not consulted.
  } else if (request < 0 || request >= context.req_rows) {
    code = kInvalidRequest;
  } else if (logical >= context.req_cols) {
    code = kLogicalOutOfBounds;
  } else {
    const std::int64_t full_location = static_cast<std::int64_t>(
        context.req_to_token[request * context.req_cols + logical]);
    if (full_location < (context.allow_zero_locations ? 0 : 1)
        || full_location >= context.full_locations) {
      code = kInvalidFullLocation;
    } else {
      swa_location = context.full_to_swa[full_location];
      if (swa_location < (context.allow_zero_locations ? 0 : 1)
          || swa_location >= context.pages * kPageTokens) {
        code = kInvalidSwaLocation;
      }
    }
  }
  if (logical < 0 || code != kOk) {
    reinterpret_cast<std::uint32_t*>(out_row)[threadIdx.x] = 0U;
    if (threadIdx.x == 0) context.status[row] = code;
    return;
  }
  const auto* layer_buffer = layer == 0 ? context.swa0
      : (layer == 1 ? context.swa1 : context.swa2);
  const std::int64_t page = swa_location / kPageTokens;
  const std::int64_t in_page = swa_location - page * kPageTokens;
  const auto* page_ptr = layer_buffer + page * kPageBytes;
  const auto* data_ptr = page_ptr + in_page * kTokenDataBytes;
  const auto* scale_ptr = page_ptr + kScaleSectionOffset + in_page * kScaleBytes;
  if (threadIdx.x < kNopeDim / 2) {
    const int element = static_cast<int>(threadIdx.x) * 2;
    const float scale = decode_ue8m0(scale_ptr[element / kQuantBlock]);
    out_row[element] = __float2bfloat16_rn(decode_e4m3fn(data_ptr[element]) * scale);
    out_row[element + 1] =
        __float2bfloat16_rn(decode_e4m3fn(data_ptr[element + 1]) * scale);
  } else {
    const int pair = static_cast<int>(threadIdx.x) - kNopeDim / 2;
    reinterpret_cast<std::uint32_t*>(out_row + kNopeDim)[pair] =
        reinterpret_cast<const std::uint32_t*>(data_ptr + kNopeDim)[pair];
  }
  if (threadIdx.x == 0) context.status[row] = kOk;
}

// Call collectively before constructing any mathematical stage descriptors.
// Private owners and physically nonoverlapping launches are mandatory. Between
// launches the host verifies arrivals == epoch*152; both start at zero. The
// return value replaces the caller's old device-epoch read.
__device__ __noinline__ std::uint32_t prepare(
    IntegrationContext* context,
    __nv_bfloat16* main_hidden,
    std::int32_t* anchor,
    float* rope,
    __nv_bfloat16* kv_cache,
    std::int64_t* start_pos,
    std::int64_t* epoch) {
  if (gridDim.x != kWorkers || gridDim.y != 1 || gridDim.z != 1
      || blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1
      || context == nullptr || main_hidden == nullptr || anchor == nullptr
      || rope == nullptr || kv_cache == nullptr || start_pos == nullptr || epoch == nullptr) {
    __trap();
  }
  const auto& c = *context;
  if (!c.bonus || !c.commit_len || !c.new_seq_len || !c.target_hidden || !c.freqs_real
      || !c.swa0 || !c.swa1 || !c.swa2 || !c.req_to_token || !c.full_to_swa
      || !c.req_pool_indices || !c.candidates || !c.status
      || c.pages <= 0 || c.pages > INT64_MAX / kPageBytes
      || c.req_rows <= 0 || c.req_cols <= 0 || c.req_rows > INT64_MAX / 4 / c.req_cols
      || c.full_locations <= 1 || c.full_locations > INT64_MAX / 8
      || c.freq_rows < 6 || c.freq_rows > INT64_MAX / 256
      || (c.allow_zero_locations != 0 && c.allow_zero_locations != 1)
      || ((reinterpret_cast<std::uintptr_t>(c.target_hidden)
           | reinterpret_cast<std::uintptr_t>(main_hidden)
           | reinterpret_cast<std::uintptr_t>(c.freqs_real)
           | reinterpret_cast<std::uintptr_t>(rope)) & 15U)) {
    __trap();
  }
  const std::int64_t sequence_length = c.new_seq_len[0];
  if (sequence_length <= 1 || sequence_length - 1 > INT32_MAX
      || sequence_length - 1 > c.freq_rows - 6) {
    __trap();
  }
  const std::int64_t position = sequence_length - 1;
  __shared__ std::uint64_t prior_epoch;
  if (threadIdx.x == 0) prior_epoch = load_acquire(epoch);
  __syncthreads();
  const std::uint64_t prior = prior_epoch;
  if (prior >= static_cast<std::uint64_t>(UINT32_MAX) - 1) __trap();
  const std::uint64_t next = prior + 1;

  if (blockIdx.x == 0) {
    const std::int32_t committed = c.commit_len[0];
    // Exactly clamp(commit_len-1,0,5), without overflow at INT32_MIN.
    const int commit_row = committed <= 1 ? 0 : (committed >= 6 ? 5 : committed - 1);
    if (threadIdx.x == 0) {
      anchor[0] = static_cast<std::int32_t>(c.bonus[0]);
      start_pos[0] = position;
    }
    const auto* source = reinterpret_cast<const uint4*>(
        c.target_hidden + commit_row * 3 * 4096);
    auto* destination = reinterpret_cast<uint4*>(main_hidden);
#pragma unroll
    for (int packet = threadIdx.x; packet < 3 * 4096 / 8; packet += kThreads) {
      destination[packet] = source[packet];
    }
    if (threadIdx.x < 6 * 32 * 2 / 4) {
      reinterpret_cast<uint4*>(rope)[threadIdx.x] =
          reinterpret_cast<const uint4*>(c.freqs_real + position * 64)[threadIdx.x];
    }
  }
  for (int row = blockIdx.x; row < kLayers * kRingRows; row += kWorkers) {
    gather_row(c, row, position, kv_cache);
  }

  // Every CTA latches prior before arriving, so observing next*152 proves no
  // lagging CTA can still read epoch after CTA0 advances it. Acq_rel RMWs form
  // a transitive release chain covering all gathered rows and CTA0's copies.
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint64_t previous_arrivals;
    asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;"
                 : "=l"(previous_arrivals)
                 : "l"(&context->arrivals), "l"(std::uint64_t{1}) : "memory");
    if (previous_arrivals < prior * kWorkers || previous_arrivals >= next * kWorkers) {
      __trap();
    }
    std::uint64_t arrived;
    do {
      arrived = load_acquire(&context->arrivals);
    } while (arrived < next * kWorkers);
    if (arrived != next * kWorkers) __trap();
  }
  __syncthreads();
  // Each CTA checks the completed global status vector before any math can run.
  // Invalid rows keep the exact gather status code and zero-filled data.
  int failed = 0;
  for (int row = threadIdx.x; row < kLayers * kRingRows; row += kThreads) {
    failed |= c.status[row] != kOk;
  }
  if (__syncthreads_or(failed)) __trap();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;"
                 : : "l"(epoch), "l"(next) : "memory");
  }
  return static_cast<std::uint32_t>(next);
}

// Caller must first acquire the existing proposal_done publication for this
// epoch. Call collectively on CTA0; all other CTAs can retire independently.
__device__ __noinline__ void publish(
    IntegrationContext* context, const std::int32_t* output_ids) {
  if (blockIdx.x != 0) return;
  if (threadIdx.x < 6) {
    context->candidates[threadIdx.x] = static_cast<std::int64_t>(output_ids[threadIdx.x]);
  }
  __syncthreads();
}

}  // namespace dspark_proposal_io

// Keep a non-inline, externally visible host symbol for the Python ABI check.
extern "C" __attribute__((visibility("default"))) std::uint64_t dspark_full_proposal_io_abi() {
  return dspark_proposal_io::kAbi;
}
