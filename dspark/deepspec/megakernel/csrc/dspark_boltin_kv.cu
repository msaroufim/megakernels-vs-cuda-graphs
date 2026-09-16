#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Pinned SGLang v0.5.16 DeepSeek-V4 SWA layout.  The first 147456 bytes of
// every page hold 256 contiguous 576-byte data rows; the next 2048 bytes hold
// 256 padded 8-byte scale rows; the final 256 bytes are page padding.
constexpr int kLayers = 3;
constexpr int kRingRows = 128;
constexpr int kHeadDim = 512;
constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kQuantBlock = 64;
constexpr int kPageTokens = 256;
constexpr int64_t kTokenDataBytes = 576;
constexpr int64_t kScaleBytes = 8;
constexpr int64_t kScaleSectionOffset = kPageTokens * kTokenDataBytes;
constexpr int64_t kPageBytes = 149760;

enum RowStatus : int32_t {
  kOk = 0,
  kInvalidRequest = 1,
  kLogicalOutOfBounds = 2,
  kInvalidFullLocation = 3,
  kInvalidSwaLocation = 4,
  kPositionMismatch = 5,
};

__device__ __forceinline__ float decode_e4m3fn(uint8_t raw) {
  const uint32_t magnitude = raw & 0x7fU;
  const uint32_t exponent = magnitude >> 3;
  const uint32_t mantissa = magnitude & 0x07U;

  float value;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else if (exponent == 0x0fU && mantissa == 0x07U) {
    value = __int_as_float(0x7fffffff);  // E4M3FN NaN encoding.
  } else {
    value = ldexpf(static_cast<float>(8U + mantissa), static_cast<int>(exponent) - 10);
  }
  return raw & 0x80U ? -value : value;
}

__device__ __forceinline__ float decode_ue8m0(uint8_t raw) {
  // The raw byte is a biased FP32 exponent.  This also gives the official
  // zero behavior for raw==0, unlike ldexpf(1.0f, -127), which is subnormal.
  return __uint_as_float(static_cast<uint32_t>(raw) << 23);
}

__device__ __forceinline__ void zero_row(__nv_bfloat16* row) {
  // Exactly 512 BF16 zeros: one naturally aligned 32-bit store per thread.
  reinterpret_cast<uint32_t*>(row)[threadIdx.x] = 0U;
}

__global__ void gather_target_kv_kernel(
    const uint8_t* swa0,
    const uint8_t* swa1,
    const uint8_t* swa2,
    int64_t pages,
    const int32_t* req_to_token,
    int64_t req_rows,
    int64_t req_cols,
    const int64_t* full_to_swa,
    int64_t full_locations,
    const int64_t* req_pool_indices,
    const int64_t* seq_lens,
    const int64_t* expected_pos,
    bool allow_zero_locations,
    __nv_bfloat16* out,
    int32_t* status) {
  const int layer = static_cast<int>(blockIdx.x);
  const int ring_offset = static_cast<int>(blockIdx.y);
  __nv_bfloat16* out_row =
      out + (static_cast<int64_t>(layer) * kRingRows + ring_offset) * kHeadDim;
  int32_t* row_status = status + layer * kRingRows + ring_offset;

  const int64_t request = req_pool_indices[0];
  const int64_t last = seq_lens[0] - 1;
  const int64_t logical = last - ((last - ring_offset) & (kRingRows - 1));

  int32_t code = kOk;
  int64_t swa_location = 0;
  if (expected_pos != nullptr && expected_pos[0] != last) {
    // The commit-tracked position and the batch's seq_lens disagree; a
    // proposal built from either would be mapping-valid garbage, so poison
    // every status word and let the consumer fail closed.
    code = kPositionMismatch;
  } else if (logical < 0) {
    // Missing history is a valid zero-filled ring row.
  } else if (request < 0 || request >= req_rows) {
    code = kInvalidRequest;
  } else if (logical >= req_cols) {
    code = kLogicalOutOfBounds;
  } else {
    const int64_t full_location =
        static_cast<int64_t>(req_to_token[request * req_cols + logical]);
    if (full_location < (allow_zero_locations ? 0 : 1)
        || full_location >= full_locations) {
      code = kInvalidFullLocation;
    } else {
      swa_location = full_to_swa[full_location];
      if (swa_location < (allow_zero_locations ? 0 : 1)
          || swa_location >= pages * kPageTokens) {
        code = kInvalidSwaLocation;
      }
    }
  }

  if (logical < 0 || code != kOk) {
    zero_row(out_row);
    if (threadIdx.x == 0) {
      *row_status = code;
    }
    return;
  }

  const uint8_t* layer_buffer = layer == 0 ? swa0 : (layer == 1 ? swa1 : swa2);
  const int64_t page = swa_location / kPageTokens;
  const int64_t in_page = swa_location - page * kPageTokens;
  const uint8_t* page_ptr = layer_buffer + page * kPageBytes;
  const uint8_t* data_ptr = page_ptr + in_page * kTokenDataBytes;
  const uint8_t* scale_ptr =
      page_ptr + kScaleSectionOffset + in_page * kScaleBytes;

  if (threadIdx.x < kNopeDim / 2) {
    const int element = static_cast<int>(threadIdx.x) * 2;
    const float scale = decode_ue8m0(scale_ptr[element / kQuantBlock]);
    out_row[element] = __float2bfloat16_rn(decode_e4m3fn(data_ptr[element]) * scale);
    out_row[element + 1] =
        __float2bfloat16_rn(decode_e4m3fn(data_ptr[element + 1]) * scale);
  } else {
    // The final warp copies the 64 BF16 RoPE values bit-for-bit.
    const int pair = static_cast<int>(threadIdx.x) - kNopeDim / 2;
    reinterpret_cast<uint32_t*>(out_row + kNopeDim)[pair] =
        reinterpret_cast<const uint32_t*>(data_ptr + kNopeDim)[pair];
  }

  if (threadIdx.x == 0) {
    *row_status = kOk;
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the SWA device");
}

}  // namespace

void gather_target_kv_cuda(
    const torch::Tensor& swa0,
    const torch::Tensor& swa1,
    const torch::Tensor& swa2,
    const torch::Tensor& req_to_token,
    const torch::Tensor& full_to_swa,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& seq_lens,
    const c10::optional<torch::Tensor>& expected_pos,
    const torch::Tensor& out,
    const torch::Tensor& status,
    bool allow_zero_locations) {
  const torch::Tensor buffers[] = {swa0, swa1, swa2};
  for (int layer = 0; layer < kLayers; ++layer) {
    const auto& buffer = buffers[layer];
    check_cuda_contiguous(buffer, "SWA buffer");
    check_same_device(buffer, swa0, "SWA buffer");
    TORCH_CHECK(buffer.scalar_type() == torch::kUInt8, "SWA buffers must be uint8");
    TORCH_CHECK(
        buffer.dim() == 2 && buffer.size(0) == swa0.size(0)
            && buffer.size(1) == kPageBytes,
        "each SWA buffer must have shape [P,149760] with a shared P");
  }
  TORCH_CHECK(swa0.size(0) > 0, "SWA buffers must contain at least one page");

  check_cuda_contiguous(req_to_token, "req_to_token");
  check_same_device(req_to_token, swa0, "req_to_token");
  TORCH_CHECK(
      req_to_token.scalar_type() == torch::kInt32 && req_to_token.dim() == 2,
      "req_to_token must be int32 [R,C]");
  TORCH_CHECK(
      req_to_token.size(0) > 0 && req_to_token.size(1) > 0,
      "req_to_token dimensions must be positive");

  check_cuda_contiguous(full_to_swa, "full_to_swa");
  check_same_device(full_to_swa, swa0, "full_to_swa");
  TORCH_CHECK(
      full_to_swa.scalar_type() == torch::kInt64 && full_to_swa.dim() == 1
          && full_to_swa.numel() > 1,
      "full_to_swa must be int64 [F] with F > 1");

  check_cuda_contiguous(req_pool_indices, "req_pool_indices");
  check_same_device(req_pool_indices, swa0, "req_pool_indices");
  TORCH_CHECK(
      req_pool_indices.scalar_type() == torch::kInt64 && req_pool_indices.dim() == 1
          && req_pool_indices.numel() == 1,
      "req_pool_indices must be int64 [1]");

  check_cuda_contiguous(seq_lens, "seq_lens");
  check_same_device(seq_lens, swa0, "seq_lens");
  TORCH_CHECK(
      seq_lens.scalar_type() == torch::kInt64 && seq_lens.dim() == 1
          && seq_lens.numel() == 1,
      "seq_lens must be int64 [1]");

  const int64_t* expected_pos_ptr = nullptr;
  if (expected_pos.has_value()) {
    const torch::Tensor& expected = expected_pos.value();
    check_cuda_contiguous(expected, "expected_pos");
    check_same_device(expected, swa0, "expected_pos");
    TORCH_CHECK(
        expected.scalar_type() == torch::kInt64 && expected.dim() == 1
            && expected.numel() == 1,
        "expected_pos must be int64 [1]");
    expected_pos_ptr = expected.data_ptr<int64_t>();
  }

  check_cuda_contiguous(out, "out");
  check_same_device(out, swa0, "out");
  TORCH_CHECK(
      out.scalar_type() == torch::kBFloat16 && out.dim() == 4 && out.size(0) == kLayers
          && out.size(1) == 1 && out.size(2) == kRingRows && out.size(3) == kHeadDim,
      "out must be BF16 [3,1,128,512]");

  check_cuda_contiguous(status, "status");
  check_same_device(status, swa0, "status");
  TORCH_CHECK(
      status.scalar_type() == torch::kInt32 && status.dim() == 2
          && status.size(0) == kLayers && status.size(1) == kRingRows,
      "status must be int32 [3,128]");

  const c10::cuda::CUDAGuard device_guard(swa0.device());
  const int device = swa0.get_device();
  int major = 0;
  int minor = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  TORCH_CHECK(
      major == 10 && (minor == 0 || minor == 3),
      "target-KV gather requires Blackwell sm_100 or sm_103");

  const auto stream = at::cuda::getCurrentCUDAStream(device);
  gather_target_kv_kernel<<<dim3(kLayers, kRingRows), 256, 0, stream.stream()>>>(
      swa0.data_ptr<uint8_t>(),
      swa1.data_ptr<uint8_t>(),
      swa2.data_ptr<uint8_t>(),
      swa0.size(0),
      req_to_token.data_ptr<int32_t>(),
      req_to_token.size(0),
      req_to_token.size(1),
      full_to_swa.data_ptr<int64_t>(),
      full_to_swa.numel(),
      req_pool_indices.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      expected_pos_ptr,
      allow_zero_locations,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
