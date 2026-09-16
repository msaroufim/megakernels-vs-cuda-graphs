// Build-time serving batch for the DeepSeek-V4 DSpark megakernel.
//
// Batch is a BUILD parameter, not a launch parameter: the generated phase
// program (work-unit counts, claim chunks, workspace offsets) is specialized
// per batch by scripts/megakernel/generate_v4_header.py, and the device
// bodies pick their row counts from the constant below.
//
// Mapping (R3 batched-serving campaign):
//   - WEIGHT-BOUND bands (main projection 50 MiB, LM head 1.06 GiB,
//     markov_w2 66 MiB x 5 steps, main-KV projection) keep a BATCH-CONSTANT
//     item count and carry the batch's rows on the tensor-core N mode, so
//     their weight stream is read exactly ONCE per proposal no matter how
//     many sequences ride it. This is the measured amortization (R3:
//     step(N) = 310.7 + 1.465*N us on the LM band, 96.4% N-invariant).
//   - Every other band tiles OVER BATCH: one item is one (batch element,
//     ...) pair. Their weights are small (<= 34 MiB) or, for the routed
//     experts, grow with the batch's route table for reasons intrinsic to
//     MoE rather than to this mapping.
//
// The default is 1, so every translation unit that does not define
// DSPARK_V4_BATCH (the contract kernel, the phase microbenchmarks) compiles
// exactly the pre-batch code.
#pragma once

#ifndef DSPARK_V4_BATCH
#define DSPARK_V4_BATCH 1
#endif

// Dynamic shared memory per CTA. 139,520 B was never a hardware wall: it is
// the value the first SMEM-bound body happened to need, and every later body
// was sized against it. Blackwell's opt-in ceiling
// (cudaDevAttrMaxSharedMemoryPerBlockOptin, requested through
// cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize), which
// the launch path already calls) is far higher, and the megakernel's 254
// registers x 256 threads ALREADY pin it to one CTA per SM -- so extra
// shared memory costs no occupancy. It is not free, though: the SMEM
// carveout is taken out of the same 256 KB unified block as L1, so raising
// this shrinks L1. Treat any increase as a hypothesis to be measured on the
// batch curve, never as a given.
//
// The default reproduces the shipped literal exactly, so every translation
// unit that does not define DSPARK_V4_DYNAMIC_SMEM_BYTES -- the contract
// kernel included -- is byte-identical to the pre-change build.
#ifndef DSPARK_V4_DYNAMIC_SMEM_BYTES
#define DSPARK_V4_DYNAMIC_SMEM_BYTES 139520
#endif

// R15 register-ceiling probe. The scheduler kernel is declared
// `__launch_bounds__(<this>, 1)`, and that maximum-thread promise is what
// ptxas turns into the per-thread register ceiling: 65536 / maxThreads,
// clamped to 255. It is an ENTRY-SPECIFIC value and it overrides a global
// -maxrregcount, so the ONLY way to ask "what would this kernel look like
// with SGLang's 128-register budget?" is to move this number.
//
// SGLang's routed FP4 kernel runs 512 threads x 128 registers; the
// megakernel runs 256 x 255. Both are one 64 Ki register file per CTA --
// the difference is purely how many warps that budget is spread over. This
// macro lets the ceiling be swept on CPU, against source that is otherwise
// byte-for-byte the shipped kernel, before any warp-role rework is done:
//   256 -> 255 registers (shipped)   384 -> 170
//   320 -> 204                       448 -> 146      512 -> 128
// Raising it above generated::kThreads is a promise about the MAXIMUM block
// size, never a change to the launch, so the probe stays a pure register
// experiment. The default is generated::kThreads itself, so every
// translation unit that does not define it is byte-identical to the
// pre-change build.
#ifndef DSPARK_V4_LAUNCH_MAX_THREADS
#define DSPARK_V4_LAUNCH_MAX_THREADS 0
#endif

namespace dspark_batch {

constexpr int kBatch = DSPARK_V4_BATCH;
constexpr int kDynamicSharedBytes = DSPARK_V4_DYNAMIC_SMEM_BYTES;
static_assert(kDynamicSharedBytes >= 139520,
              "lowering the SMEM budget would break the sized bodies");
constexpr int kBlock = 5;
// Total activation rows in flight: the workspace is (batch, block, ...)
// contiguous, so a flat row index walks the batch in order and every
// row-indexed body stays correct by construction.
constexpr int kRows = kBatch * kBlock;
// N mode of an N-WIDENED band. tcgen05's f16/f8 N mode must be a multiple of
// 8, so batch 4 (20 rows) runs a padded atom: the pad columns re-read the
// last real row (never out of bounds) and are simply not stored, exactly as
// the frozen bodies pad 5 -> 8.
constexpr int kWideRows = (kRows + 7) / 8 * 8;

static_assert(kBatch == 1 || kBatch == 2 || kBatch == 4 || kBatch == 8,
              "generated headers exist for batch 1, 2, 4 and 8");

}  // namespace dspark_batch
