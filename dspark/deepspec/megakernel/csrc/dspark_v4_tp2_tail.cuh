// Static-TP2 serving-only tail protocol shared by the rank-0 DSpark
// scheduler and the prearmed rank-1 cooperative helper.
//
// This file is inert unless the extension is compiled with
// DSPARK_V4_STATIC_TP2_TAIL.  The protocol deliberately exchanges only the
// five normalized head rows, one 256-BF16 Markov embedding per step, and one
// packed maximum per step.  Full-vocabulary logits never cross GPUs.
#pragma once

#if defined(DSPARK_V4_STATIC_TP2_TAIL)

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace dspark_v4_tp2 {

constexpr uint64_t kLayoutMagic = 0x445350545032544cULL;  // "DSPTP2TL"
// v4 adds the symmetric math-TP all-reduce exchange region (both ranks now
// carry bulk payload; v3 gave bulk payload to rank 1 only).
constexpr int64_t kLayoutVersion = 4;
constexpr int kParitySlots = 2;
constexpr int kDraftBlock = 5;
// One packed serving proposal: [status | anchor | 5 token ids | 5
// confidences] in exact fp32 (vocab 129280 < 2^24).  The mailbox in rank 0's
// control block replaces the per-step NCCL payload broadcast.
constexpr int kPayloadFloats = 12;
constexpr int kHidden = 4096;
constexpr int kMarkovRank = 256;
constexpr int kVocab = 129280;
constexpr int kInputStatusWords = 3 * 128;
constexpr int kContractOffsetSlots = 4724;
constexpr int kTotalTiles = kVocab / 128;
constexpr int kRank0Tiles = kTotalTiles * 2 / 5;
constexpr int kRank0Vocab = kRank0Tiles * 128;
constexpr int kRank1TileBegin = kRank0Tiles;
constexpr int kRank1TileEnd = kTotalTiles;
constexpr int kRank1Tiles = kRank1TileEnd - kRank1TileBegin;
constexpr int kRank1Vocab = kVocab - kRank0Vocab;
static_assert(kRank0Tiles == 404);
static_assert(kRank1Tiles == 606);
constexpr int kThreads = 256;
constexpr int kGrid = 148;
constexpr int kDynamicSharedBytes = 139520;
constexpr uint64_t kWatchdogTimeoutNs = 5ULL * 1000ULL * 1000ULL * 1000ULL;
constexpr size_t kExportGranularity = 2ULL * 1024ULL * 1024ULL;
constexpr size_t kControlBytes = 4096;
constexpr size_t kPayloadOffset = kExportGranularity;
constexpr size_t kAnchorOffset = kPayloadOffset;
constexpr size_t kAnchorBytes =
    kParitySlots * sizeof(int32_t);
constexpr size_t kNormalizedOffset = kAnchorOffset + 128;
constexpr size_t kNormalizedBytes =
    kParitySlots * kDraftBlock * kHidden * sizeof(__nv_bfloat16);
constexpr size_t kMarkovEmbeddingOffset =
    (kNormalizedOffset + kNormalizedBytes + 127) & ~size_t{127};
constexpr size_t kMarkovEmbeddingBytes =
    kParitySlots * kDraftBlock * kMarkovRank * sizeof(__nv_bfloat16);
constexpr size_t kSelectedTokenOffset =
    (kMarkovEmbeddingOffset + kMarkovEmbeddingBytes + 127) & ~size_t{127};
constexpr size_t kSelectedTokenBytes =
    kParitySlots * kDraftBlock * sizeof(int32_t);
constexpr size_t kRank1PayloadEnd = kSelectedTokenOffset + kSelectedTokenBytes;
constexpr size_t kRank1ScratchBytes =
    kParitySlots * kDraftBlock * kVocab * sizeof(float);

// ---------------------------------------------------------------------------
// Layout v4: the symmetric two-rank all-reduce exchange region.
//
// Math-level tensor parallelism splits a row-parallel GEMM's K axis across the
// two ranks, so each rank owns a PARTIAL 5 x 4096 FP32 result that must be
// summed with the peer's.  Both ranks therefore need bulk peer-visible
// payload, which v3 gave to rank 1 only.  The region sits at one FIXED offset
// on BOTH ranks (rank 0 simply leaves a hole where rank 1's tail payload
// lives) so the device code computes the same pointer arithmetic on either
// side and the descriptor validation stays canonical.
//
// One slot per (parity, layer).  Parity doubling is what keeps epoch N+1 from
// overwriting an unconsumed epoch N; the layer axis keeps the three per-layer
// exchanges of one proposal from aliasing each other.
// ---------------------------------------------------------------------------
constexpr int kAllReduceLayers = 3;
constexpr int kAllReduceRows = kDraftBlock;
constexpr int kAllReduceWidth = kHidden;
constexpr int kAllReduceFloats = kAllReduceRows * kAllReduceWidth;  // 20,480
constexpr int kAllReduceSlots = kParitySlots * kAllReduceLayers;    // 6
constexpr size_t kAllReduceOffset = (kRank1PayloadEnd + 127) & ~size_t{127};
constexpr size_t kAllReduceBytes =
    static_cast<size_t>(kAllReduceSlots) * kAllReduceFloats * sizeof(float);
constexpr size_t kOwnerPayloadEnd = kAllReduceOffset + kAllReduceBytes;
// The publish fan-in is per 128-column output tile of the 4096-wide result;
// the last tile to arrive release-publishes the slot's exact ticket.
constexpr int kAllReduceTileWidth = 128;
constexpr int kAllReduceTiles = kAllReduceWidth / kAllReduceTileWidth;  // 32
static_assert(kAllReduceWidth % kAllReduceTileWidth == 0);

enum StatusIndex : int {
  kStatusProtocol = 0,
  kStatusWatchdog = 1,
  kStatusStale = 2,
  kStatusFuture = 3,
  kStatusPayloadMismatch = 4,
  kStatusDone = 5,
  kStatusRank0Started = 6,
  kStatusInputRejected = 7,
};

enum ProtocolStatus : uint64_t {
  kProtocolOk = 0,
  kProtocolFutureTicket = 1,
  kProtocolWatchdog = 2,
  kProtocolPayloadMismatch = 3,
  kProtocolInputStatus = 4,
  kProtocolModeMismatch = 5,
};

enum AggregateIndex : int {
  kAggregateInputStatusRejects = 0,
  kAggregateProtocolErrors = 1,
  kAggregateWatchdogs = 2,
  kAggregateFutureRejects = 3,
  kAggregatePayloadMismatches = 4,
  kAggregateCount = 5,
};

enum WaitIndex : int {
  kWaitPrearm = 0,
  kWaitParityReuse = 1,
  kWaitRequest = 2,
  kWaitNormalized = 3,
  kWaitEmbeddingBase = 4,       // + step
  kWaitSelectedBase = 9,        // + step
  kWaitPackedBase = 14,         // + step
  kWaitFinalConsumed = 19,
  kWaitHelperDone = 20,
  kWaitPayload = 21,
  kWaitAllReduceBase = 24,  // + layer
  kWaitCount = 28,
};

struct alignas(64) ControlBlock {
  uint64_t magic;
  uint64_t version;
  uint64_t session_nonce;
  uint64_t owner_rank;
  uint64_t serving_mode;

  alignas(16) uint64_t prearm_ready[kParitySlots];
  alignas(16) uint64_t request_start[kParitySlots];
  alignas(16) uint64_t request_started[kParitySlots];
  alignas(16) uint64_t normalized_ready[kParitySlots];
  alignas(16) uint64_t normalized_consumed[kParitySlots];
  alignas(16) uint64_t embedding_ready[kParitySlots][kDraftBlock];
  alignas(16) uint64_t selected_ready[kParitySlots][kDraftBlock];
  alignas(16) uint64_t packed_ready[kParitySlots][kDraftBlock];
  alignas(16) uint64_t result_consumed[kParitySlots];
  alignas(16) uint64_t helper_done[kParitySlots];

  alignas(16) uint64_t rank1_packed[kParitySlots][kDraftBlock];
  alignas(16) uint64_t remote_packed[kParitySlots][kDraftBlock];
  alignas(16) uint64_t rank0_local_packed[kParitySlots][kDraftBlock];
  alignas(16) uint64_t selected_tokens[kParitySlots][kDraftBlock];

  alignas(16) uint64_t normalized_rows_done[kParitySlots];
  alignas(16) uint64_t status[kParitySlots][8];
  alignas(16) uint64_t aggregate[kAggregateCount];
  alignas(16) uint64_t waits[kParitySlots][kWaitCount];
  alignas(16) uint64_t request_stamp[kParitySlots];
  alignas(16) uint64_t normalized_stamp[kParitySlots];
  alignas(16) uint64_t lm_done_stamp[kParitySlots];
  alignas(16) uint64_t embedding_stamp[kParitySlots][kDraftBlock];
  alignas(16) uint64_t packed_stamp[kParitySlots][kDraftBlock];
  alignas(16) uint64_t selected_stamp[kParitySlots][kDraftBlock];
  alignas(16) uint64_t done_stamp[kParitySlots];

  // Serving payload mailbox.  Rank 0 writes its per-parity 12-float packed
  // proposal here (its own peer-visible owner allocation) and then
  // release-publishes the exact epoch ticket; rank 1 acquire-waits on the
  // ticket across the CUDA-IPC mapping and copies the floats out.  This is
  // the hot-path transport for both the success payload and the fatal
  // status=-1 sentinel.
  alignas(16) float payload_data[kParitySlots][kPayloadFloats];
  alignas(16) uint64_t payload_ready[kParitySlots];

  // Layout v4 math-TP all-reduce protocol state, one entry per
  // (parity, layer).  `partial_tiles_done` is a LOCAL fan-in counter (the
  // publishing rank owns it); `partial_ready` is the peer-written exact
  // ticket the consuming rank acquire-waits on; `partial_consumed` closes the
  // slot so the parity handshake can prove the previous epoch is retired.
  alignas(16) uint32_t partial_tile_parts[kParitySlots][kAllReduceLayers]
                                         [kAllReduceTiles];
  alignas(16) uint64_t partial_tiles_done[kParitySlots][kAllReduceLayers];
  alignas(16) uint64_t partial_ready[kParitySlots][kAllReduceLayers];
  alignas(16) uint64_t partial_consumed[kParitySlots][kAllReduceLayers];
  alignas(16) uint64_t partial_publish_stamp[kParitySlots][kAllReduceLayers];
  alignas(16) uint64_t partial_consume_stamp[kParitySlots][kAllReduceLayers];
  alignas(16) uint64_t epoch_done[kParitySlots];
};
static_assert(sizeof(ControlBlock) <= kControlBytes);
static_assert(alignof(ControlBlock) >= alignof(uint64_t));

struct DeviceContext {
  ControlBlock* owner_control = nullptr;
  ControlBlock* peer_control = nullptr;
  __nv_bfloat16* peer_normalized = nullptr;
  __nv_bfloat16* peer_markov_embeddings = nullptr;
  int32_t* peer_anchor = nullptr;
  int32_t* peer_selected_tokens = nullptr;
  const int32_t* input_status = nullptr;
  // Layout v4: `owner_partials` is where the PEER deposits its partials (this
  // rank reads them); `peer_partials` is where THIS rank deposits its own.
  // Both are the same fixed offset inside the two owner allocations.
  float* owner_partials = nullptr;
  float* peer_partials = nullptr;
  uint64_t ticket = 0;
  uint32_t generation = 0;
  uint32_t epoch = 0;
  uint32_t previous_epoch = 0;
  int parity = 0;
  int rank = 0;
  bool enabled = false;
  bool serving = false;
  // Set only by the math-TP build: both ranks run the full scheduler and the
  // per-layer row-parallel all-reduce replaces the tail-shard exchange.
  bool math_tp = false;
};

__host__ __device__ __forceinline__ uint64_t make_ticket(
    uint32_t generation,
    uint32_t epoch) {
  return (static_cast<uint64_t>(generation) << 32) | epoch;
}

__host__ __device__ __forceinline__ uint32_t ticket_epoch(uint64_t ticket) {
  return static_cast<uint32_t>(ticket);
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint64_t acquire_system(uint64_t* address) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> atomic(*address);
  return atomic.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void release_system(
    uint64_t* address,
    uint64_t value) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> atomic(*address);
  atomic.store(value, cuda::memory_order_release);
}

__device__ __forceinline__ void set_protocol_error(
    ControlBlock* control,
    int parity,
    ProtocolStatus value) {
  const unsigned long long prior = atomicCAS(
      reinterpret_cast<unsigned long long*>(
          &control->status[parity][kStatusProtocol]),
      static_cast<unsigned long long>(kProtocolOk),
      static_cast<unsigned long long>(value));
  if (prior == static_cast<unsigned long long>(kProtocolOk)) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(
            &control->aggregate[kAggregateProtocolErrors]),
        1ULL);
  }
}

__device__ inline bool wait_for_ticket(
    uint64_t* address,
    uint64_t expected,
    uint32_t generation,
    ControlBlock* report,
    int parity,
    int wait_index) {
  const uint64_t started = globaltimer_ns();
  bool counted_wait = false;
  bool counted_stale = false;
  while (true) {
    const uint64_t observed = acquire_system(address);
    if (observed == expected) {
      return true;
    }
    if (!counted_wait && wait_index >= 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(
              &report->waits[parity][wait_index]),
          1ULL);
      counted_wait = true;
    }
    if (observed == 0) {
      if (!counted_stale) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                &report->status[parity][kStatusStale]),
            1ULL);
        counted_stale = true;
      }
    } else {
      const uint32_t observed_generation =
          static_cast<uint32_t>(observed >> 32);
      const uint32_t observed_epoch = ticket_epoch(observed);
      const uint32_t expected_epoch = ticket_epoch(expected);
      if (observed_generation != generation || observed_epoch > expected_epoch) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                &report->status[parity][kStatusFuture]),
            1ULL);
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                &report->aggregate[kAggregateFutureRejects]),
            1ULL);
        set_protocol_error(report, parity, kProtocolFutureTicket);
        return false;
      }
      if (!counted_stale && observed_epoch < expected_epoch) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                &report->status[parity][kStatusStale]),
            1ULL);
        counted_stale = true;
      }
    }
    if (globaltimer_ns() - started > kWatchdogTimeoutNs) {
      report->status[parity][kStatusWatchdog] = 1;
      atomicAdd(
          reinterpret_cast<unsigned long long*>(
              &report->aggregate[kAggregateWatchdogs]),
          1ULL);
      set_protocol_error(report, parity, kProtocolWatchdog);
      return false;
    }
  }
}

__device__ __forceinline__ bool protocol_ok(
    const DeviceContext& context) {
  return context.owner_control->status[context.parity][kStatusProtocol]
      == kProtocolOk;
}

// Called by every rank-0 persistent CTA before any scheduler root is
// published.  Worker 0 performs the cross-rank handshake; all other workers
// wait on the local exact-ticket release, avoiding any unsupported grid sync.
__device__ inline void rank0_begin_audited(
    const DeviceContext& context,
    const int32_t* anchor,
    int worker) {
  if (threadIdx.x == 0 && worker == 0) {
    ControlBlock* control = context.owner_control;
    ControlBlock* peer = context.peer_control;
    const int parity = context.parity;
    const uint64_t previous = context.previous_epoch == 0
        ? 0
        : make_ticket(context.generation, context.previous_epoch);
    bool ok = wait_for_ticket(
        &peer->helper_done[parity], previous, context.generation,
        control, parity, kWaitParityReuse);
    if (ok) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        control->status[parity][index] = 0;
      }
#pragma unroll
      for (int index = 0; index < kWaitCount; ++index) {
        control->waits[parity][index] = 0;
      }
      control->normalized_rows_done[parity] = 0;
      ok = wait_for_ticket(
          &peer->prearm_ready[parity], context.ticket, context.generation,
          control, parity, kWaitPrearm);
    }
    if (ok) {
      control->request_stamp[parity] = globaltimer_ns();
      context.peer_anchor[parity] = anchor[0];
      __threadfence_system();
      release_system(&peer->request_start[parity], context.ticket);
      control->status[parity][kStatusRank0Started] = 1;
    }
    // Release followers even on failure so the rank-0 kernel retires and the
    // audit can report the fail-closed protocol status.
    release_system(&control->request_started[parity], context.ticket);
  }
  __syncthreads();
  if (threadIdx.x == 0 && worker != 0) {
    // Worker 0 owns the bounded remote waits and always releases this local
    // ticket on success or failure. Followers must not race it with an
    // independent timeout whose error could be cleared by a late success.
    while (acquire_system(
               &context.owner_control->request_started[context.parity])
        != context.ticket) {
    }
  }
  __syncthreads();
}

// Serving mode keeps all lifecycle progress on the two bound device streams.
// Rank 0 consumes the producer's [3, 128] status only after observing rank 1's
// exact prearm.  A rejected input is delivered to rank 1 as an exact request
// carrying a terminal protocol status, so the helper retires immediately
// instead of waiting for its watchdog.
__device__ inline bool rank0_begin_serving(
    const DeviceContext& context,
    const int32_t* anchor,
    int worker,
    bool local_input_valid) {
  __shared__ int prearm_ok;
  __shared__ int input_failures;
  __shared__ int mode_mismatch;
  if (threadIdx.x == 0) {
    prearm_ok = 0;
    input_failures = local_input_valid ? 0 : 1;
    mode_mismatch = 0;
  }
  __syncthreads();

  if (worker == 0) {
    ControlBlock* control = context.owner_control;
    ControlBlock* peer = context.peer_control;
    const int parity = context.parity;
    if (threadIdx.x == 0) {
      const uint64_t previous = context.previous_epoch == 0
          ? 0
          : make_ticket(context.generation, context.previous_epoch);
      bool ok = wait_for_ticket(
          &peer->helper_done[parity], previous, context.generation,
          control, parity, kWaitParityReuse);
      if (ok) {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          control->status[parity][index] = 0;
        }
#pragma unroll
        for (int index = 0; index < kWaitCount; ++index) {
          control->waits[parity][index] = 0;
        }
        control->normalized_rows_done[parity] = 0;
        ok = wait_for_ticket(
            &peer->prearm_ready[parity], context.ticket, context.generation,
            control, parity, kWaitPrearm);
      }
      if (ok) {
        prearm_ok = 1;
        if (control->serving_mode != 1 || peer->serving_mode != 1) {
          mode_mismatch = 1;
        }
      }
    }
    __syncthreads();

    if (prearm_ok && !mode_mismatch) {
      for (int index = threadIdx.x; index < kInputStatusWords;
           index += blockDim.x) {
        if (context.input_status[index] != 0) {
          atomicAdd(&input_failures, 1);
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0 && prearm_ok) {
      control->request_stamp[parity] = globaltimer_ns();
      if (mode_mismatch || input_failures != 0) {
        const ProtocolStatus failure = mode_mismatch
            ? kProtocolModeMismatch
            : kProtocolInputStatus;
        set_protocol_error(control, parity, failure);
        control->status[parity][kStatusInputRejected] =
            static_cast<uint64_t>(input_failures);
        if (failure == kProtocolInputStatus) {
          atomicAdd(
              reinterpret_cast<unsigned long long*>(
                  &control->aggregate[kAggregateInputStatusRejects]),
              1ULL);
        }
        __threadfence_system();
        release_system(&peer->request_start[parity], context.ticket);
        // The exact request wakes rank 1's already-resident helper, which
        // observes the terminal status and release-publishes helper_done.
        (void)wait_for_ticket(
            &peer->helper_done[parity], context.ticket, context.generation,
            control, parity, kWaitHelperDone);
      } else {
        context.peer_anchor[parity] = anchor[0];
        __threadfence_system();
        release_system(&peer->request_start[parity], context.ticket);
        control->status[parity][kStatusRank0Started] = 1;
      }
      release_system(&control->request_started[parity], context.ticket);
    } else if (threadIdx.x == 0 && !prearm_ok) {
      release_system(&control->request_started[parity], context.ticket);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && worker != 0) {
    while (acquire_system(
               &context.owner_control->request_started[context.parity])
        != context.ticket) {
    }
  }
  __syncthreads();
  return protocol_ok(context);
}

__device__ inline bool rank0_begin(
    const DeviceContext& context,
    const int32_t* anchor,
    int worker,
    bool local_input_valid) {
  if (!context.enabled) {
    return true;
  }
  if (context.serving) {
    return rank0_begin_serving(context, anchor, worker, local_input_valid);
  }
  rank0_begin_audited(context, anchor, worker);
  return true;
}

// The five useful head-reduce work items each own one complete normalized
// row.  Their CTAs copy those rows directly into rank 1's parity payload;
// the fifth fenced CTA release-publishes the exact ticket.
__device__ inline void rank0_publish_normalized_row(
    const DeviceContext& context,
    const __nv_bfloat16* normalized,
    int row) {
  if (!context.enabled || !protocol_ok(context)) {
    return;
  }
  __nv_bfloat16* destination = context.peer_normalized
      + (static_cast<int64_t>(context.parity) * kDraftBlock + row) * kHidden;
  const __nv_bfloat16* source = normalized + row * kHidden;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    destination[column] = source[column];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    const uint64_t prior = atomicAdd(
        reinterpret_cast<unsigned long long*>(
            &context.owner_control->normalized_rows_done[context.parity]),
        1ULL);
    if (prior + 1 == kDraftBlock) {
      context.owner_control->normalized_stamp[context.parity] =
          globaltimer_ns();
      release_system(
          &context.peer_control->normalized_ready[context.parity],
          context.ticket);
    }
  }
}

__device__ inline void rank0_publish_markov_embedding(
    const DeviceContext& context,
    const __nv_bfloat16* embedding,
    int step) {
  if (!context.enabled || !protocol_ok(context)) {
    return;
  }
  __nv_bfloat16* destination = context.peer_markov_embeddings
      + (static_cast<int64_t>(context.parity) * kDraftBlock + step)
          * kMarkovRank;
  destination[threadIdx.x] = embedding[step * kMarkovRank + threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    context.owner_control->embedding_stamp[context.parity][step] =
        globaltimer_ns();
    release_system(
        &context.peer_control->embedding_ready[context.parity][step],
        context.ticket);
  }
}

// Called only by the final rank-0 CTA completing a Markov-W2 phase.  It
// acquire-joins rank 1's upper-shard packed maximum, performs the order-free
// max merge (the complemented low word preserves lowest-ID ties), and
// publishes the selected token for the next step.
__device__ inline bool rank0_merge_step(
    const DeviceContext& context,
    uint64_t local_packed,
    int step,
    int32_t* selected_token) {
  if (!context.enabled || !protocol_ok(context)) {
    return false;
  }
  ControlBlock* control = context.owner_control;
  ControlBlock* peer = context.peer_control;
  const int parity = context.parity;
  if (!wait_for_ticket(
          &control->packed_ready[parity][step], context.ticket,
          context.generation, control, parity, kWaitPackedBase + step)) {
    return false;
  }
  const uint64_t remote = control->remote_packed[parity][step];
  const uint64_t merged = remote > local_packed ? remote : local_packed;
  const int32_t token = static_cast<int32_t>(
      0xFFFFFFFFu - static_cast<uint32_t>(merged));
  control->rank0_local_packed[parity][step] = local_packed;
  control->selected_tokens[parity][step] = static_cast<uint64_t>(token);
  *selected_token = token;
  context.peer_selected_tokens[parity * kDraftBlock + step] = token;
  __threadfence_system();
  control->selected_stamp[parity][step] = globaltimer_ns();
  release_system(&peer->selected_ready[parity][step], context.ticket);
  if (step == kDraftBlock - 1) {
    release_system(&peer->result_consumed[parity], context.ticket);
    if (!wait_for_ticket(
            &peer->helper_done[parity], context.ticket, context.generation,
            control, parity, kWaitHelperDone)) {
      return false;
    }
    control->done_stamp[parity] = globaltimer_ns();
    control->status[parity][kStatusDone] = 1;
  }
  return true;
}

// ---------------------------------------------------------------------------
// The two-rank all-reduce (layout v4).
//
// With exactly two ranks an all-reduce is: each rank deposits its partial in
// the peer's exchange slot, both ranks acquire-join the peer's exact ticket,
// and both then sum the SAME two operands in the SAME fixed order -- rank 0's
// partial first, rank 1's second.  Both ranks therefore end the exchange
// holding a bitwise-identical result produced by a bitwise-identical
// accumulation order, so every downstream audit, sentinel and watchdog keeps
// working: the ONLY new numerical difference is against a single-GPU control,
// which is inherent to any row-parallel split and is why this build is gated
// on drift + acceptance rather than bitwise exactness.
//
// Deliberately NOT done: letting each rank sum in its own arrival order.  That
// would be one instruction cheaper and would silently desynchronise the two
// ranks' downstream state.
//
// Protocol, per (parity, layer) slot:
//   producer  store_element* -> commit_tile (per 128-column tile, fenced,
//             local fan-in) -> the 32nd tile release-publishes the peer's
//             exact (generation<<32|epoch) ticket
//   consumer  wait (acquire, 5 s watchdog, sticky protocol error, future- and
//             stale-ticket accounting) -> read -> combine -> release_slot
// The producer never waits, so a phase may publish and a LATER phase consume
// without any circular dependency between the ranks.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int allreduce_slot(
    const DeviceContext& context,
    int layer) {
  return context.parity * kAllReduceLayers + layer;
}

__device__ __forceinline__ float* allreduce_peer_slot(
    const DeviceContext& context,
    int layer) {
  return context.peer_partials
      + static_cast<int64_t>(allreduce_slot(context, layer)) * kAllReduceFloats;
}

__device__ __forceinline__ const float* allreduce_owner_slot(
    const DeviceContext& context,
    int layer) {
  return context.owner_partials
      + static_cast<int64_t>(allreduce_slot(context, layer)) * kAllReduceFloats;
}

// One element of this rank's partial into the peer's slot.  No fence: the
// caller batches its stores and then calls commit_tile exactly once per tile.
__device__ __forceinline__ void allreduce_store(
    const DeviceContext& context,
    int layer,
    int row,
    int column,
    float value) {
  allreduce_peer_slot(context, layer)[row * kAllReduceWidth + column] = value;
}

// Sub-fan-in for producers whose work items do not map one-to-one onto the
// 128-column publish tiles (wo_b splits each tile across kParts K-chunks).
// Returns true for the caller that observed the last part, i.e. the one that
// must sum the splits and publish the tile.
__device__ __forceinline__ bool allreduce_tile_part_last(
    const DeviceContext& context,
    int layer,
    int tile,
    uint32_t parts) {
  const uint32_t prior = atomicAdd(
      &context.owner_control->partial_tile_parts[context.parity][layer][tile],
      1U);
  return prior + 1U == parts;
}

// Publishes one 128-column tile: fences this CTA's peer stores, then counts
// the tile into the slot's local fan-in.  The last tile release-publishes the
// exact ticket into the PEER's control block.  Must be called by exactly one
// thread per tile, after that tile's stores are visible to this thread.
__device__ __forceinline__ void allreduce_commit_tile(
    const DeviceContext& context,
    int layer) {
  __threadfence_system();
  const uint64_t prior = atomicAdd(
      reinterpret_cast<unsigned long long*>(
          &context.owner_control->partial_tiles_done[context.parity][layer]),
      1ULL);
  if (prior + 1 == kAllReduceTiles) {
    context.owner_control->partial_publish_stamp[context.parity][layer] =
        globaltimer_ns();
    release_system(
        &context.peer_control->partial_ready[context.parity][layer],
        context.ticket);
  }
}

// Acquire-join the peer's partial for this layer.  Every consuming CTA calls
// this; they all spin on the same word, so the cost is one acquire load per
// CTA once the publish has landed.
__device__ __forceinline__ bool allreduce_wait(
    const DeviceContext& context,
    int layer) {
  if (!protocol_ok(context)) {
    return false;
  }
  return wait_for_ticket(
      &context.owner_control->partial_ready[context.parity][layer],
      context.ticket,
      context.generation,
      context.owner_control,
      context.parity,
      kWaitAllReduceBase + layer);
}

__device__ __forceinline__ float allreduce_peer_value(
    const DeviceContext& context,
    int layer,
    int row,
    int column) {
  return allreduce_owner_slot(context, layer)[row * kAllReduceWidth + column];
}

// THE fixed-order sum.  rank-0 partial first, rank-1 partial second, on BOTH
// ranks, so the two ranks cannot diverge from each other.
__device__ __forceinline__ float allreduce_combine(
    const DeviceContext& context,
    float local_partial,
    float peer_partial) {
  const float rank0 = context.rank == 0 ? local_partial : peer_partial;
  const float rank1 = context.rank == 0 ? peer_partial : local_partial;
  return rank0 + rank1;
}

// Closes the slot.  One caller per (parity, layer) -- the consuming phase's
// last completer -- so the next reuse of this parity can prove the previous
// epoch retired.
__device__ __forceinline__ void allreduce_release_slot(
    const DeviceContext& context,
    int layer) {
  context.owner_control->partial_consume_stamp[context.parity][layer] =
      globaltimer_ns();
  __threadfence_system();
  release_system(
      &context.peer_control->partial_consumed[context.parity][layer],
      context.ticket);
}

// Symmetric epoch entry for the math-TP build.  Both ranks run the same
// scheduler, so neither is "the helper": each proves the peer retired the
// epoch that last used this parity slot, resets its own slot state, and then
// release-publishes its own start.  Worker 0 owns the bounded remote wait and
// always releases the local ticket, exactly like rank0_begin_audited.
__device__ inline bool math_tp_begin(
    const DeviceContext& context,
    int worker) {
  if (!context.enabled || !context.math_tp) {
    return true;
  }
  if (threadIdx.x == 0 && worker == 0) {
    ControlBlock* control = context.owner_control;
    ControlBlock* peer = context.peer_control;
    const int parity = context.parity;
    const uint64_t previous = context.previous_epoch == 0
        ? 0
        : make_ticket(context.generation, context.previous_epoch);
    const bool ok = wait_for_ticket(
        &control->epoch_done[parity], previous, context.generation,
        control, parity, kWaitParityReuse);
    if (ok) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        control->status[parity][index] = 0;
      }
      for (int index = 0; index < kWaitCount; ++index) {
        control->waits[parity][index] = 0;
      }
      for (int layer = 0; layer < kAllReduceLayers; ++layer) {
        control->partial_tiles_done[parity][layer] = 0;
        for (int tile = 0; tile < kAllReduceTiles; ++tile) {
          control->partial_tile_parts[parity][layer][tile] = 0;
        }
      }
      control->request_stamp[parity] = globaltimer_ns();
      control->status[parity][kStatusRank0Started] = 1;
      __threadfence_system();
      release_system(&peer->request_start[parity], context.ticket);
    }
    release_system(&control->request_started[parity], context.ticket);
  }
  __syncthreads();
  if (threadIdx.x == 0 && worker != 0) {
    while (acquire_system(
               &context.owner_control->request_started[context.parity])
        != context.ticket) {
    }
  }
  __syncthreads();
  // The peer's start is NOT joined here: its partials carry their own exact
  // tickets, so a slow peer costs a wait at the first all-reduce rather than a
  // barrier at entry.  Joining here would serialise the two ranks' replicated
  // prefixes for no benefit.
  return protocol_ok(context);
}

// Symmetric epoch exit: publishes this rank's retirement so the peer can reuse
// the parity slot two epochs later.
__device__ inline void math_tp_end(const DeviceContext& context) {
  if (!context.enabled || !context.math_tp) {
    return;
  }
  context.owner_control->done_stamp[context.parity] = globaltimer_ns();
  context.owner_control->status[context.parity][kStatusDone] = 1;
  __threadfence_system();
  release_system(
      &context.peer_control->epoch_done[context.parity], context.ticket);
}

// Host bridge used only by dspark_v4_kernel.cu.  It validates the rank-0
// lifecycle, records the authoritative event before launch, and returns the
// imported relative-layout pointers carried as explicit kernel arguments.
DeviceContext prepare_rank0_launch(
    uint32_t epoch,
    cudaStream_t stream,
    const uint8_t* weight_arena,
    const int64_t* weight_offsets);
void record_rank0_launch_complete(uint32_t epoch, cudaStream_t stream);

}  // namespace dspark_v4_tp2

#endif  // DSPARK_V4_STATIC_TP2_TAIL
