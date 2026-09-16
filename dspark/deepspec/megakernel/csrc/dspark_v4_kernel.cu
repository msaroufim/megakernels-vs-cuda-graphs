#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <vector>

#ifdef DSPARK_V4_INTEGRATED_PROPOSAL_IO
#include "dspark_proposal_io.cuh"
#if !defined(DSPARK_V4_RELAXED_DAG) || !defined(DSPARK_V4_FULL_LOOP_DEVICE_EPOCH) \
    || defined(DSPARK_V4_STATIC_TP2_TAIL)
#error "integrated proposal IO requires the TP1 device-epoch scheduler"
#endif
#endif
// Outlining isolates phase register lifetimes from the persistent scheduler.
// Bits select experts, head/tail, and remaining phase families respectively.
#ifndef DSPARK_PHASE_OUTLINE
#define DSPARK_PHASE_OUTLINE 0
#endif
#if (DSPARK_PHASE_OUTLINE & 1)
#define DSPARK_EXPERT_ENTRY __noinline__
#else
#define DSPARK_EXPERT_ENTRY
#endif
#if (DSPARK_PHASE_OUTLINE & 2)
#define DSPARK_HEAD_ENTRY __noinline__
#else
#define DSPARK_HEAD_ENTRY
#endif
#if (DSPARK_PHASE_OUTLINE & 4)
#define DSPARK_OTHER_ENTRY __noinline__
#else
#define DSPARK_OTHER_ENTRY
#endif

// Relaxed-numerics DAG (run-notes relaxed-drafter/PLAN.md stage 2): a
// separate generated header fuses the tail's 7 phases/step to 4. Selected at
// compile time so the contract module is byte-identical without the define.
#if defined(DSPARK_V4_GREEDY_TAIL) && !defined(DSPARK_V4_RELAXED_DAG)
#error "DSPARK_V4_GREEDY_TAIL requires DSPARK_V4_RELAXED_DAG"
#endif
#if defined(DSPARK_V4_COMPILED_EXECUTION_MASK) \
    != defined(DSPARK_V4_COMPILED_DRAFT_LAYER_MASK)
#error "compiled execution and draft-layer masks must be defined together"
#endif
#if defined(DSPARK_V4_STATIC_TP2_TAIL) \
    && (!defined(DSPARK_V4_GREEDY_TAIL) \
        || !defined(DSPARK_V4_RELAXED_DAG))
#error "DSPARK_V4_STATIC_TP2_TAIL requires greedy + relaxed modes"
#endif
#include "dspark_hc_overlap.cuh"
#include "dspark_batch.h"

// Private source-only experiment: retain the exact confidence work in Direct2.
#ifdef DSPARK_PRIVATE_DIRECT2_CONFIDENCE_0908
#if DSPARK_PRIVATE_DIRECT2_CONFIDENCE_0908 != 1 \
    || !defined(DSPARK_CONFIDENCE_HEAD_PREFIX) || DSPARK_CONFIDENCE_HEAD_PREFIX != 1 \
    || !defined(DSPARK_V4_DIRECT_DEPENDENCIES) || DSPARK_V4_DIRECT_DEPENDENCIES != 2 \
    || !defined(DSPARK_V4_COMPLETION_RELEASE) || DSPARK_V4_COMPLETION_RELEASE != 1
#error "private confidence experiment requires prefix=1, direct=2, completion-release=1"
#endif
#endif

// This isolated opt-in reserves greedy-only softmax partial sums for four
// exact confidence prefixes. Every other scheduler/ABI configuration is excluded.
#ifdef DSPARK_CONFIDENCE_HEAD_PREFIX
#if !defined(DSPARK_V4_GREEDY_TAIL) || !defined(DSPARK_V4_RELAXED_DAG) \
    || !defined(DSPARK_V4_STATIC_QUEUES) || DSPARK_V4_BATCH != 1 \
    || !defined(DSPARK_V4_ROUTED_DYNAMIC_CLAIMS) \
    || !defined(DSPARK_V4_SHARED_STAGE_CONTEXT) \
    || !defined(DSPARK_SHARED_CONTINUE) || DSPARK_ROUTED_W2_WORKERS != 128 \
    || DSPARK_V4_COMPILED_EXECUTION_MASK != 2047 \
    || DSPARK_V4_COMPILED_DRAFT_LAYER_MASK != 7 \
    || DSPARK_LM_STREAMED_K != 256
#error "confidence prefix requires the frozen batch-one full-mask greedy scheduler"
#endif
#if defined(DSPARK_V4_STATIC_TP2_TAIL) || defined(DSPARK_V4_TP2_ABLATE) \
    || (defined(DSPARK_V4_DIRECT_DEPENDENCIES) && !defined(DSPARK_PRIVATE_DIRECT2_CONFIDENCE_0908)) \
    || (defined(DSPARK_V4_COMPLETION_RELEASE) && !defined(DSPARK_PRIVATE_DIRECT2_CONFIDENCE_0908)) \
    || defined(DSPARK_V4_QUEUE_LOOKAHEAD) || defined(DSPARK_ROUTED_GROUP_READY) \
    || defined(DSPARK_V4_ENABLE_DEVICE_TRACE) || defined(DSPARK_MARKOV_GATHER_CONTINUE)
#error "confidence prefix excludes alternate scheduling and trace variants"
#endif
#endif

#include "dspark_v4_tp2_ablate.cuh"
#if DSPARK_V4_BATCH != 1 && !defined(DSPARK_V4_GREEDY_TAIL)
#error "DSPARK_V4_BATCH > 1 requires the greedy relaxed build"
#endif
#if defined(DSPARK_V4_STATIC_TP2_TAIL)
#error "the legacy B200 static TP2 tail is not supported by the GB300 path"
#elif defined(DSPARK_V4_FINE_GRAINED_OVERLAP)
#if !defined(DSPARK_V4_GREEDY_TAIL) || !defined(DSPARK_V4_STATIC_QUEUES) \
    || DSPARK_V4_BATCH != 1
#error "fine-grained overlap requires batch-1 relaxed greedy static queues"
#endif
#include "dspark_v4_generated_greedy_overlap.h"
#elif defined(DSPARK_V4_GREEDY_TAIL)
#if DSPARK_V4_BATCH == 1
#if defined(DSPARK_SHARED_CONTINUE)
#ifdef DSPARK_SHARED_SPLIT_K
#error "shared continuation is only implemented for unsplit shared W13"
#endif
#ifdef DSPARK_ROUTED_W2_WORKERS
static_assert(DSPARK_ROUTED_W2_WORKERS == 128);
#include "dspark_v4_generated_greedy_shared_continue_w2_128.h"
#else
#error "This retired experimental schedule is not shipped; use the retained shared-continue W2-128 schedule"
#endif
#elif defined(DSPARK_FRONT_EMBED_FIRST) || defined(DSPARK_ROUTED_INTERLEAVE) \
    || defined(DSPARK_ROUTED_GROUP_READY) || defined(DSPARK_ROUTED_FUSED_SWIGLU) \
    || defined(DSPARK_WOA_SPLIT_K) || DSPARK_V4_STATIC_GAPS == 1 \
    || DSPARK_V4_STATIC_GAPS == 2 || defined(DSPARK_HC_SPLIT) \
    || defined(DSPARK_V4_SHARED_FIRST)
#error "This retired experimental schedule is not shipped; use the retained shared-continue W2-128 schedule"
#else
#include "dspark_v4_generated_greedy.h"
#endif
#elif DSPARK_V4_BATCH == 2
#include "dspark_v4_generated_greedy_b2.h"
#elif DSPARK_V4_BATCH == 4
#include "dspark_v4_generated_greedy_b4.h"
#elif DSPARK_V4_BATCH == 8
#include "dspark_v4_generated_greedy_b8.h"
#else
#error "no generated header for this DSPARK_V4_BATCH"
#endif
#elif defined(DSPARK_V4_RELAXED_DAG)
#include "dspark_v4_generated_relaxed.h"
#else
#include "dspark_v4_generated.h"
#endif
#include "dspark_w13_phase.cuh"
#ifdef DSPARK_ROUTED_W13_STAGE3
#include "dspark_w13_stage3.cuh"
#endif
#include "dspark_shared_phase.cuh"
#ifdef DSPARK_SHARED_BULK_STAGING
#include "dspark_shared_bulk_phase.cuh"
#endif
#include "dspark_quant_scale.cuh"
#include "dspark_lm_phase.cuh"
#ifdef DSPARK_LM_STREAMED_K
#include "dspark_lm_streamed_phase.cuh"
#endif
#include "dspark_attn_phase.cuh"
#include "dspark_reduction.cuh"
#include "dspark_router_top6.cuh"
#include "dspark_head_prefetch.cuh"
#include "dspark_proj_phase.cuh"
#ifdef DSPARK_WOA_STREAMED_K
#include "dspark_woa_streamed_phase.cuh"
#endif
#ifdef DSPARK_WOA_SPLIT_K
#if DSPARK_WOA_SPLIT_K != 2 || DSPARK_WOA_STREAMED_K != 256 || !defined(DSPARK_HC_SPLIT)
#error "WO_A split-K2 requires streamed K256 and HC split"
#endif
#include "dspark_woa_splitk.cuh"
#endif
#if defined(DSPARK_QB_BLOCK_SCALED) || defined(DSPARK_DENSE_BLOCK_SCALED)
#if defined(DSPARK_FP8_PRESERVE_K128) || !defined(DSPARK_FP8_SCALE4)
#error "The retained bulk FP8 path requires packed scale4 arithmetic"
#endif
#include "dspark_fp8_scale4_phase.cuh"
namespace dspark_fp8_scaled = dspark_fp8_scale4;
#endif
#include "dspark_small_phase.cuh"
#include "dspark_hc_weight_stage.cuh"
#include "dspark_epi_phase.cuh"
#ifdef DSPARK_SWIGLU_WARP_QUANT
#include "dspark_swiglu_warp.cuh"
#endif
#include "dspark_qnorm_phase.cuh"
#include "dspark_router_phase.cuh"
#include "dspark_markov_phase.cuh"
#ifdef DSPARK_V4_STATIC_TP2_TAIL
#include "dspark_v4_tp2_tail.cuh"
#endif

// B2: packed-subbyte tensor-map (TMA) weight staging for the routed FP4
// W13/W2 grouped bodies. Bitwise-proven against the retained DeepGEMM body
// on the bench lane (B1: 0 mismatches at every chain boundary, 1.27x
// three-layer chain). Engages ONLY in relaxed builds: the contract module is
// compiled without DSPARK_W13_TMA_CANDIDATE and stays byte-identical.
#if defined(DSPARK_V4_RELAXED_DAG) && defined(DSPARK_W13_TMA_CANDIDATE)
#if CUDA_VERSION < 12080
#error "DSPARK_V4_TMA_WEIGHTS requires CUDA 12.8 or newer"
#endif
#define DSPARK_V4_TMA_WEIGHTS 1
#include <cstdio>
#include <limits>
#endif

namespace {

namespace generated = deepspec::v4_generated;

// Private Direct2 candidate: ordinary completion has no last-CTA action.
#ifdef DSPARK_PRIVATE_COMPLETION_RED_0908
#if DSPARK_PRIVATE_COMPLETION_RED_0908 != 1 || DSPARK_PRIVATE_DIRECT2_CONFIDENCE_0908 != 1 || DSPARK_V4_DIRECT_DEPENDENCIES != 2 || DSPARK_V4_COMPLETION_RELEASE != 1
#error "completion reduction requires the exact Direct2 confidence release protocol"
#endif
#if DSPARK_V4_STATIC_QUEUES != 1 || DSPARK_V4_ROUTED_DYNAMIC_CLAIMS != 1 || DSPARK_V4_FULL_LOOP_DEVICE_EPOCH != 1 || DSPARK_V4_GREEDY_TAIL != 1 || DSPARK_SHARED_CONTINUE != 1 || DSPARK_V4_BATCH != 1
#error "completion reduction requires the frozen batch-one persistent schedule"
#endif
#if DSPARK_V4_COMPILED_EXECUTION_MASK != 2047 || DSPARK_V4_COMPILED_DRAFT_LAYER_MASK != 7 || DSPARK_CONFIDENCE_HEAD_PREFIX != 1
#error "completion reduction requires every original phase and confidence output"
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE) || defined(DSPARK_V4_FINE_GRAINED_OVERLAP) || defined(DSPARK_V4_STATIC_TP2_TAIL) || defined(DSPARK_V4_QUEUE_LOOKAHEAD) || defined(DSPARK_ROUTED_GROUP_READY)
#error "completion reduction excludes alternate completion and probe paths"
#endif
#endif

#ifdef DSPARK_V4_STATIC_TP2_TAIL
static_assert(
    generated::kWeightOffsetSlots == dspark_v4_tp2::kContractOffsetSlots);
#endif

constexpr int kDynamicSharedBytes = dspark_batch::kDynamicSharedBytes;
// Host launch reservation. Body template budgets remain frozen.
#ifndef DSPARK_V4_LAUNCH_SMEM_BYTES
#define DSPARK_V4_LAUNCH_SMEM_BYTES DSPARK_V4_DYNAMIC_SMEM_BYTES
#endif
constexpr int kLaunchSharedBytes = DSPARK_V4_LAUNCH_SMEM_BYTES;
static_assert(kLaunchSharedBytes >= kDynamicSharedBytes);
#ifdef DSPARK_ROUTED_W13_STAGE3
static_assert(kLaunchSharedBytes >= 227328);
#endif

// R10 unroll sweep. The scheduler kernel is ALWAYS launched with
// generated::kThreads threads per CTA (see the two cudaLaunchKernelEx sites),
// but `blockDim.x` is a RUNTIME special register: every strided loop written
// as `i += blockDim.x` therefore has a trip count nvcc cannot see, so it
// cannot unroll, and each trip issues exactly one load and eats the full
// memory latency before the next. R8 measured that on the HC band -- fixing
// the stride alone, with identical addresses and identical accumulation
// order, was 1.60x. This constant is the compile-time spelling of the same
// value; substituting it is bitwise-neutral by construction.
constexpr int kBlockThreads = generated::kThreads;
static_assert(
    kBlockThreads == 256,
    "the vector bodies index with a compile-time 256-lane stride");

// Idle-window L2 prefetch cursor over the weight arena (consumption order ==
// arena order). Queue-empty CTAs stream 128 KB slices through
// prefetch.global.L2 so scheduler bubbles become warm cache lines for the
// phases about to run. Purely a cache hint: no numerical effect.
__device__ unsigned long long g_idle_prefetch_cursor;
#ifdef DSPARK_V4_RELAXED_DAG
// One-word publish broadcast: idle CTAs are mid-steal-sweep (~147 lock
// probes, 20-30 us) when a chained phase publishes; this hint short-circuits
// discovery to one load. Stale values are harmless — phase_is_ready gates.
__device__ unsigned long long g_ready_hint;
#endif
constexpr unsigned long long kIdlePrefetchBytes = 128ULL * 1024ULL;

// ---------------------------------------------------------------------------
// Scheduler policy (runtime-selectable, host-set via set_scheduler_policy /
// the DSPARK_V4_* environment variables). Nothing here touches phase bodies,
// the DAG, work-unit counts, or the launch ABI — it only changes how idle
// CTAs discover claimable work.
//
// The compiled-in default is the 2026-08-08 measured winner (stage2 median
// 3.5280 ms vs 5.4862 ms for the pre-change behaviour on the same run).
// The pre-change behaviour is still exactly selectable and is what the
// serving co-residency study was run against:
//
//   DSPARK_V4_SCHED_FLAGS=1 DSPARK_V4_DEEP_PERIOD=0 \
//   DSPARK_V4_IDLE_SLEEP_NS=512 DSPARK_V4_IDLE_SLEEP_LONG_NS=4096 \
//   DSPARK_V4_EXPRESS_WORKERS=16 DSPARK_V4_PREFETCH_PERIOD=1
//
// Citizenship note: the new default is strictly LESS hostile to co-resident
// blocks on every axis except poll frequency. It removes the 147-lock victim
// steal sweep (measured at 3.21 ms of lock traffic per CTA per proposal) and
// the 128 KB-per-idle-round L2 weight prefetch entirely; what remains is one
// relaxed load per idle round. Only the parked-warp nap got shorter
// (64 ns vs 512/4096 ns) — raise DSPARK_V4_IDLE_SLEEP_NS if a serving
// measurement asks for it; 512/4096 costs 88 us standalone and still leaves
// 1.75 ms of the win in place.
// ---------------------------------------------------------------------------
struct V4SchedPolicy {
  unsigned int idle_sleep_ns;       // backoff after `idle_sleep_after` rounds
  unsigned int idle_sleep_long_ns;  // backoff after 8 idle rounds
  unsigned int idle_sleep_after;    // idle rounds before backing off at all
  unsigned int flags;
  unsigned int deep_period;       // 0 => deep sweep on every claim attempt
  unsigned int express_workers;   // CTAs reserved for narrow (<=256) phases
  unsigned int prefetch_period;   // idle rounds per L2 prefetch slice; 0 = off
};

// Lock-based victim steal sweep: 147 queue_lock acquisitions per idle round.
constexpr unsigned int kSchedStealSweep = 1u << 0;
// Read publish_epoch/next_tile with relaxed device-scope loads instead of
// atomicAdd(..., 0) read-modify-writes. Pure hint; the next_tile claim atomic
// stays authoritative, so this cannot change which CTA runs which work item.
constexpr unsigned int kSchedRelaxedProbe = 1u << 1;
// Publish-sequence gate: skip the express/steal/scan deep sweep unless the
// global publish counter moved or the periodic safety sweep is due.
constexpr unsigned int kSchedSeqGate = 1u << 2;
// Multi-slot publish ring (replaces the single-word g_ready_hint lookup).
constexpr unsigned int kSchedReadyRing = 1u << 3;
// Ring watermark: never re-walk a slot already rejected. Readiness is
// monotone within an epoch (next_tile only advances), so a slot observed as
// unclaimable can never become claimable again, and the walk collapses from
// 16 dependent L2 round trips to one.
constexpr unsigned int kSchedRingWatermark = 1u << 4;
// Local-queue emptiness hint: skip the queue_lock round trip on our own ready
// queue while it is provably empty. Anything that can fill it either comes
// from this CTA's own partial claim or from a publish, and a publish moves the
// odometer.
constexpr unsigned int kSchedLocalHint = 1u << 5;
// Lean ring probe: bound the walk with the publish odometer instead of a
// separate cursor load, and trust the ring entry's epoch tag as proof of
// publication (publish_phase stores publish_epoch BEFORE the ring slot), so a
// candidate costs one dependent load instead of three.
constexpr unsigned int kSchedRingLean = 1u << 6;

// R16 LEAN PUBLISH. The measured transition cost is a chain of serialized
// device-scope atomics, not a discovery problem: of the 0.518 ms critical-path
// gap, T2 (completion atomic + successor loop) is 0.134 ms and T3 (the
// publish_phase prologue) is 0.192 ms -- 63% of the gap between them, at
// ~4.3 us per node for roughly a dozen dependent atomics. This flag removes
// three of them per transition without changing a single scheduling decision:
//
//   1. broadcast_ready_hint is DEAD under kSchedReadyRing. g_ready_hint is
//      read in exactly one place (the else-branch of the ready-ring test), so
//      the shipped policy pays a globally-contended atomicExch on a single
//      word that nothing ever reads.
//   2. next_tile/completed_tiles are initialized with an unconditional store
//      instead of a read-then-CAS. publish_phase runs for a given (phase,
//      epoch) on exactly ONE CTA, and the phase is not claimable until
//      publish_epoch is stamped afterwards, so nothing else can be touching
//      those two words at that instant -- the read the CAS loop needs is pure
//      overhead here.
//   3. dependency_arrivals keeps its CAS (several predecessors CAN race on it)
//      but probes with a relaxed load rather than atomicAdd(...,0). A stale
//      probe just costs one CAS retry, which the loop already handles.
constexpr unsigned int kSchedLeanPublish = 1u << 7;

// R16 SOLO-DEPENDENCY PUBLISH. 87 of 109 phases have exactly ONE dependency,
// and for those the arrival counter cannot say anything the caller does not
// already know: "every predecessor has arrived" IS "this predecessor just
// completed". Skipping it removes the relaxed probe, the initializing CAS and
// the arrival atomicAdd from the T2 leg of those transitions. Nothing else in
// the kernel reads dependency_arrivals, so leaving it untouched for a
// solo-dependency phase is unobservable.
//
// (Bit 8 previously carried a direct-handoff experiment -- letting the
// publishing CTA carry the successor id straight to its next claim instead of
// rediscovering it through the ring. Measured on a same-session control at
// 2.335136 vs 2.332976 ms: no effect, because the publisher was never the CTA
// that set first-claim latency. Removed rather than kept dark; it cost 4
// registers of a 255-register budget.)
constexpr unsigned int kSchedSoloDep = 1u << 8;

// GB300 publish initialization. publish_phase is the sole writer of a
// phase's epoch-stamped next/completed counters until publish_epoch is made
// visible.  Under this flag those two initialization RMWs become coherent
// device-scope stores; the following publish_epoch atomic and existing
// publish-ring threadfence retain the cross-CTA publication edge.  The
// authoritative next_tile claim remains atomic.
constexpr unsigned int kSchedPlainInit = 1u << 9;

// A ready-ring ticket has one writer.  The slot is only a hint, is naturally
// aligned, and is ordered before publish_seq by the existing device fence.
constexpr unsigned int kSchedPlainRing = 1u << 10;

// kSchedRingWatermark is deliberately NOT in the default: measured twice
// (run 3 -73 us buggy, run 5 -19 us fixed) as a small net loss. Once the walk
// is bounded by the odometer and exits at the first ready slot, skipping
// already-rejected slots saves less than the extra state costs.
// kSchedLeanPublish and kSchedSoloDep ARE in the default: measured together on
// a same-session control at 2.330608 -> 2.228624 ms (-102.0 us), gap 0.527 ->
// 0.386 ms, bitwise exact on every swept row.
// kSchedPlainInit and kSchedPlainRing are GB300-retained: 30 interleaved
// samples measured 2.084176 -> 2.062896 ms (-21.3 us), p90 2.099264 ->
// 2.074496 ms, with identical output ids. A plain completion stamp was neutral
// and is deliberately absent.
constexpr unsigned int kSchedDefaultFlags = kSchedRelaxedProbe
    | kSchedSeqGate | kSchedReadyRing | kSchedLocalHint | kSchedRingLean
    | kSchedLeanPublish | kSchedSoloDep | kSchedPlainInit | kSchedPlainRing;
__device__ V4SchedPolicy g_sched_policy =
    {64u, 64u, 2u, kSchedDefaultFlags, 128u, 0u, 0u};

// Monotonic publish odometer plus a small ring of recently published phases.
// Never reset between proposals: entries are epoch-tagged and phase_is_ready
// gates every candidate, so a stale slot costs one probe and nothing else.
constexpr int kReadyRingSlots = 16;
__device__ unsigned long long g_ready_ring_cursor;
__device__ unsigned long long g_ready_ring[kReadyRingSlots];
// Bumped AFTER the ring slot is visible, so seq == N implies N completed slot
// writes. Idle CTAs cache it and re-probe only when it moves.
__device__ unsigned long long g_publish_seq;

// Per-CTA scheduler state carried across claim attempts (shared, thread 0).
enum SchedStateSlot {
  kSchedSeenSeq = 0,
  kSchedRingFloor = 1,
  kSchedLocalMaybe = 2,
  kSchedStateWords = 3,
};

#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
// Attribution probes. Trace builds only: the timed (non-instrumented) module
// compiles none of this.
constexpr int kProbeWorkerSlots = 24;
enum ProbeWorkerSlot {
  kProbeClaimCalls = 0,
  kProbeClaimHits = 1,
  kProbeNsRing = 2,
  kProbeNsExpress = 3,
  kProbeNsLocal = 4,
  kProbeNsSteal = 5,
  kProbeNsScan = 6,
  kProbeNsClaimAtomic = 7,
  kProbeNsSleep = 8,
  kProbeCountSleep = 9,
  kProbeNsPrefetch = 10,
  kProbeCountPrefetch = 11,
  kProbeNsFinish = 12,
  kProbeCountFinish = 13,
  kProbeHitRing = 14,
  kProbeHitExpress = 15,
  kProbeHitLocal = 16,
  kProbeHitSteal = 17,
  kProbeHitScan = 18,
  kProbeDeepSweeps = 19,
  kProbeNsBody = 20,
  kProbeCountBody = 21,
  kProbeNsIdle = 22,
  kProbeClaimAtomics = 23,
};
// R16 GAP ATTRIBUTION. `critical_path` reports 0.503 ms of scheduler gap
// (phase READY -> first USEFUL start) while publish->first-claim is only
// 0.149 ms, leaving 0.354 ms unattributed. The host can only see what the
// trace ring records, and the ring records USEFUL segments -- so it cannot
// tell "the device really was idle" from "the host's notion of READY is
// earlier than the device's". These slots close the decomposition on the
// device, where every boundary is a real timestamp:
//
//   T0 = body_end[dep]    - ready_host      host-vs-device completion skew
//   T1 = trigger_end      - body_end[dep]   last body end -> finish_phase
//   T2 = arrive           - trigger_end     completion atomic + successor loop
//   T3 = publish          - arrive          publish_phase prologue
//   T4 = first_claim      - publish         wake (the 0.149 ms already known)
//   T5 = useful_start_dev - first_claim     claim -> body start (prologue)
//   T6 = first_start_host - useful_start_dev  host-vs-device start skew
//
// T0 and T6 are MEASUREMENT ARTIFACT; T1..T5 are real device latency. The
// sum is exactly the reported gap, so the split is checkable, not inferred.
constexpr int kProbePhaseSlots = 9;
enum ProbePhaseSlot {
  kProbePublishNs = 0,
  kProbeFirstClaimNs = 1,
  kProbeLastClaimNs = 2,
  kProbeClaimTries = 3,
  // Device-side phase completion: atomicMax over every task's body end,
  // recorded for EVERY phase regardless of role mask. The host trace only
  // sees USEFUL segments, so a Controller-bodied phase (role_mask == 1)
  // reports no completion at all to critical_path and silently defaults to
  // the kernel origin there.
  kProbeBodyEndNs = 4,
  // Device-side phase start: atomicMin over every task's body start.
  kProbeUsefulStartNs = 5,
  // finish_phase entry timestamp of the completer that published this phase.
  kProbeTriggerEndNs = 6,
  // Timestamp immediately before publish_phase is invoked for this phase,
  // i.e. after the dependency-arrival atomic that made it ready.
  kProbeArriveNs = 7,
  // Which phase's completion triggered the publish (+1; 0 == never published
  // by a predecessor, e.g. a DAG root). This is the device's own opinion of
  // the binding predecessor, which need not match the host's max-completion
  // guess -- and when it does not, the host's gap is measured from the wrong
  // instant.
  kProbeTriggerPhase = 8,
};
__device__ unsigned long long
    g_probe_worker[generated::kWorkers * kProbeWorkerSlots];
__device__ unsigned long long
    g_probe_phase[generated::kPhaseCount * kProbePhaseSlots];
#endif
constexpr uint64_t kAuditMagic = 0x44535041524b5634ULL;  // "DSPARKV4"
constexpr int kWorkerWords = 8;
constexpr int kQueueStateWords = 2;
constexpr int kMainHidden = 4096;
constexpr int kMainFeatureWidth = 12288;
constexpr int kMainQuantBlock = 128;
constexpr int kMainSplitK = 8;
constexpr int kMainOutputTile = 128;
constexpr int kMainNormTile = 512;
constexpr float kMainNormEpsilon = 1.0e-6f;
constexpr int kDraftBlock = 5;
// R12 router-band geometry (dspark_router_phase.cuh), RELAXED BUILDS ONLY --
// the contract body below never reads these.
//
// SHIPPED: 8 chains per work item, 32 threads per chain, one row per chain,
// four packed groups per lane per step. That is 160 work items of 256 live
// threads each, against the frozen #101 shape's 32 items of 40 live threads.
// MEASURED same-session on one B200, one process, five geometries
// interleaved (v4_batch_bench.py --router-configs), on-path over 3 layers:
//
//   40x1 (the frozen body, reproduced exactly)  218.4 us   b1 2.354560 ms
//   40x4                                         81.6 us   b1 2.264496
//   20x8                                         51.2 us   b1 2.216768
//   10x16                                        36.3 us   b1 2.206368
//   8x32  RETAINED                               35.8 us   b1 2.201664
//
// 6.10x on the band. The curve is flat past 16 threads per chain, and 8x32
// is taken over 10x16 because it also costs FOUR FEWER REGISTERS (251 vs
// 255 on the b8 translation unit) in a kernel that has none to spare.
//
// `40x1x1x4` stays exactly selectable and is the in-tree numerical control:
// it reproduces the frozen thread->(expert, row) map and the frozen
// ascending FMA order, so it is bitwise the contract body.
#ifndef DSPARK_ROUTER_CHAINS
#define DSPARK_ROUTER_CHAINS 8
#endif
#ifndef DSPARK_ROUTER_SPLIT
#define DSPARK_ROUTER_SPLIT 32
#endif
#ifndef DSPARK_ROUTER_ROWS
#define DSPARK_ROUTER_ROWS 1
#endif
#ifndef DSPARK_ROUTER_UNROLL
#define DSPARK_ROUTER_UNROLL 4
#endif
constexpr int kRouterChainsPerItem = DSPARK_ROUTER_CHAINS;
constexpr int kRouterSplit = DSPARK_ROUTER_SPLIT;
constexpr int kRouterRowsPerChain = DSPARK_ROUTER_ROWS;
constexpr int kRouterUnroll = DSPARK_ROUTER_UNROLL;
// Serving batch (build-time, dspark_batch.h). kDraftRows is the flat row
// count over the (batch, block, ...) contiguous workspace.
constexpr int kBatch = dspark_batch::kBatch;
constexpr int kDraftRows = dspark_batch::kRows;
// tcgen05's f16 N mode is a multiple of 8, so a batch whose row count is not
// (batch 2 -> 10, batch 4 -> 20) runs a padded atom exactly as the frozen
// body pads 5 -> 8; pad columns are computed and discarded. The ladder is
// restricted to the widths R3 verified host-side against cute 4.2.0.0 layout
// algebra AND measured bitwise on the bench lane (8/16/32/40/48).
constexpr int kUmmaBatchRows = kDraftRows <= 8
    ? 8
    : (kDraftRows <= 16 ? 16
                        : (kDraftRows <= 32 ? 32 : (kDraftRows <= 40 ? 40 : 48)));
static_assert(kDraftRows <= 48, "LM activation window fits at most 48 rows");
static_assert(dspark_batch::kBlock == kDraftBlock);
constexpr int kHcStreams = 4;
constexpr int kExpertIntermediate = 2048;
constexpr int kRoutedExperts = 256;
constexpr int kActivatedExperts = 6;
constexpr int kFp4Block = 32;
constexpr float kSwiGluLimit = 10.0f;
constexpr int kNoiseToken = 128799;
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
constexpr bool kTraceEnabled = true;
#else
constexpr bool kTraceEnabled = false;
#endif
constexpr uint32_t kExecuteMain = 1 << 0;
constexpr uint32_t kExecuteEmbedding = 1 << 1;
constexpr uint32_t kExecuteMainKv = 1 << 2;
constexpr uint32_t kExecuteLayer0AttnHc = 1 << 3;
constexpr uint32_t kExecuteLayer0AttnProjections = 1 << 4;
constexpr uint32_t kExecuteLayer0SparseAttention = 1 << 5;
constexpr uint32_t kExecuteLayer0AttentionOutput = 1 << 6;
constexpr uint32_t kExecuteLayer0FfnRouter = 1 << 7;
constexpr uint32_t kExecuteLayer0Experts = 1 << 8;
constexpr uint32_t kExecuteHead = 1 << 9;
constexpr uint32_t kExecuteTail = 1 << 10;
constexpr uint32_t kExecuteAll =
    kExecuteMain | kExecuteEmbedding | kExecuteMainKv | kExecuteLayer0AttnHc
    | kExecuteLayer0AttnProjections | kExecuteLayer0SparseAttention
    | kExecuteLayer0AttentionOutput | kExecuteLayer0FfnRouter
    | kExecuteLayer0Experts | kExecuteHead | kExecuteTail;

__device__ __forceinline__ bool phase_body_enabled(
    int phase,
    uint32_t execution_mask,
    uint8_t draft_layer_mask) {
  const int group = generated::kExecutionGroups[phase];
  uint32_t group_mask = 0;
  switch (group) {
    case generated::kExecutionMain:
      group_mask = kExecuteMain;
      break;
    case generated::kExecutionEmbedding:
      group_mask = kExecuteEmbedding;
      break;
    case generated::kExecutionMainKv:
      group_mask = kExecuteMainKv;
      break;
    case generated::kExecutionAttnHc:
      group_mask = kExecuteLayer0AttnHc;
      break;
    case generated::kExecutionAttnProjection:
      group_mask = kExecuteLayer0AttnProjections;
      break;
    case generated::kExecutionSparseAttention:
      group_mask = kExecuteLayer0SparseAttention;
      break;
    case generated::kExecutionAttentionOutput:
      group_mask = kExecuteLayer0AttentionOutput;
      break;
    case generated::kExecutionFfnHc:
    case generated::kExecutionRouter:
      group_mask = kExecuteLayer0FfnRouter;
      break;
    case generated::kExecutionExpert:
      group_mask = kExecuteLayer0Experts;
      break;
    case generated::kExecutionHead:
      group_mask = kExecuteHead;
      break;
    case generated::kExecutionTail:
      group_mask = kExecuteTail;
      break;
  }
  if ((execution_mask & group_mask) == 0) {
    return false;
  }
  const uint8_t layer = generated::kPhaseLayers[phase];
  return layer == 0xFF || (draft_layer_mask & (1u << layer)) != 0;
}

// The routed-W13 phase bodies (scalar and SM100a TCGen05) live in
// dspark_w13_phase.cuh so the phase microbenchmark compiles exactly the same
// implementation the production kernel executes.

enum class TraceRole : int64_t {
  Controller = 0,
  Tma = 1,
  Mma = 2,
  Epilogue = 3,
};

enum class TraceSegment : int64_t {
  Useful = 0,
  Controller = 1,
  DependencyWait = 2,
  QueueEmptyWait = 3,
  RoleWait = 4,
  // Idle-window L2 weight prefetch. Deliberately NOT Useful: profiles must
  // show bandwidth conversion without painting scheduler bubbles as work.
  Prefetch = 5,
};

enum TraceFlags : int64_t {
  kTraceNone = 0,
  kTraceStolen = 1 << 0,
  kTraceLastTile = 1 << 1,
  kTraceRemotePublish = 1 << 2,
  kTraceQueueOverflow = 1 << 3,
};

struct SchedulerWorkspace {
  uint64_t* dependency_arrivals;
  uint64_t* next_tile;
  uint64_t* completed_tiles;
  uint64_t* publish_epoch;
  uint32_t* ready_queues;
  uint64_t* queue_state;
  uint64_t* worker_state;
};

struct ClaimedTask {
  int phase;
  uint32_t begin;
  uint32_t end;
  uint32_t ticket;
  bool stolen;
};

struct MainStage {
  const __nv_bfloat16* input;
  __nv_bfloat16* output;
  cutlass::float_e4m3_t* quantized_input;
  uint8_t* input_scales;
  float* partials;
  float* projection;
  float* rms_partials;
  const cutlass::float_e4m3_t* weight;
  const uint8_t* weight_scales;
  const float* norm_weight;
  bool enabled;
};

struct EmbeddingStage {
  const int32_t* anchor;
  __nv_bfloat16* output;
  __nv_bfloat16* streams;
  const __nv_bfloat16* weight;
  bool enabled;
};

struct MainKvStage {
  const __nv_bfloat16* projected;
  cutlass::float_e4m3_t* quantized;
  uint8_t* activation_scales;
  float* partials;
  const cutlass::float_e4m3_t* weight[3];
  const uint8_t* weight_scales[3];
  const float* norm_weight[3];
  const float* rope_cos_sin;
  __nv_bfloat16* kv_cache;
  int start_pos;
  bool enabled;
};

struct Layer0AttnHcStage {
  const __nv_bfloat16* streams;
  float* mixes;
  float* pre;
  float* post;
  float* comb;
  __nv_bfloat16* normalized;
  cutlass::float_e4m3_t* quantized;
  uint8_t* activation_scales;
  const float* fn_weight;
  const float* base;
  const float* scale;
  const float* norm_weight;
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  uint8_t layer_mask;
  bool enabled;
};

struct Layer0AttnProjectionStage {
  const cutlass::float_e4m3_t* input_quantized;
  const uint8_t* input_scales;
  float* qa_partials;
  __nv_bfloat16* q_lora;
  cutlass::float_e4m3_t* q_lora_quantized;
  uint8_t* q_lora_scales;
  __nv_bfloat16* query_projection;
  __nv_bfloat16* query_inverse_rms;
  __nv_bfloat16* queries;
  float* draft_kv_partials;
  __nv_bfloat16* draft_kv;
  const cutlass::float_e4m3_t* wqa;
  const uint8_t* wqa_scales;
  const float* qa_norm_weight;
  const cutlass::float_e4m3_t* wqb;
  const uint8_t* wqb_scales;
  const cutlass::float_e4m3_t* wkv;
  const uint8_t* wkv_scales;
  const float* kv_norm_weight;
  const float* rope_cos_sin;
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  uint8_t layer_mask;
  bool enabled;
};

struct Layer0SparseAttentionStage {
  const __nv_bfloat16* queries;
  const __nv_bfloat16* target_kv;
  const __nv_bfloat16* draft_kv;
  float* accumulator;
  __nv_bfloat16* raw_output;
  __nv_bfloat16* output;
  const float* attention_sink;
  const float* rope_cos_sin;
  int start_pos;
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  uint8_t layer_mask;
  bool enabled;
};

struct Layer0AttentionOutputStage {
  const __nv_bfloat16* attention;
  __nv_bfloat16* output_lora;
  cutlass::float_e4m3_t* output_lora_quantized;
  uint8_t* output_lora_scales;
  float* output_partials;
  __nv_bfloat16* output;
  __nv_bfloat16* streams;
  __nv_bfloat16* streams_snapshot;
  const float* post;
  const float* comb;
  const __nv_bfloat16* wo_a;
  const cutlass::float_e4m3_t* wo_b;
  const uint8_t* wo_b_scales;
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  uint8_t layer_mask;
  bool enabled;
};

struct Layer0RouterStage {
  const __nv_bfloat16* input;
  float* scores;
  int32_t* indices;
  float* weights;
  const __nv_bfloat16* gate;
  const float* bias;
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  uint8_t layer_mask;
  bool enabled;
#ifdef DSPARK_ROUTED_GROUP_READY
  uint32_t* group_ready;
#endif
};

struct Layer0ExpertsStage {
  const cutlass::float_e4m3_t* input;
  const uint8_t* input_scales;
  const int32_t* indices;
  const float* route_weights;
  __nv_bfloat16* routed_w13;
  __nv_bfloat16* shared_w13;
  __nv_bfloat16* routed_swiglu;
  cutlass::float_e4m3_t* routed_swiglu_quantized;
  uint8_t* routed_swiglu_scales;
  __nv_bfloat16* shared_swiglu;
  cutlass::float_e4m3_t* shared_swiglu_quantized;
  uint8_t* shared_swiglu_scales;
  __nv_bfloat16* routed_output_partials;
  float* routed_output;
  __nv_bfloat16* shared_output;
  const __nv_bfloat16* residual_streams;
  __nv_bfloat16* streams;
  const float* post;
  const float* comb;
  const uint8_t* weight_arena;
  const int64_t* weight_offsets;
  int layer;
  uint8_t layer_mask;
  bool enabled;
#ifdef DSPARK_V4_TMA_WEIGHTS
  // Device pointers to the {W13, W2} packed-subbyte tensor maps over the
  // routed weight arena (value-initialized to nullptr by the positional
  // stage brace-init; assigned after construction when the launch provides
  // them). nullptr falls back to the retained cp.async staging body.
  const CUtensorMap* w13_weight_tma;
  const CUtensorMap* w2_weight_tma;
#endif
#ifdef DSPARK_ROUTED_GROUP_READY
  uint32_t* group_ready;
#endif
};

struct HeadStage {
  const __nv_bfloat16* streams;
  __nv_bfloat16* hidden;
  __nv_bfloat16* normalized;
  float* base_logits;
  const float* hc_fn;
  const float* hc_base;
  const float* hc_scale;
  const float* norm_weight;
  const float* lm_head;
  bool enabled;
  // Relaxed-numerics probe (acceptance-parity gate, NOT the frozen oracle
  // contract): BF16 LM-head copy appended past the frozen arena. Null in
  // contract mode; value-initialized by the positional stage brace-init.
  const __nv_bfloat16* lm_head_bf16;
};

struct TailStage {
  const int32_t* anchor;
  const __nv_bfloat16* head_hidden;
  const float* base_logits;
  __nv_bfloat16* markov_embeddings;
  float* markov_logits;
  float* partial_max;
  float* partial_sum;
  float* sample_scan;
  int32_t* output_ids;
  float* corrected_logits;
  float* probabilities;
  float* confidence_logits;
  float* calibrated_confidences;
  int32_t* scheduled_prefix_lengths;
  uint8_t* scheduler_read_mask;
  float* scheduler_summary;
  const float* uniforms;
  const float* sts_temperatures;
  const float* steps_per_second;
  const __nv_bfloat16* markov_w1;
  const float* markov_w2;
  const float* confidence_weight;
  float sampling_temperature;
  bool calibration_enabled;
  bool prefix_enabled;
  bool enabled;
  // Relaxed-numerics probe: BF16 markov_w2 copy (checkpoint-exact — the FP32
  // arena entry is itself a widened BF16 tensor). Null in contract mode.
  const __nv_bfloat16* markov_w2_bf16;
};

__device__ __forceinline__ bool layer_enabled(uint8_t layer_mask, int layer) {
  return (layer_mask & (1U << layer)) != 0;
}

template <typename T>
__device__ __forceinline__ const T* layer_weight(
    const uint8_t* arena,
    const int64_t* offsets,
    const uint16_t* slots,
    int layer) {
  return reinterpret_cast<const T*>(arena + offsets[slots[layer]]);
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

// Device-scope relaxed load of a scheduler counter. Same coherence domain as
// the atomicAdd(..., 0) probe it replaces (both are L2 device-scope), but it
// does not occupy an atomic unit slot. Only ever used for READ-ONLY readiness
// hints: every claim still goes through the authoritative next_tile atomic.
__device__ __forceinline__ uint64_t load_relaxed_u64(const uint64_t* address) {
  unsigned long long value;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return static_cast<uint64_t>(value);
}

__device__ __forceinline__ uint64_t load_acquire_u64(const uint64_t* address) {
  unsigned long long value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return static_cast<uint64_t>(value);
}

__device__ __forceinline__ void store_relaxed_u64(
    uint64_t* address,
    uint64_t value) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}

__device__ __forceinline__ void store_release_u64(
    uint64_t* address,
    uint64_t value) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}

__device__ __forceinline__ uint64_t pack_epoch(uint32_t epoch, uint32_t value) {
  return (static_cast<uint64_t>(epoch) << 32) | value;
}

__device__ __forceinline__ uint32_t counter_epoch(uint64_t value) {
  return static_cast<uint32_t>(value >> 32);
}

__device__ __forceinline__ uint32_t counter_value(uint64_t value) {
  return static_cast<uint32_t>(value);
}

// Polling a non-ready counter needs coherence, not an acquire fence on every
// retry. The successful observation still uses acquire before any body reads.
// The initial acquire preserves the common already-ready path's single load.
template <bool AllowCompleted>
__device__ __forceinline__ uint64_t wait_published_phase(
    const uint64_t* address, uint32_t epoch) {
  const auto ready = [epoch](uint64_t value) {
    return counter_epoch(value) == epoch
        && (AllowCompleted ? counter_value(value) >= 1 : counter_value(value) == 1);
  };
  uint64_t observed = load_acquire_u64(address);
  while (!ready(observed)) {
    do {
      __nanosleep(32);
      observed = load_relaxed_u64(address);
    } while (!ready(observed));
    observed = load_acquire_u64(address);
  }
  return observed;
}

#ifdef DSPARK_V4_DIRECT_DEPENDENCIES
// Every phase's counters are initialized before its owner publishes state 1.
// A worker acquires each predecessor's final state 2 directly. Readiness is
// monotonic within one proposal, so repeated claims can reuse the acquire.
__device__ __forceinline__ uint64_t wait_direct_dependencies(
    SchedulerWorkspace workspace, int phase, uint32_t epoch, int& ready_cache) {
  if (ready_cache == phase) return pack_epoch(epoch, 1);
  const uint64_t initialized = wait_published_phase<true>(
      &workspace.publish_epoch[phase], epoch);
  if (counter_value(initialized) == 2) {
    ready_cache = phase;
    return initialized;
  }
  const int begin = generated::kPredecessorOffsets[phase];
  const int end = generated::kPredecessorOffsets[phase + 1];
  for (int i = begin; i < end; ++i) {
    const int predecessor = generated::kPredecessors[i];
#if DSPARK_V4_DIRECT_DEPENDENCIES == 2
    bool epilogue = false;
#pragma unroll
    for (int step = 0; step < kDraftBlock; ++step)
      epilogue |= predecessor == generated::kMarkovW2Phases[step];
    const uint64_t completed = pack_epoch(
        epoch, epilogue ? 2u : generated::kWorkUnits[predecessor]);
    const uint64_t* counter = epilogue
        ? &workspace.publish_epoch[predecessor]
        : &workspace.completed_tiles[predecessor];
    // Acquiring the last release RMW joins every producer's output through
    // that counter's release sequence. Only Markov has a last-CTA output
    // epilogue that requires a separate completed flag.
    while (load_acquire_u64(counter) != completed) __nanosleep(32);
#else
    const uint64_t completed = pack_epoch(epoch, 2);
    while (load_acquire_u64(&workspace.publish_epoch[predecessor]) != completed)
      __nanosleep(32);
#endif
  }
  ready_cache = phase;
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  atomicCAS(&g_probe_phase[phase * kProbePhaseSlots + kProbePublishNs],
            0ULL, static_cast<unsigned long long>(globaltimer_ns()));
#endif
  return pack_epoch(epoch, 1);
}
#ifdef DSPARK_V4_QUEUE_LOOKAHEAD
// Probe without parking the worker: another assigned phase may already be
// runnable. All task ownership stays in the GPU's per-worker stream; this
// changes only readiness selection, never the DAG or completion credits.
__device__ __forceinline__ bool direct_dependencies_ready(
    SchedulerWorkspace workspace, int phase, uint32_t epoch, int& ready_cache) {
  static_assert(DSPARK_V4_DIRECT_DEPENDENCIES == 2);
  if (ready_cache == phase) return true;
  if (load_acquire_u64(&workspace.publish_epoch[phase]) != pack_epoch(epoch, 1))
    return false;
  const int begin = generated::kPredecessorOffsets[phase];
  const int end = generated::kPredecessorOffsets[phase + 1];
  for (int i = begin; i < end; ++i) {
    const int predecessor = generated::kPredecessors[i];
    bool epilogue = false;
#pragma unroll
    for (int step = 0; step < kDraftBlock; ++step)
      epilogue |= predecessor == generated::kMarkovW2Phases[step];
    const uint64_t completed = pack_epoch(
        epoch, epilogue ? 2u : generated::kWorkUnits[predecessor]);
    const uint64_t* counter = epilogue
        ? &workspace.publish_epoch[predecessor]
        : &workspace.completed_tiles[predecessor];
    if (load_acquire_u64(counter) != completed) return false;
  }
  ready_cache = phase;
  return true;
}

__device__ __forceinline__ ClaimedTask claim_ready_static_task(
    SchedulerWorkspace workspace, uint32_t epoch, int& cursor, int end,
    uint32_t& retired, int& ready_cache) {
  static_assert(DSPARK_V4_QUEUE_LOOKAHEAD <= 32);
  while (cursor < end) {
    while ((retired & 1u) != 0) {
      retired >>= 1;
      ++cursor;
    }
    if (cursor == end) break;
    for (int offset = 0;
         offset < DSPARK_V4_QUEUE_LOOKAHEAD && cursor + offset < end; ++offset) {
      const uint32_t bit = 1u << offset;
      if ((retired & bit) != 0) continue;
      const int slot = cursor + offset;
      const int phase = generated::kStaticTaskPhases[slot];
      // Markov completion includes a final output epilogue and changes its
      // initialized flag to state 2. Its already-retired slots can be skipped.
      if (load_relaxed_u64(&workspace.publish_epoch[phase]) == pack_epoch(epoch, 2)) {
        retired |= bit;
        continue;
      }
      if (!direct_dependencies_ready(workspace, phase, epoch, ready_cache)) continue;
      const uint32_t chunk = generated::kClaimChunks[phase];
      uint32_t first;
      bool routed = false;
#pragma unroll
      for (int layer = 0; layer < 3; ++layer)
        routed |= phase == generated::kRoutedW13Phases[layer]
            || phase == generated::kRoutedW2Phases[layer];
      if (routed) {
        const uint64_t claimed = atomicAdd(
            reinterpret_cast<unsigned long long*>(&workspace.next_tile[phase]),
            static_cast<unsigned long long>(chunk));
        if (counter_epoch(claimed) != epoch) __trap();
        first = counter_value(claimed);
        if (first >= generated::kWorkUnits[phase]) {
          retired |= bit;
          continue;
        }
      } else {
        first = generated::kStaticTaskClaims[slot] * chunk;
        retired |= bit;
      }
      const uint32_t last = first + chunk < generated::kWorkUnits[phase]
          ? first + chunk : generated::kWorkUnits[phase];
      return {phase, first, last, first / chunk, false};
    }
    __nanosleep(32);
  }
  return {-1, 0, 0, 0, false};
}
#endif
#endif

#ifdef DSPARK_V4_COMPLETION_RELEASE
__device__ __forceinline__ uint64_t complete_items_release(
    uint64_t* address, uint64_t count) {
  uint64_t previous;
  // The caller's CTA barrier joins all lanes' output writes. Release them to
  // the counter release sequence. Only its final participant needs an acquire
  // fence before publishing completion and performing the Markov epilogue.
  asm volatile("atom.release.gpu.global.add.u64 %0, [%1], %2;"
               : "=l"(previous) : "l"(address), "l"(count) : "memory");
  return previous;
}
#endif

__device__ __forceinline__ float decode_e8m0(uint8_t bits) {
  const uint32_t exponent = bits;
  const uint32_t float_bits =
      exponent == 0 ? 0x00400000U : exponent << 23;
  return __uint_as_float(float_bits);
}

__device__ __forceinline__ float decode_e2m1_code(uint32_t code) {
  const uint32_t magnitude = code & 0x07;
  uint32_t bits = 0;
  if (magnitude != 0) {
    bits = magnitude == 1
        ? 0x3f000000U
        : 0x3f000000U + (magnitude << 22);
  }
  bits |= (code & 0x08) << 28;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float decode_e2m1(uint8_t packed, int logical_index) {
  const uint32_t code = logical_index & 1 ? packed >> 4 : packed & 0x0f;
  return decode_e2m1_code(code);
}

// R1 leg 6, intra-node retile: one WARP owns one 128-element quantization
// block, instead of the whole CTA walking the blocks one at a time.
//
// The retired shape ran, per block, a 256-lane stride-halving fmaxf tree in
// shared memory with a __syncthreads() at every level: 11 CTA-wide barriers
// per block, 352 of them for the 32-block activation row that the fused
// sinkhorn band emits, with half the lanes holding nothing but a 0.0f pad and
// the 32 independent blocks fully serialized behind each other.
//
// amax is a MAX reduction. fmaxf is associative, commutative and exact (no
// rounding, and these activations are never NaN), so the tree TOPOLOGY is not
// part of the numerics -- the same order-free fact the FP8 requant epilogue
// already relies on. Every lane of a warp ends the butterfly holding the
// identical amax, so the e8m0 exponent, the committed scale byte and every
// quantized element are bit-for-bit what the CTA-wide tree produced. No
// accumulation chain is touched.
__device__ void quantize_bf16_blocks(
    const __nv_bfloat16* input,
    cutlass::float_e4m3_t* output,
    uint8_t* scales,
    int blocks,
    float* shared) {
  (void)shared;
  constexpr int kLanes = 32;
  constexpr int kPerLane = kMainQuantBlock / kLanes;
  static_assert(kPerLane * kLanes == kMainQuantBlock);
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  // Compile-time warp count: the block-strided loop below is otherwise
  // bounded by a runtime stride and cannot unroll (R10).
  constexpr int warps = kBlockThreads / kLanes;
  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  for (int block = warp; block < blocks; block += warps) {
    const int base = block * kMainQuantBlock;
    float values[kPerLane];
    float amax = 0.0f;
#pragma unroll
    for (int index = 0; index < kPerLane; ++index) {
      values[index] = __bfloat162float(input[base + index * kLanes + lane]);
      amax = fmaxf(amax, fabsf(values[index]));
    }
#pragma unroll
    for (int delta = kLanes / 2; delta > 0; delta /= 2) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
    }
    amax = fmaxf(amax, 1.0e-4f);
    const int exponent = dspark_quant_scale::exponent(amax);
    if (lane == 0) {
      const int raw_encoded = exponent + 127;
      scales[block] = static_cast<uint8_t>(
          raw_encoded < 0 ? 0 : (raw_encoded > 254 ? 254 : raw_encoded));
    }
    const float scale = dspark_quant_scale::scale(exponent);
#pragma unroll
    for (int index = 0; index < kPerLane; ++index) {
      output[base + index * kLanes + lane] = convert(values[index] / scale);
    }
  }
  // The retired body ended every block with a CTA barrier; callers reuse the
  // quantized row (and `shared`) immediately after returning.
  __syncthreads();
}

__device__ void execute_main_quant(
    const ClaimedTask& task,
    MainStage stage,
    float* shared) {
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int base = static_cast<int>(item) * kMainQuantBlock;
    float value = 0.0f;
    if (threadIdx.x < kMainQuantBlock) {
      value = __bfloat162float(stage.input[base + threadIdx.x]);
      shared[threadIdx.x] = fabsf(value);
    } else {
      shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int delta = kBlockThreads / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + delta]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float amax = fmaxf(shared[0], 1.0e-4f);
      const int exponent = dspark_quant_scale::exponent(amax);
      const int raw_encoded = exponent + 127;
      const int encoded = raw_encoded < 0 ? 0 : (raw_encoded > 254 ? 254 : raw_encoded);
      stage.input_scales[item] = static_cast<uint8_t>(encoded);
      shared[0] = dspark_quant_scale::scale(exponent);
    }
    __syncthreads();
    if (threadIdx.x < kMainQuantBlock) {
      stage.quantized_input[base + threadIdx.x] = convert(value / shared[0]);
    }
    __syncthreads();
  }
}

__device__ void execute_main_projection(
    const ClaimedTask& task,
    MainStage stage) {
  // Body extracted verbatim into dspark_epi_phase.cuh (relaxed-drafter
  // stage 3, front/epilogue bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion; serves contract AND
  // relaxed builds.
  dspark_epi::MainProjArgs args;
  args.quantized_input = stage.quantized_input;
  args.input_scales = stage.input_scales;
  args.weight = stage.weight;
  args.weight_scales = stage.weight_scales;
  args.partials = stage.partials;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
#ifdef DSPARK_DENSE_BLOCK_SCALED
  dspark_fp8_scaled::execute<DSPARK_DENSE_BLOCK_SCALED, 12288, 4096, 1, 8, false>(
      {args.quantized_input, args.input_scales,
       args.weight, args.weight_scales,
       args.partials, nullptr, args.begin, args.end});
#else
  dspark_epi::execute_mainproj_tcgen(args);
#endif
#else
  dspark_epi::execute_mainproj_reference(args);
#endif
}

__device__ void execute_main_reduce(
    const ClaimedTask& task,
    MainStage stage,
    float* shared) {
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int tile_base = static_cast<int>(item) * kMainNormTile;
    float local_square = 0.0f;
    #pragma unroll
    for (int lane_item = threadIdx.x; lane_item < kMainNormTile; lane_item += kBlockThreads) {
      const int column = tile_base + lane_item;
      float value = 0.0f;
#pragma unroll
      for (int split = 0; split < kMainSplitK; ++split) {
        value += stage.partials[column * kMainSplitK + split];
      }
      // The official FP8 GEMM publishes BF16 before RMSNorm. Preserve that
      // arithmetic boundary even though split-K scratch remains FP32.
      const float rounded = __bfloat162float(__float2bfloat16_rn(value));
      stage.projection[column] = rounded;
      local_square = fmaf(rounded, rounded, local_square);
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    dspark_reduction::sum_256(shared);
    if (threadIdx.x == 0) {
      stage.rms_partials[item] = shared[0];
    }
    __syncthreads();
  }
}

__device__ void execute_main_norm(
    const ClaimedTask& task,
    MainStage stage) {
  // Batched: rms_partials is (batch, main_rows, 8) and the item space is
  // (batch x 8) tiles, so the row's eight partials belong to the item's own
  // batch element. At batch 1 element is always 0 and this is the retired
  // expression verbatim.
  constexpr int kTilesPerRow = kMainHidden / kMainNormTile;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int element = static_cast<int>(item) / kTilesPerRow;
    float sum_square = 0.0f;
#pragma unroll
    for (int tile = 0; tile < kTilesPerRow; ++tile) {
      sum_square += stage.rms_partials[element * kTilesPerRow + tile];
    }
    const float inverse_rms =
        rsqrtf(sum_square / kMainHidden + kMainNormEpsilon);
    const int tile_base = static_cast<int>(item) * kMainNormTile;
    #pragma unroll
    for (int lane_item = threadIdx.x; lane_item < kMainNormTile; lane_item += kBlockThreads) {
      const int column = tile_base + lane_item;
      // `column` is FLAT over (batch, hidden); the RMS weight is one
      // hidden-wide vector shared by every element.
      const float value = stage.projection[column] * inverse_rms
          * stage.norm_weight[column % kMainHidden];
      stage.output[column] = __float2bfloat16_rn(value);
    }
  }
}

__device__ DSPARK_OTHER_ENTRY bool execute_main_phase(
    const ClaimedTask& task,
    MainStage stage,
    float* shared) {
  if (!stage.enabled) {
    return false;
  }
  if (task.phase == generated::kMainQuantPhase) {
    execute_main_quant(task, stage, shared);
  } else if (task.phase == generated::kMainProjectionPhase) {
    execute_main_projection(task, stage);
  } else if (task.phase == generated::kMainReducePhase) {
    execute_main_reduce(task, stage, shared);
  } else if (task.phase == generated::kMainNormPhase) {
    execute_main_norm(task, stage);
  } else {
    return false;
  }
  return true;
}

__device__ DSPARK_OTHER_ENTRY void execute_embedding_phase(
    const ClaimedTask& task,
    EmbeddingStage stage) {
  if (!stage.enabled || task.phase != generated::kEmbeddingPhase) {
    return;
  }
  constexpr int kTile = 512;
  constexpr int kElements = kDraftRows * kHcStreams * kMainHidden;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int tile_base = static_cast<int>(item) * kTile;
    #pragma unroll
    for (int lane_item = threadIdx.x; lane_item < kTile; lane_item += kBlockThreads) {
      const int output_index = tile_base + lane_item;
      if (output_index < kElements) {
        const int column = output_index % kMainHidden;
        // Flat row over the (batch, block, hc, hidden) region: row r belongs
        // to batch element r / 5 and is that element's anchor when it is the
        // element's first draft position.
        const int position = output_index / (kHcStreams * kMainHidden);
        const int token = position % kDraftBlock == 0
            ? stage.anchor[position / kDraftBlock]
            : kNoiseToken;
        const __nv_bfloat16 value =
            stage.weight[static_cast<int64_t>(token) * kMainHidden + column];
        stage.output[output_index] = value;
        stage.streams[output_index] = value;
      }
    }
  }
}

__device__ void execute_main_projected_quant(
    const ClaimedTask& task,
    MainKvStage stage,
    float* shared) {
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int base = static_cast<int>(item) * kMainQuantBlock;
    float value = 0.0f;
    if (threadIdx.x < kMainQuantBlock) {
      value = __bfloat162float(stage.projected[base + threadIdx.x]);
      shared[threadIdx.x] = fabsf(value);
    } else {
      shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int delta = kBlockThreads / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + delta]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float amax = fmaxf(shared[0], 1.0e-4f);
      const int exponent = dspark_quant_scale::exponent(amax);
      const int raw_encoded = exponent + 127;
      const int encoded = raw_encoded < 0 ? 0 : (raw_encoded > 254 ? 254 : raw_encoded);
      stage.activation_scales[item] = static_cast<uint8_t>(encoded);
      shared[0] = dspark_quant_scale::scale(exponent);
    }
    __syncthreads();
    if (threadIdx.x < kMainQuantBlock) {
      stage.quantized[base + threadIdx.x] = convert(value / shared[0]);
    }
    __syncthreads();
  }
}

__device__ void execute_main_kv_projection(
    const ClaimedTask& task,
    MainKvStage stage,
    int layer) {
  // Body extracted verbatim into dspark_small_phase.cuh (relaxed-drafter
  // stage 3, small dense bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion (the [layer] indirections
  // become pre-offset pointers); serves contract AND relaxed builds.
  dspark_small::MainKvArgs args;
  args.quantized = stage.quantized;
  args.activation_scales = stage.activation_scales;
  args.weight = stage.weight[layer];
  args.weight_scales = stage.weight_scales[layer];
  // main_kv_partials is (layers, batch, main_rows, kv_width, 4): the layer
  // slab holds every batch element's split partials back to back, which is
  // exactly the row-major operand the batch-widened body writes.
  args.partials = stage.partials + layer * kBatch * 512 * 4;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
#ifdef DSPARK_DENSE_BLOCK_SCALED
  dspark_fp8_scaled::execute<DSPARK_DENSE_BLOCK_SCALED, 4096, 512, 1, 4, false>(
      {args.quantized, args.activation_scales,
       args.weight, args.weight_scales,
       args.partials, nullptr, args.begin, args.end});
#else
  dspark_small::execute_main_kv_tcgen(args);
#endif
#else
  dspark_small::execute_main_kv_reference(args);
#endif
}

__device__ void execute_main_kv_finalize(
    MainKvStage stage,
    int layer,
    int element,
    float* shared) {
  constexpr int kSplits = 4;
  float local_square = 0.0f;
  #pragma unroll
  for (int column = threadIdx.x; column < 512; column += kBlockThreads) {
    float value = 0.0f;
#pragma unroll
    for (int split = 0; split < kSplits; ++split) {
      value +=
          stage.partials[((layer * kBatch + element) * 512 + column) * kSplits
                         + split];
    }
    // fp8_gemm returns BF16, then DeepSeek's RMSNorm widens that rounded
    // tensor to FP32. Skipping this boundary amplifies small accumulator
    // differences at the following in-place FP8-simulation boundary.
    const float rounded = __bfloat162float(__float2bfloat16_rn(value));
    shared[256 + column] = rounded;
    local_square = fmaf(rounded, rounded, local_square);
  }
  shared[threadIdx.x] = local_square;
  __syncthreads();
  dspark_reduction::sum_256(shared);
  const float inverse_rms = rsqrtf(shared[0] / 512.0f + kMainNormEpsilon);
  // kv_cache is (layers, batch, window, kv_width). Every element shares
  // start_pos in this batched specialization (see the report's deferred
  // list), so only the batch stride is new here.
  const int cache_base =
      ((layer * kBatch + element) * 128 + stage.start_pos % 128) * 512;
  #pragma unroll
  for (int column = threadIdx.x; column < 512; column += kBlockThreads) {
    const float normalized =
        shared[256 + column] * inverse_rms * stage.norm_weight[layer][column];
    stage.kv_cache[cache_base + column] = __float2bfloat16_rn(normalized);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    const int pair = threadIdx.x;
    const int first = cache_base + 448 + 2 * pair;
    const float x0 = __bfloat162float(stage.kv_cache[first]);
    const float x1 = __bfloat162float(stage.kv_cache[first + 1]);
    const float cosine = stage.rope_cos_sin[pair * 2];
    const float sine = stage.rope_cos_sin[pair * 2 + 1];
    stage.kv_cache[first] = __float2bfloat16_rn(x0 * cosine - x1 * sine);
    stage.kv_cache[first + 1] = __float2bfloat16_rn(x1 * cosine + x0 * sine);
  }
  __syncthreads();

  cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
#ifdef DSPARK_MAIN_KV_WARP_QUANT
  // Seven independent 64-element groups occupy seven warps. The max reduction
  // is order-independent; all rounding and quantize/dequantize steps stay put.
  const int group = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (group < 7) {
    const int first = cache_base + group * 64 + lane;
    const float a = __bfloat162float(stage.kv_cache[first]);
    const float b = __bfloat162float(stage.kv_cache[first + 32]);
    float amax = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
    amax = fmaxf(amax, 1.0e-4f);
    const float scale = dspark_quant_scale::scale(dspark_quant_scale::exponent(amax));
    stage.kv_cache[first] = __float2bfloat16_rn(static_cast<float>(convert(a / scale)) * scale);
    stage.kv_cache[first + 32] = __float2bfloat16_rn(static_cast<float>(convert(b / scale)) * scale);
  }
  __syncthreads();
#else
  for (int group = 0; group < 7; ++group) {
    float value = 0.0f;
    if (threadIdx.x < 64) {
      value = __bfloat162float(stage.kv_cache[cache_base + group * 64 + threadIdx.x]);
      shared[threadIdx.x] = fabsf(value);
    } else {
      shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int delta = kBlockThreads / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + delta]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float amax = fmaxf(shared[0], 1.0e-4f);
      const int exponent = dspark_quant_scale::exponent(amax);
      shared[0] = dspark_quant_scale::scale(exponent);
    }
    __syncthreads();
    if (threadIdx.x < 64) {
      const float quantized = static_cast<float>(convert(value / shared[0])) * shared[0];
      stage.kv_cache[cache_base + group * 64 + threadIdx.x] =
          __float2bfloat16_rn(quantized);
    }
    __syncthreads();
  }
#endif
}

__device__ DSPARK_OTHER_ENTRY void execute_main_kv_phase(
    const ClaimedTask& task,
    MainKvStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  if (task.phase == generated::kMainProjectedQuantPhase) {
    execute_main_projected_quant(task, stage, shared);
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase == generated::kMainKvProjectionPhases[layer]) {
      execute_main_kv_projection(task, stage, layer);
      return;
    }
    if (task.phase == generated::kMainKvNormPhases[layer]) {
      // One item per batch element (work units = batch).
      for (uint32_t item = task.begin; item < task.end; ++item) {
        execute_main_kv_finalize(stage, layer, static_cast<int>(item), shared);
      }
      return;
    }
  }
}

static_assert(
    generated::kThreads == dspark_small::kHcThreads,
    "the HC bodies index columns with a compile-time 256-lane stride");
// R10: the same contract for every other extracted band whose strided loops
// now carry a compile-time lane stride instead of blockDim.x.
static_assert(
    generated::kThreads == dspark_attn::kAttnThreads,
    "the batched attention body indexes with a compile-time lane stride");
static_assert(
    generated::kThreads == dspark_epi::kEpiThreads,
    "the SwiGLU bodies index with a compile-time lane stride");
static_assert(
    generated::kThreads == dspark_qnorm::kQNormThreads,
    "the q/kv norm bodies index with a compile-time lane stride");

__device__ void execute_layer0_attn_hc_projection(
    const ClaimedTask& task,
    Layer0AttnHcStage stage,
    float* shared) {
  // Body extracted verbatim into dspark_small_phase.cuh (relaxed-drafter
  // stage 3, small dense bands; the kDynamicSharedBytes static_assert moved
  // with it). Pure code motion; serves contract AND relaxed builds, and both
  // the attn and ffn HC phases (this function is called by both).
  dspark_small::HcArgs args;
  args.streams = stage.streams;
  args.fn_weight = stage.fn_weight;
  args.mixes = stage.mixes;
  args.begin = task.begin;
  args.end = task.end;
#ifdef DSPARK_V4_RELAXED_DAG
  // R7: execute_hc_parallel was 65.6% long-scoreboard at 1.29% of DRAM peak
  // with a perfect access pattern (sectors actual/ideal 1.000, zero bank
  // conflicts) -- one load outstanding at a time, because the 16,384-wide
  // loop was bounded by blockDim.x and nvcc could not unroll it. This body
  // caches the row through SMEM with cp.async and runs the weight stream 16
  // loads deep. WORK UNITS ARE UNCHANGED at 120 (row, mix) items: the
  // measured mixes-per-CTA sweep turns NEGATIVE immediately at batch 1,
  // where 120 items on 152 workers means the span is one item's latency and
  // every extra mix per CTA lengthens it. Bitwise-identical to
  // execute_hc_parallel by construction and by measurement.
  dspark_small::execute_hc_mixgroup_staged_weights(args, shared);
#else
  dspark_small::execute_hc_reference(args, shared);
#endif
}

__device__ __forceinline__ float sigmoidf(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ void execute_layer0_attn_hc_sinkhorn(
    const ClaimedTask& task,
    Layer0AttnHcStage stage) {
#ifdef DSPARK_HC_ELEMENT_LANES
  // One lane per matrix element. Each ordered four-element sum is gathered
  // without reassociation; the IEEE divisions and epsilon placements match
  // the four-lane control below. All sixteen named lanes execute every shuffle.
  constexpr float kHcEpsilon = 1.0e-6f;
  constexpr unsigned kHcMask = 0xffffu;
  static_assert(kHcStreams == 4, "element-lane Sinkhorn specializes the 4x4 matrix");
  for (uint32_t item = task.begin; item < task.end; ++item) {
    if (threadIdx.x >= 16) {
      continue;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int output = lane / 4;
    const int input = lane % 4;
    const int row = static_cast<int>(item);
    const int mix_base = row * 24;
    if (lane < 4) {
      stage.pre[row * 4 + lane] =
          sigmoidf(stage.mixes[mix_base + lane] * stage.scale[0]
                   + stage.base[lane]) + kHcEpsilon;
      stage.post[row * 4 + lane] =
          2.0f * sigmoidf(stage.mixes[mix_base + 4 + lane] * stage.scale[1]
                         + stage.base[4 + lane]);
    }
    float value = stage.mixes[mix_base + 8 + lane] * stage.scale[2]
        + stage.base[8 + lane];
    float maximum = -FLT_MAX;
#pragma unroll
    for (int source = 0; source < 4; ++source) {
      maximum = fmaxf(maximum, __shfl_sync(kHcMask, value, output * 4 + source));
    }
    value = expf(value - maximum);
    float row_sum = 0.0f;
#pragma unroll
    for (int source = 0; source < 4; ++source) {
      row_sum += __shfl_sync(kHcMask, value, output * 4 + source);
    }
    value = value / row_sum + kHcEpsilon;
    float column_sum = 0.0f;
#pragma unroll
    for (int source = 0; source < 4; ++source) {
      column_sum += __shfl_sync(kHcMask, value, source * 4 + input);
    }
    value /= column_sum + kHcEpsilon;
    for (int iteration = 1; iteration < 20; ++iteration) {
      row_sum = 0.0f;
#pragma unroll
      for (int source = 0; source < 4; ++source) {
        row_sum += __shfl_sync(kHcMask, value, output * 4 + source);
      }
      value /= row_sum + kHcEpsilon;
      column_sum = 0.0f;
#pragma unroll
      for (int source = 0; source < 4; ++source) {
        column_sum += __shfl_sync(kHcMask, value, source * 4 + input);
      }
      value /= column_sum + kHcEpsilon;
    }
    stage.comb[row * 16 + lane] = value;
  }
#else
  // R1 leg 6, intra-node retile: one LANE per Sinkhorn output row instead of
  // the whole 4x4 doubly-stochastic projection on thread 0.
  //
  // The retired shape left 255 of the CTA's 256 threads parked at the fused
  // band's __syncthreads() while thread 0 ran 20 iterations x 32 IEEE divides
  // serially. Lane o now owns matrix row o in registers:
  //   - the row half-iteration is entirely lane-local, and its sum still runs
  //     input-ascending, so it is the identical expression;
  //   - the column half-iteration gathers output-ascending with four
  //     __shfl_sync reads, so its sum is still ((m0+m1)+m2)+m3 in the same
  //     order, and every lane computes the identical column total.
  // No sum is reassociated and no value crosses a rounding boundary it did
  // not cross before; the divide count per lane drops 4x and the four rows
  // proceed in parallel.
  constexpr float kHcEpsilon = 1.0e-6f;
  constexpr unsigned kHcLaneMask = (1u << kHcStreams) - 1u;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    if (threadIdx.x >= static_cast<unsigned>(kHcStreams)) {
      continue;
    }
    const int output = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(item);
    const int mix_base = row * 24;
    stage.pre[row * kHcStreams + output] =
        sigmoidf(
            stage.mixes[mix_base + output] * stage.scale[0]
            + stage.base[output])
        + kHcEpsilon;
    stage.post[row * kHcStreams + output] =
        2.0f
        * sigmoidf(
            stage.mixes[mix_base + kHcStreams + output] * stage.scale[1]
            + stage.base[kHcStreams + output]);

    // matrix[output][*] for this lane's output row.
    float matrix[kHcStreams];
    {
      float maximum = -FLT_MAX;
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        const int index = 2 * kHcStreams + output * kHcStreams + input;
        matrix[input] =
            stage.mixes[mix_base + index] * stage.scale[2] + stage.base[index];
        maximum = fmaxf(maximum, matrix[input]);
      }
      float sum = 0.0f;
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        matrix[input] = expf(matrix[input] - maximum);
        sum += matrix[input];
      }
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        matrix[input] = matrix[input] / sum + kHcEpsilon;
      }
    }

    // Column normalization: column totals accumulate output-ascending, which
    // is exactly the retired `for (output) sum += matrix[output][input]`.
    {
      float column_sum[kHcStreams];
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        float sum = 0.0f;
#pragma unroll
        for (int source = 0; source < kHcStreams; ++source) {
          sum += __shfl_sync(kHcLaneMask, matrix[input], source);
        }
        column_sum[input] = sum;
      }
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        matrix[input] /= column_sum[input] + kHcEpsilon;
      }
    }

    for (int iteration = 1; iteration < 20; ++iteration) {
      float sum = 0.0f;
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        sum += matrix[input];
      }
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        matrix[input] /= sum + kHcEpsilon;
      }
      float column_sum[kHcStreams];
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        float total = 0.0f;
#pragma unroll
        for (int source = 0; source < kHcStreams; ++source) {
          total += __shfl_sync(kHcLaneMask, matrix[input], source);
        }
        column_sum[input] = total;
      }
#pragma unroll
      for (int input = 0; input < kHcStreams; ++input) {
        matrix[input] /= column_sum[input] + kHcEpsilon;
      }
    }

#pragma unroll
    for (int input = 0; input < kHcStreams; ++input) {
      stage.comb[(row * kHcStreams + output) * kHcStreams + input] =
          matrix[input];
    }
  }
#endif
}

__device__ void execute_layer0_attn_hc_reduce_norm(
    const ClaimedTask& task,
    Layer0AttnHcStage stage,
    float* shared) {
  constexpr int kRowElements = kHcStreams * kMainHidden;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int row = static_cast<int>(item);
#ifdef DSPARK_V4_RELAXED_DAG
    // GB300 relaxed path: keep the reduced BF16 row in shared memory until
    // the row RMS is known, then normalize and FP8-quantize in one warp-mapped
    // pass. The old path wrote the same 8 KiB intermediate to global, read it
    // for normalization, and read it again in quantize_bf16_blocks.
    //
    // Arithmetic is unchanged: each column's four-FMA chain, BF16 rounding,
    // per-thread square chain, 256-lane sum tree, final BF16 rounding, warp
    // fmax, E8M0 scale, and FP8 conversion are exactly the retained sequence.
    __nv_bfloat16* reduced_row = reinterpret_cast<__nv_bfloat16*>(
        shared + kBlockThreads);
    static_assert(
        kBlockThreads * static_cast<int>(sizeof(float))
                + kMainHidden * static_cast<int>(sizeof(__nv_bfloat16))
            <= kDynamicSharedBytes,
        "HC reduced-row scratch must fit dynamic shared memory");
    float local_square = 0.0f;
#pragma unroll 4
    for (int column = threadIdx.x; column < kMainHidden;
         column += kBlockThreads) {
      float reduced = 0.0f;
#pragma unroll
      for (int stream = 0; stream < kHcStreams; ++stream) {
        reduced = fmaf(
            stage.pre[row * kHcStreams + stream],
            __bfloat162float(
                stage.streams[
                    row * kRowElements + stream * kMainHidden + column]),
            reduced);
      }
      const __nv_bfloat16 stored = __float2bfloat16_rn(reduced);
      reduced_row[column] = stored;
      const float value = __bfloat162float(stored);
      local_square = fmaf(value, value, local_square);
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    dspark_reduction::sum_256(shared);
    const float inverse_rms =
        rsqrtf(shared[0] / kMainHidden + kMainNormEpsilon);
    constexpr int kLanes = 32;
    constexpr int kWarps = kBlockThreads / kLanes;
    constexpr int kValuesPerLane = kMainQuantBlock / kLanes;
    const int warp = static_cast<int>(threadIdx.x) / kLanes;
    const int lane = static_cast<int>(threadIdx.x) % kLanes;
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> convert;
    for (int block = warp; block < kMainHidden / kMainQuantBlock;
         block += kWarps) {
      const int base = block * kMainQuantBlock;
      float values[kValuesPerLane];
      float amax = 0.0f;
#pragma unroll
      for (int index = 0; index < kValuesPerLane; ++index) {
        const int column = base + index * kLanes + lane;
        const float reduced = __bfloat162float(reduced_row[column]);
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            reduced * inverse_rms * stage.norm_weight[column]);
        stage.normalized[row * kMainHidden + column] = normalized;
        values[index] = __bfloat162float(normalized);
        amax = fmaxf(amax, fabsf(values[index]));
      }
#pragma unroll
      for (int delta = kLanes / 2; delta > 0; delta /= 2) {
        amax = fmaxf(
            amax, __shfl_xor_sync(0xffffffffu, amax, delta));
      }
      amax = fmaxf(amax, 1.0e-4f);
      const int exponent = dspark_quant_scale::exponent(amax);
      if (lane == 0) {
        const int raw_encoded = exponent + 127;
        stage.activation_scales[
            row * (kMainHidden / kMainQuantBlock) + block] =
                static_cast<uint8_t>(
                    raw_encoded < 0
                        ? 0
                        : (raw_encoded > 254 ? 254 : raw_encoded));
      }
      const float scale = dspark_quant_scale::scale(exponent);
#pragma unroll
      for (int index = 0; index < kValuesPerLane; ++index) {
        stage.quantized[row * kMainHidden + base + index * kLanes + lane] =
            convert(values[index] / scale);
      }
    }
    __syncthreads();
#else
    // R1 leg 6, intra-node retile: the RMS pass squared exactly the bf16 value
    // the stream-reduce pass had just stored, over the same columns, in the
    // same thread. It rides the reduce instead of re-walking the row. The
    // square consumes the ROUNDED value (`stored`), not the fp32 accumulator,
    // so the sum is bit-for-bit the read-back version.
    float local_square = 0.0f;
    #pragma unroll 4
    for (int column = threadIdx.x; column < kMainHidden; column += kBlockThreads) {
      float reduced = 0.0f;
#pragma unroll
      for (int stream = 0; stream < kHcStreams; ++stream) {
        reduced = fmaf(
            stage.pre[row * kHcStreams + stream],
            __bfloat162float(
                stage.streams[row * kRowElements + stream * kMainHidden + column]),
            reduced);
      }
      const __nv_bfloat16 stored = __float2bfloat16_rn(reduced);
      stage.normalized[row * kMainHidden + column] = stored;
      const float value = __bfloat162float(stored);
      local_square = fmaf(value, value, local_square);
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    dspark_reduction::sum_256(shared);
    const float inverse_rms = rsqrtf(shared[0] / kMainHidden + kMainNormEpsilon);
    #pragma unroll 8
    for (int column = threadIdx.x; column < kMainHidden; column += kBlockThreads) {
      const float value =
          __bfloat162float(stage.normalized[row * kMainHidden + column]);
      stage.normalized[row * kMainHidden + column] =
          __float2bfloat16_rn(value * inverse_rms * stage.norm_weight[column]);
    }
    __syncthreads();
    quantize_bf16_blocks(
        stage.normalized + row * kMainHidden,
        stage.quantized + row * kMainHidden,
        stage.activation_scales + row * (kMainHidden / kMainQuantBlock),
        kMainHidden / kMainQuantBlock,
        shared);
#endif
  }
}

#ifdef DSPARK_HC_SPLIT
// The pre-gates are independent of Sinkhorn's combination matrix. Recompute
// them in private shared memory so this task neither reads nor races with
// the concurrently executing Sinkhorn task's global pre/post/comb stores.
__device__ void execute_independent_hc_norm(
    const ClaimedTask& task, Layer0AttnHcStage stage, float* shared) {
  auto* cached_streams = reinterpret_cast<__nv_bfloat16*>(shared + 4096);
  auto* cached_norm = shared + 12288;
  float* local_pre = shared + 2304;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    Layer0AttnHcStage local = stage;
    local.streams += item * 16384;
    local.mixes += item * 24;
    local.pre = local_pre;
    local.normalized += item * 4096;
    local.quantized += item * 4096;
    local.activation_scales += item * 32;
    if (threadIdx.x < 4) {
      const int lane = threadIdx.x;
      local_pre[lane] = sigmoidf(local.mixes[lane] * local.scale[0]
                                + local.base[lane]) + 1.0e-6f;
    }
    for (int p = threadIdx.x; p < 2048; p += 256)
      cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
          cached_streams + p * 8, local.streams + p * 8);
    for (int p = threadIdx.x; p < 1024; p += 256)
      cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
          cached_norm + p * 4, local.norm_weight + p * 4);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    local.streams = cached_streams;
    local.norm_weight = cached_norm;
    ClaimedTask one = task;
    one.begin = 0;
    one.end = 1;
    execute_layer0_attn_hc_reduce_norm(one, local, shared);
  }
}
#endif

#ifdef DSPARK_HC_PREFETCH
// Warp zero runs Sinkhorn while the other warps stage its dependent reduction
// inputs. Scratch lives beyond the reduction buffers and is CTA-private.
__device__ void execute_prefetched_hc(
    const ClaimedTask& task, Layer0AttnHcStage stage, float* shared) {
  auto* cached_streams = reinterpret_cast<__nv_bfloat16*>(shared + 4096);
  auto* cached_norm = shared + 12288;
  static_assert(65536 <= kDynamicSharedBytes);
  for (uint32_t item = task.begin; item < task.end; ++item) {
    Layer0AttnHcStage local = stage;
    local.streams += item * 16384;
    local.mixes += item * 24;
    local.pre += item * 4;
    local.post += item * 4;
    local.comb += item * 16;
    local.normalized += item * 4096;
    local.quantized += item * 4096;
    local.activation_scales += item * 32;
    ClaimedTask one = task;
    one.begin = 0;
    one.end = 1;
    if (threadIdx.x >= 32) {
      const int lane = threadIdx.x - 32;
      for (int p = lane; p < 2048; p += 224)
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            cached_streams + p * 8, local.streams + p * 8);
      for (int p = lane; p < 1024; p += 224)
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            cached_norm + p * 4, local.norm_weight + p * 4);
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
    }
#ifdef DSPARK_HC_COMPUTE_OVERLAP
    if (threadIdx.x < 32) {
      execute_layer0_attn_hc_sinkhorn(one, local);
    } else {
      local.streams = cached_streams;
      local.norm_weight = cached_norm;
      dspark_hc_overlap::reduce_norm(local, shared);
    }
    __syncthreads();
#else
    execute_layer0_attn_hc_sinkhorn(one, local);
    __syncthreads();
    local.streams = cached_streams;
    local.norm_weight = cached_norm;
    execute_layer0_attn_hc_reduce_norm(one, local, shared);
#endif
  }
}
#endif

__device__ DSPARK_OTHER_ENTRY void execute_layer0_attn_hc_phase(
    const ClaimedTask& task,
    Layer0AttnHcStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase != generated::kAttnHcProjectionPhases[layer]
        && task.phase != generated::kAttnHcSinkhornPhases[layer]
        && task.phase != generated::kAttnHcReduceNormPhases[layer]) {
      continue;
    }
    if (!layer_enabled(stage.layer_mask, layer)) {
      return;
    }
    stage.fn_weight = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kAttnHcFnSlots,
        layer);
    stage.base = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kAttnHcBaseSlots,
        layer);
    stage.scale = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kAttnHcScaleSlots,
        layer);
    stage.norm_weight = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kAttnNormWeightSlots,
        layer);
    if (task.phase == generated::kAttnHcProjectionPhases[layer]) {
      execute_layer0_attn_hc_projection(task, stage, shared);
    } else if (task.phase == generated::kAttnHcSinkhornPhases[layer]) {
#ifdef DSPARK_HC_SPLIT
      execute_layer0_attn_hc_sinkhorn(task, stage);
#elif defined(DSPARK_HC_PREFETCH)
      execute_prefetched_hc(task, stage, shared);
#else
      execute_layer0_attn_hc_sinkhorn(task, stage);
#ifdef DSPARK_V4_RELAXED_DAG
      // Stage-2b fusion: reduce_rmsnorm rides the sinkhorn phase.
      __syncthreads();
      execute_layer0_attn_hc_reduce_norm(task, stage, shared);
#endif
#endif
    } else {
#ifdef DSPARK_HC_SPLIT
      execute_independent_hc_norm(task, stage, shared);
#else
      execute_layer0_attn_hc_reduce_norm(task, stage, shared);
#endif
    }
    return;
  }
}

__device__ void execute_layer0_qa_projection(
    const ClaimedTask& task,
    Layer0AttnProjectionStage stage) {
  // Body extracted verbatim into dspark_small_phase.cuh (relaxed-drafter
  // stage 3, small dense bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion; serves contract AND
  // relaxed builds.
  dspark_small::QaArgs args;
  args.input_quantized = stage.input_quantized;
  args.input_scales = stage.input_scales;
  args.wqa = stage.wqa;
  args.wqa_scales = stage.wqa_scales;
  args.qa_partials = stage.qa_partials;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
#ifdef DSPARK_DENSE_BLOCK_SCALED
  dspark_fp8_scaled::execute<DSPARK_DENSE_BLOCK_SCALED, 4096, 1024, 5, 4, false>(
      {args.input_quantized, args.input_scales,
       args.wqa, args.wqa_scales,
       args.qa_partials, nullptr, args.begin, args.end});
#else
  dspark_small::execute_qa_tcgen(args);
#endif
#else
  dspark_small::execute_qa_reference(args);
#endif
}

__device__ void execute_layer0_qa_norm(
    const ClaimedTask& task,
    Layer0AttnProjectionStage stage,
    float* shared) {
  constexpr int kRank = 1024;
  constexpr int kSplits = 4;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int row = static_cast<int>(item);
    #pragma unroll
    for (int column = threadIdx.x; column < kRank; column += kBlockThreads) {
      float value = 0.0f;
#pragma unroll
      for (int split = 0; split < kSplits; ++split) {
        value += stage.qa_partials[(row * kRank + column) * kSplits + split];
      }
      stage.q_lora[row * kRank + column] = __float2bfloat16_rn(value);
    }
    __syncthreads();

    float local_square = 0.0f;
    #pragma unroll
    for (int column = threadIdx.x; column < kRank; column += kBlockThreads) {
      const float value = __bfloat162float(stage.q_lora[row * kRank + column]);
      local_square = fmaf(value, value, local_square);
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    dspark_reduction::sum_256(shared);
    const float inverse_rms = rsqrtf(shared[0] / kRank + kMainNormEpsilon);
    // Issue the four independent norm-weight loads before any output store.
    float norm_weights[kRank / kBlockThreads];
    #pragma unroll
    for (int j = 0; j < kRank / kBlockThreads; ++j)
      norm_weights[j] = stage.qa_norm_weight[threadIdx.x + j * kBlockThreads];
    #pragma unroll
    for (int j = 0; j < kRank / kBlockThreads; ++j) {
      const int column = threadIdx.x + j * kBlockThreads;
      const float value = __bfloat162float(stage.q_lora[row * kRank + column]);
      stage.q_lora[row * kRank + column] =
          __float2bfloat16_rn(value * inverse_rms * norm_weights[j]);
    }
    __syncthreads();
    quantize_bf16_blocks(
        stage.q_lora + row * kRank,
        stage.q_lora_quantized + row * kRank,
        stage.q_lora_scales + row * (kRank / kMainQuantBlock),
        kRank / kMainQuantBlock,
        shared);
  }
}

__device__ void execute_layer0_qb_projection(
    const ClaimedTask& task,
    Layer0AttnProjectionStage stage) {
  // Body extracted verbatim into dspark_proj_phase.cuh (relaxed-drafter
  // stage 3, dense-projection bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion; serves both the contract
  // and relaxed builds.
  dspark_proj::QbArgs args;
  args.q_lora_quantized = stage.q_lora_quantized;
  args.q_lora_scales = stage.q_lora_scales;
  args.wqb = stage.wqb;
  args.wqb_scales = stage.wqb_scales;
  args.query_projection = stage.query_projection;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_TP2_ABLATE)
  // q_b column split: 256 tiles of 128 columns, head = item / 4.
  dspark_tp2_ablate::clip_half(
      256, dspark_tp2_ablate::kBandAttention, args.begin, args.end);
#endif
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
#if DSPARK_V4_BATCH == 1
#if defined(DSPARK_QB_BLOCK_SCALED)
  dspark_fp8_scaled::qb<DSPARK_QB_BLOCK_SCALED>(args);
#elif defined(DSPARK_PROJ_K128_RING)
  dspark_proj::execute_qb_k128_ring(args);
#else
  dspark_proj::execute_qb_tcgen(args);
#endif
#else
  // Batched serving (R4 band 2): 33.5 MiB of FP8 q_b weights read ONCE for
  // the whole batch; the 5*batch flat rows ride the UMMA N mode and the item
  // count stays at the batch-invariant 256 output tiles.
  dspark_shared::execute_fp8_tcgen_wide<
      dspark_proj::kQbRank, 1, dspark_proj::kWideRows,
      dspark_proj::kRealRows, dspark_proj::kQbWidth, true>(
      args.q_lora_quantized,
      args.q_lora_scales,
      args.wqb,
      args.wqb_scales,
      nullptr,
      args.query_projection,
      args.begin,
      args.end);
#endif
#else
  dspark_proj::execute_qb_reference(args);
#endif
}

__device__ void execute_layer0_query_norm_rope(
    const ClaimedTask& task,
    Layer0AttnProjectionStage stage,
    float* shared) {
  // Body extracted verbatim into dspark_qnorm_phase.cuh (relaxed-drafter
  // stage 3, q_norm_rope band) so the bench lane compiles exactly the
  // production implementation. Pure code motion; serves both the contract
  // and relaxed builds.
  dspark_qnorm::QNormArgs args;
  args.query_projection = stage.query_projection;
  args.query_inverse_rms = stage.query_inverse_rms;
  args.queries = stage.queries;
  args.rope_cos_sin = stage.rope_cos_sin;
  args.begin = task.begin;
  args.end = task.end;
#ifdef DSPARK_V4_RELAXED_DAG
  // Stage-3 iter-9: bitwise batched body; relaxed DAG carries 40 items.
  dspark_qnorm::execute_qnorm_batched<8>(args);
#else
  dspark_qnorm::execute_qnorm_reference(args, shared);
#endif
}

__device__ void execute_layer0_draft_kv_projection(
    const ClaimedTask& task,
    Layer0AttnProjectionStage stage) {
  // Body extracted verbatim into dspark_small_phase.cuh (relaxed-drafter
  // stage 3, small dense bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion; serves contract AND
  // relaxed builds.
  dspark_small::DraftKvArgs args;
  args.input_quantized = stage.input_quantized;
  args.input_scales = stage.input_scales;
  args.wkv = stage.wkv;
  args.wkv_scales = stage.wkv_scales;
  args.draft_kv_partials = stage.draft_kv_partials;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
#ifdef DSPARK_DENSE_BLOCK_SCALED
  dspark_fp8_scaled::execute<DSPARK_DENSE_BLOCK_SCALED, 4096, 512, 5, 4, false>(
      {args.input_quantized, args.input_scales,
       args.wkv, args.wkv_scales,
       args.draft_kv_partials, nullptr, args.begin, args.end});
#else
  dspark_small::execute_draft_kv_tcgen(args);
#endif
#else
  dspark_small::execute_draft_kv_reference(args);
#endif
}

__device__ void execute_layer0_draft_kv_finalize(
    const ClaimedTask& task,
    Layer0AttnProjectionStage stage,
    float* shared) {
  // Body extracted verbatim into dspark_qnorm_phase.cuh (relaxed-drafter
  // stage 3, draft_kv_norm_rope_quant diagnostic) so the bench lane
  // compiles exactly the production implementation. Pure code motion;
  // serves both the contract and relaxed builds.
  dspark_qnorm::DraftKvNormArgs args;
  args.draft_kv_partials = stage.draft_kv_partials;
  args.kv_norm_weight = stage.kv_norm_weight;
  args.draft_kv = stage.draft_kv;
  args.rope_cos_sin = stage.rope_cos_sin;
  args.begin = task.begin;
  args.end = task.end;
  dspark_qnorm::execute_draft_kv_norm_reference(args, shared);
}

__device__ DSPARK_OTHER_ENTRY void execute_layer0_attn_projection_phase(
    const ClaimedTask& task,
    Layer0AttnProjectionStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase != generated::kQaPhases[layer]
        && task.phase != generated::kQaNormPhases[layer]
        && task.phase != generated::kQbPhases[layer]
        && task.phase != generated::kDraftKvProjectionPhases[layer]
        && task.phase != generated::kQueryNormRopePhases[layer]
        && task.phase != generated::kDraftKvFinalizePhases[layer]) {
      continue;
    }
    if (!layer_enabled(stage.layer_mask, layer)) {
      return;
    }
    stage.wqa = layer_weight<cutlass::float_e4m3_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kWqaWeightSlots,
        layer);
    stage.wqa_scales = layer_weight<uint8_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kWqaScaleSlots,
        layer);
    stage.qa_norm_weight = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kQaNormWeightSlots,
        layer);
    stage.wqb = layer_weight<cutlass::float_e4m3_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kWqbWeightSlots,
        layer);
    stage.wqb_scales = layer_weight<uint8_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kWqbScaleSlots,
        layer);
    stage.wkv = layer_weight<cutlass::float_e4m3_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kDraftWkvWeightSlots,
        layer);
    stage.wkv_scales = layer_weight<uint8_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kDraftWkvScaleSlots,
        layer);
    stage.kv_norm_weight = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kDraftKvNormWeightSlots,
        layer);
    if (task.phase == generated::kQaPhases[layer]) {
      execute_layer0_qa_projection(task, stage);
    } else if (task.phase == generated::kQaNormPhases[layer]) {
      execute_layer0_qa_norm(task, stage, shared);
    } else if (task.phase == generated::kQbPhases[layer]) {
      execute_layer0_qb_projection(task, stage);
    } else if (task.phase == generated::kDraftKvProjectionPhases[layer]) {
      execute_layer0_draft_kv_projection(task, stage);
    } else if (task.phase == generated::kQueryNormRopePhases[layer]) {
      execute_layer0_query_norm_rope(task, stage, shared);
    } else {
      execute_layer0_draft_kv_finalize(task, stage, shared);
    }
    return;
  }
}

__device__ void execute_layer0_sparse_attention(
    const ClaimedTask& task,
    Layer0SparseAttentionStage stage,
    float* shared) {
  // Body extracted verbatim into dspark_attn_phase.cuh (relaxed program
  // stage 3) so the phase microbenchmark compiles exactly the production
  // implementation; both contract and relaxed builds run this reference.
  dspark_attn::Args args;
  args.queries = stage.queries;
  args.target_kv = stage.target_kv;
  args.draft_kv = stage.draft_kv;
  args.accumulator = stage.accumulator;
  args.raw_output = stage.raw_output;
  args.output = stage.output;
  args.attention_sink = stage.attention_sink;
  args.rope_cos_sin = stage.rope_cos_sin;
  args.start_pos = stage.start_pos;
  args.begin = task.begin;
  args.end = task.end;
#ifdef DSPARK_V4_RELAXED_DAG
  // Stage-3 iter-4: 8 heads per item (relaxed DAG carries 40 attention items,
  // not 320); bitwise-identical to the reference body per the bench oracle.
  //
  // R6: kOpt adds the three body-efficiency levers the attribution probe
  // pointed at. The probe split the 46.4 us body into PV wmma 16.5 / QK wmma
  // 14.0 / KV staging 7.9 / softmax 4.4 / rescale 2.0 / barriers 1.4, i.e.
  // 66% of it was shared-memory fragment traffic in the two wmma stages --
  // every tile sat at a leading dimension that is a multiple of 128 B, which
  // is an 8-way bank conflict on every wmma::load_matrix_sync. Padding the
  // strides is 1.70x on its own; cp.async staging and the hoisted PV weight
  // fragments take the band to 2.06x. All three move DATA only, so the body
  // stays bitwise-identical to execute_reference (bench oracle: 0 mismatches
  // on output and raw_output at start_pos 511 and 40).
  dspark_attn::execute_batched<
      8,
      true,
      0,
      dspark_attn::opt::kPadShared | dspark_attn::opt::kCpAsync
          | dspark_attn::opt::kHoistWeights
#ifdef DSPARK_ATTN_MMA8
          | dspark_attn::opt::kMma8
#endif
          >(args, shared);
#else
  dspark_attn::execute_reference(args, shared);
#endif
}

__device__ DSPARK_OTHER_ENTRY void execute_layer0_sparse_attention_phase(
    const ClaimedTask& task,
    Layer0SparseAttentionStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase != generated::kSparseAttentionPhases[layer]) {
      continue;
    }
    if (!layer_enabled(stage.layer_mask, layer)) {
      return;
    }
    // kv_cache is (layers, batch, window, kv_width); the body adds the
    // per-element stride itself.
    stage.target_kv += layer * kBatch * 128 * 512;
    stage.attention_sink = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kAttnSinkWeightSlots,
        layer);
    execute_layer0_sparse_attention(task, stage, shared);
    return;
  }
}

__device__ void execute_layer0_output_a_projection(
    const ClaimedTask& task,
    Layer0AttentionOutputStage stage,
    float* shared) {
  // Body extracted verbatim into dspark_proj_phase.cuh (relaxed-drafter
  // stage 3, dense-projection bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion; serves both the contract
  // and relaxed builds. Item = (group, rank_tile, row) triple (#102 retile).
#ifdef DSPARK_WOA_SPLIT_K
  dspark_woa_splitk::Args args;
  args.partials = stage.output_partials;
#else
  dspark_proj::WoaArgs args;
#endif
  args.attention = stage.attention;
  args.wo_a = stage.wo_a;
  args.output_lora = stage.output_lora;
  args.output_lora_quantized = stage.output_lora_quantized;
  args.output_lora_scales = stage.output_lora_scales;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_TP2_ABLATE)
  // wo_a group split: item = group * 8 + rank_tile, group g owns heads 8g..8g+7.
  dspark_tp2_ablate::clip_half(
      64, dspark_tp2_ablate::kBandAttention, args.begin, args.end);
#endif
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
  // Relaxed DAG carries 64 wo_a items: item = group * 8 + rank_tile.
#if DSPARK_V4_BATCH == 1
#ifdef DSPARK_WOA_SPLIT_K
  dspark_woa_splitk::execute<256, DSPARK_WOA_SPLIT_K>(args);
#elif defined(DSPARK_WOA_STREAMED_K)
  dspark_woa_streamed::execute<DSPARK_WOA_STREAMED_K>(args);
#elif defined(DSPARK_WOA_BULK_STAGING)
  dspark_proj::execute_woa_umma<true>(args);
#else
  dspark_proj::execute_woa_umma(args);
#endif
#else
  // Batched serving (R4 band 1): the SAME 64 MiB wo_a stream, 5*batch flat
  // draft rows on the UMMA N mode. Item count stays batch-invariant at 64.
  dspark_proj::execute_woa_umma_batch<
      dspark_proj::kWideRows, dspark_proj::kRealRows>(args);
#endif
#else
  dspark_proj::execute_woa_reference(args, shared);
#endif
}

__device__ void execute_layer0_output_b_projection(
    const ClaimedTask& task,
    Layer0AttentionOutputStage stage) {
  // Body extracted verbatim into dspark_proj_phase.cuh (relaxed-drafter
  // stage 3, dense-projection bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion; serves both the contract
  // and relaxed builds.
  dspark_proj::WobArgs args;
  args.output_lora_quantized = stage.output_lora_quantized;
  args.output_lora_scales = stage.output_lora_scales;
  args.wo_b = stage.wo_b;
  args.wo_b_scales = stage.wo_b_scales;
  args.output_partials = stage.output_partials;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
#if DSPARK_V4_BATCH == 1
#ifdef DSPARK_DENSE_BLOCK_SCALED
  dspark_fp8_scaled::execute<DSPARK_DENSE_BLOCK_SCALED, 8192, 4096, 5, 4, false>(
      {args.output_lora_quantized, args.output_lora_scales,
       args.wo_b, args.wo_b_scales,
       args.output_partials, nullptr, args.begin, args.end});
#else
  dspark_proj::execute_wob_tcgen(args);
#endif
#else
  // Batched serving (R4 band 3): 33.5 MiB of FP8 wo_b weights read ONCE for
  // the whole batch; the 5*batch flat rows ride the UMMA N mode and the item
  // count stays at the batch-invariant 128 (output tile, split) pairs.
  dspark_shared::execute_fp8_tcgen_wide<
      dspark_proj::kWobInput, dspark_proj::kWobSplits, dspark_proj::kWideRows,
      dspark_proj::kRealRows, dspark_proj::kWobHidden, false>(
      args.output_lora_quantized,
      args.output_lora_scales,
      args.wo_b,
      args.wo_b_scales,
      args.output_partials,
      nullptr,
      args.begin,
      args.end);
#endif
#else
  dspark_proj::execute_wob_reference(args);
#endif
}

__device__ void execute_layer0_attn_post(
    const ClaimedTask& task,
    Layer0AttentionOutputStage stage) {
  constexpr int kSplits = 4;
  constexpr int kTile = 512;
  constexpr int kElements = kDraftRows * kMainHidden;
  constexpr int kRowElements = kHcStreams * kMainHidden;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int tile_base = static_cast<int>(item) * kTile;
    #pragma unroll
    for (int lane_item = threadIdx.x; lane_item < kTile; lane_item += kBlockThreads) {
      const int index = tile_base + lane_item;
      if (index >= kElements) {
        continue;
      }
      float projected = 0.0f;
#pragma unroll
      for (int split = 0; split < kSplits; ++split) {
        projected += stage.output_partials[index * kSplits + split];
      }
      const __nv_bfloat16 rounded = __float2bfloat16_rn(projected);
      stage.output[index] = rounded;
      const float attention_value = __bfloat162float(rounded);
      const int row = index / kMainHidden;
      const int column = index % kMainHidden;
      float residual[kHcStreams];
#pragma unroll
      for (int stream = 0; stream < kHcStreams; ++stream) {
        residual[stream] = __bfloat162float(
            stage.streams[row * kRowElements + stream * kMainHidden + column]);
      }
      __nv_bfloat16 updated[kHcStreams];
#pragma unroll
      for (int output_stream = 0; output_stream < kHcStreams; ++output_stream) {
        float residual_sum = 0.0f;
#pragma unroll
        for (int input_stream = 0; input_stream < kHcStreams; ++input_stream) {
          residual_sum = fmaf(
              stage.comb[
                  (row * kHcStreams + input_stream) * kHcStreams + output_stream],
              residual[input_stream],
              residual_sum);
        }
        updated[output_stream] = __float2bfloat16_rn(
            stage.post[row * kHcStreams + output_stream] * attention_value
            + residual_sum);
      }
#pragma unroll
      for (int stream = 0; stream < kHcStreams; ++stream) {
        const int stream_index =
            row * kRowElements + stream * kMainHidden + column;
#ifndef DSPARK_V4_RELAXED_DAG
        // Dead half of this band edge: the live consumer is the snapshot.
        // Every phase between here and expert_combine reads
        // AttentionHiddenStreams (ffn HC projection, the combine residual),
        // and expert_combine overwrites ALL 5 x 4 x 4096 HiddenStreams
        // elements before the next layer's attn HC projection or
        // head_reduce reads them. The relaxed program drops the write; the
        // contract build keeps it byte-for-byte.
        stage.streams[stream_index] = updated[stream];
#endif
        stage.streams_snapshot[stream_index] = updated[stream];
      }
    }
  }
}

__device__ DSPARK_OTHER_ENTRY void execute_layer0_attention_output_phase(
    const ClaimedTask& task,
    Layer0AttentionOutputStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase != generated::kOutputAProjectionPhases[layer]
#ifdef DSPARK_WOA_SPLIT_K
        && task.phase != generated::kOutputAReductionPhases[layer]
#endif
        && task.phase != generated::kOutputBProjectionPhases[layer]
        && task.phase != generated::kAttnPostPhases[layer]) {
      continue;
    }
    if (!layer_enabled(stage.layer_mask, layer)) {
      return;
    }
    stage.wo_a = layer_weight<__nv_bfloat16>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kWoaWeightSlots,
        layer);
    stage.wo_b = layer_weight<cutlass::float_e4m3_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kWobWeightSlots,
        layer);
    stage.wo_b_scales = layer_weight<uint8_t>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kWobScaleSlots,
        layer);
    if (task.phase == generated::kOutputAProjectionPhases[layer]) {
      execute_layer0_output_a_projection(task, stage, shared);
    }
#ifdef DSPARK_WOA_SPLIT_K
    else if (task.phase == generated::kOutputAReductionPhases[layer]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
      dspark_woa_splitk::Args args{};
      args.partials = stage.output_partials;
      args.output_lora = stage.output_lora;
      args.output_lora_quantized = stage.output_lora_quantized;
      args.output_lora_scales = stage.output_lora_scales;
      args.begin = task.begin;
      args.end = task.end;
      dspark_woa_splitk::reduce<DSPARK_WOA_SPLIT_K>(args);
#endif
    }
#endif
    else if (task.phase == generated::kOutputBProjectionPhases[layer]) {
      execute_layer0_output_b_projection(task, stage);
    } else {
      execute_layer0_attn_post(task, stage);
    }
    return;
  }
}

__device__ DSPARK_OTHER_ENTRY void execute_layer0_ffn_hc_phase(
    const ClaimedTask& task,
    Layer0AttnHcStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase != generated::kFfnHcProjectionPhases[layer]
        && task.phase != generated::kFfnHcSinkhornPhases[layer]
        && task.phase != generated::kFfnHcReduceNormPhases[layer]) {
      continue;
    }
    if (!layer_enabled(stage.layer_mask, layer)) {
      return;
    }
    stage.fn_weight = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kFfnHcFnSlots,
        layer);
    stage.base = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kFfnHcBaseSlots,
        layer);
    stage.scale = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kFfnHcScaleSlots,
        layer);
    stage.norm_weight = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kFfnNormWeightSlots,
        layer);
    if (task.phase == generated::kFfnHcProjectionPhases[layer]) {
      execute_layer0_attn_hc_projection(task, stage, shared);
    } else if (task.phase == generated::kFfnHcSinkhornPhases[layer]) {
#ifdef DSPARK_HC_SPLIT
      execute_layer0_attn_hc_sinkhorn(task, stage);
#elif defined(DSPARK_HC_PREFETCH)
      execute_prefetched_hc(task, stage, shared);
#else
      execute_layer0_attn_hc_sinkhorn(task, stage);
#ifdef DSPARK_V4_RELAXED_DAG
      __syncthreads();
      execute_layer0_attn_hc_reduce_norm(task, stage, shared);
#endif
#endif
    } else {
#ifdef DSPARK_HC_SPLIT
      execute_independent_hc_norm(task, stage, shared);
#else
      execute_layer0_attn_hc_reduce_norm(task, stage, shared);
#endif
    }
    return;
  }
}

__device__ void execute_layer0_router_scores(
    const ClaimedTask& task,
    Layer0RouterStage stage) {
#if defined(DSPARK_V4_RELAXED_DAG)
  // R12, relaxed build only: the 4,096-step chain is K-SPLIT across
  // kRouterSplit threads and closed with a warp-shuffle tree
  // (dspark_router_phase.cuh). This REASSOCIATES the FP32 sum -- it cannot
  // be bitwise and is not claimed to be -- and rides the same
  // acceptance-parity gate as every other relaxed body. The shipped tiling
  // is reproduced exactly by <40, 1, 1, 4>, which is the in-tree control.
  //
  // The generated program and the body must agree on the chain geometry or
  // items would silently drop chains, so the header carries the counts the
  // schedule was generated with and they are asserted here.
  static_assert(
      generated::kRouterChainsPerItem == kRouterChainsPerItem,
      "regenerate the V4 headers after changing DSPARK_ROUTER_CHAINS");
  static_assert(
      generated::kRouterRowsPerChain == kRouterRowsPerChain,
      "regenerate the V4 headers after changing DSPARK_ROUTER_ROWS");
  static_assert(
      generated::kThreads == dspark_router::kRouterThreads,
      "the router bodies index threads with a compile-time 256-lane width");
  static_assert(
      kMainHidden == dspark_router::kRouterHidden
          && kDraftBlock == dspark_router::kRouterBlock,
      "router band shape");
  dspark_router::RouterScoreArgs args;
  args.gate = stage.gate;
  args.input = stage.input;
  args.scores = stage.scores;
  args.begin = task.begin;
  args.end = task.end;
  dspark_router::execute_router_scores_split<
      kRouterChainsPerItem,
      kRouterSplit,
      kRouterRowsPerChain,
      kRouterUnroll>(args);
#else
  // Retiled (#101): one task item covers kRouterExpertsPerTile experts and
  // every (expert, row) pair runs as its own independent serial chain on its
  // own thread. Each chain's FMA order is the original ascending-column
  // walk, so partials are bitwise-identical to the 128-expert-per-thread
  // body; the phase just exposes 32 tiles x 40 chains instead of 2 x 128.
  // Gate and input rows stream through 16-byte aligned loads (#84 pattern);
  // the values are consumed in ascending order, leaving the chain intact.
  constexpr int kExperts = 256;
  constexpr int kRouterExpertsPerTile = 8;
  constexpr int kChains = kRouterExpertsPerTile * kDraftBlock;
  static_assert(kChains <= 256, "one chain per thread");
  constexpr int kTilesPerElement = kExperts / kRouterExpertsPerTile;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int chain = static_cast<int>(threadIdx.x);
    if (chain >= kChains) {
      continue;
    }
    // Batched serving: the phase carries a `batch *` factor, so one item is
    // one (batch element, expert tile) pair and the chain count stays 40. A
    // batch factor is MANDATORY here -- 8 experts x 5 rows x batch chains
    // would exceed the CTA's 256 threads at batch > 6 and silently drop rows.
    const int expert_tile = static_cast<int>(item) % kTilesPerElement;
    const int element = static_cast<int>(item) / kTilesPerElement;
    const int expert =
        expert_tile * kRouterExpertsPerTile + chain / kDraftBlock;
    const int row = element * kDraftBlock + chain % kDraftBlock;
    if (expert >= kExperts) {
      continue;
    }
    const __nv_bfloat16* gate_row = stage.gate + expert * kMainHidden;
    const __nv_bfloat16* input_row = stage.input + row * kMainHidden;
    // R1 leg 6, intra-node retile: the chain is one thread's 4,096-element
    // dependent walk and only 40 of the CTA's 256 threads own a chain, so the
    // band is pure per-thread load latency -- more warps cannot hide it, only
    // more loads in flight per thread can. Four 16-byte pairs are issued up
    // front and then consumed; the fmaf sequence is still strictly ascending
    // column order, so the chain is bitwise what the one-pair-at-a-time shape
    // produced.
    constexpr int kPacked = 8;
    constexpr int kUnroll = 4;
    static_assert(kMainHidden % (kPacked * kUnroll) == 0);
    float logit = 0.0f;
    for (int column = 0; column < kMainHidden; column += kPacked * kUnroll) {
      uint4 packed_gate[kUnroll];
      uint4 packed_input[kUnroll];
#pragma unroll
      for (int step = 0; step < kUnroll; ++step) {
        packed_gate[step] =
            *reinterpret_cast<const uint4*>(gate_row + column + step * kPacked);
        packed_input[step] =
            *reinterpret_cast<const uint4*>(input_row + column + step * kPacked);
      }
#pragma unroll
      for (int step = 0; step < kUnroll; ++step) {
        const __nv_bfloat16* gate_values =
            reinterpret_cast<const __nv_bfloat16*>(&packed_gate[step]);
        const __nv_bfloat16* input_values =
            reinterpret_cast<const __nv_bfloat16*>(&packed_input[step]);
#pragma unroll
        for (int vector_offset = 0; vector_offset < kPacked; ++vector_offset) {
          logit = fmaf(
              __bfloat162float(input_values[vector_offset]),
              __bfloat162float(gate_values[vector_offset]),
              logit);
        }
      }
    }
    const float softplus = logit > 20.0f ? logit : log1pf(expf(logit));
    stage.scores[row * kExperts + expert] = sqrtf(softplus);
  }
#endif
}

__device__ void execute_layer0_router_top6(
    const ClaimedTask& task,
    Layer0RouterStage stage,
    float* shared) {
  // Parallelized (#105): the serial 6x256 selection scan becomes six
  // 256-thread argmax reductions. The serial ascending scan keeps the
  // LOWEST index on exact float ties; the reduction uses the identical
  // (value greater, tie -> lower index) rule on the identical fp32
  // `score + bias` sums, so every selection — and therefore the serial
  // selection-order renormalization — is bitwise unchanged.
  constexpr int kExperts = 256;
  constexpr int kTop = 6;
  int* shared_index = reinterpret_cast<int*>(shared + kBlockThreads);
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int row = static_cast<int>(item);
#if DSPARK_ROUTER_WARP_TOP6 == 2
    if (dspark_router::top6_single_warp(
            stage.scores + row * kExperts, stage.bias,
            stage.indices + row * kTop, stage.weights + row * kTop)) continue;
#endif
    const int expert = static_cast<int>(threadIdx.x);
    float adjusted = -FLT_MAX;
    if (expert < kExperts) {
      adjusted = stage.scores[row * kExperts + expert] + stage.bias[expert];
    }
    bool already_selected = false;
    for (int selected = 0; selected < kTop; ++selected) {
      shared[threadIdx.x] = already_selected ? -FLT_MAX : adjusted;
      shared_index[threadIdx.x] = expert < kExperts ? expert : kExperts;
      __syncthreads();
#ifdef DSPARK_ROUTER_WARP_TOP6
      dspark_reduction::argmax_256<true>(shared, shared_index);
#else
      #pragma unroll
      for (int delta = kBlockThreads / 2; delta > 0; delta /= 2) {
        if (threadIdx.x < delta) {
          const float other = shared[threadIdx.x + delta];
          const int other_index = shared_index[threadIdx.x + delta];
          if (other > shared[threadIdx.x]
              || (other == shared[threadIdx.x]
                  && other_index < shared_index[threadIdx.x])) {
            shared[threadIdx.x] = other;
            shared_index[threadIdx.x] = other_index;
          }
        }
        __syncthreads();
      }
#endif
      const int best_expert = shared_index[0];
      if (expert == best_expert) {
        already_selected = true;
      }
      if (threadIdx.x == 0) {
        stage.indices[row * kTop + selected] = best_expert;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float sum = 0.0f;
      for (int selected = 0; selected < kTop; ++selected) {
        const int chosen = stage.indices[row * kTop + selected];
        const float weight = stage.scores[row * kExperts + chosen];
        stage.weights[row * kTop + selected] = weight;
        sum += weight;
      }
      for (int selected = 0; selected < kTop; ++selected) {
        stage.weights[row * kTop + selected] =
            1.5f * stage.weights[row * kTop + selected] / sum;
      }
    }
    __syncthreads();
  }
}

__device__ DSPARK_OTHER_ENTRY void execute_layer0_router_phase(
    const ClaimedTask& task,
    Layer0RouterStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase != generated::kRouterScorePhases[layer]
        && task.phase != generated::kRouterTop6Phases[layer]) {
      continue;
    }
    if (!layer_enabled(stage.layer_mask, layer)) {
      return;
    }
    stage.gate = layer_weight<__nv_bfloat16>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kRouterWeightSlots,
        layer);
    stage.bias = layer_weight<float>(
        stage.weight_arena,
        stage.weight_offsets,
        generated::kRouterBiasSlots,
        layer);
    if (task.phase == generated::kRouterScorePhases[layer]) {
      execute_layer0_router_scores(task, stage);
    } else {
#ifdef DSPARK_ROUTED_GROUP_READY
      if (task.begin == 0 && threadIdx.x < 30) stage.group_ready[threadIdx.x] = 0;
#endif
      execute_layer0_router_top6(task, stage, shared);
    }
    return;
  }
}

__device__ __forceinline__ const uint8_t* routed_expert_weight(
    Layer0ExpertsStage stage,
    int expert,
    int slot) {
  const int table_slot = generated::kRoutedExpertWeightBases[stage.layer]
      + expert * generated::kExpertWeightSlots + slot;
  return stage.weight_arena + stage.weight_offsets[table_slot];
}

__device__ __forceinline__ const uint8_t* shared_expert_weight(
    Layer0ExpertsStage stage,
    int slot) {
  return stage.weight_arena
      + stage.weight_offsets[generated::kSharedExpertWeightBases[stage.layer] + slot];
}

__device__ dspark_w13::Args routed_w13_args(
    const ClaimedTask& task,
    Layer0ExpertsStage stage) {
  dspark_w13::Args args;
  args.input = stage.input;
  args.input_scales = stage.input_scales;
  args.indices = stage.indices;
  args.routed_w13 = stage.routed_w13;
#ifdef DSPARK_ROUTED_FUSED_SWIGLU
  args.route_weights = stage.route_weights;
  args.swiglu = stage.routed_swiglu;
  args.swiglu_quantized = stage.routed_swiglu_quantized;
  args.swiglu_scales = stage.routed_swiglu_scales;
#ifdef DSPARK_ROUTED_GROUP_READY
  args.group_ready = stage.group_ready;
#endif
#endif
  args.weight_arena = stage.weight_arena;
  args.weight_offsets = stage.weight_offsets;
  args.weight_base = generated::kRoutedExpertWeightBases[stage.layer];
  args.expert_weight_slots = generated::kExpertWeightSlots;
  args.w1_slot = generated::kExpertW1Slot;
  args.w3_slot = generated::kExpertW3Slot;
  args.w1_scale_slot = generated::kExpertW1ScaleSlot;
  args.w3_scale_slot = generated::kExpertW3ScaleSlot;
#if DSPARK_ROUTED_COMPACT_TILES
  args.begin = task.ticket;
  args.end = task.ticket + 1;
#else
  args.begin = task.begin;
  args.end = task.end;
#endif
#ifdef DSPARK_V4_FINE_GRAINED_OVERLAP
  args.item_gap_begin = task.end;
  args.item_gap = 0;
#endif
  return args;
}

__device__ void execute_layer0_routed_w13_args(
    const dspark_w13::Args& args,
    Layer0ExpertsStage stage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#ifdef DSPARK_V4_RELAXED_DAG
  // Stage-3 iter-6: expert-grouped N batching (bitwise vs scalar oracle);
  // streams each selected expert's FP4 weights once per <=8 route_rows.
  // Leg-2 iter-1: DeepGEMM-derived block-scaled UMMA core (see NOTICE).
  // B2: tensor-map weight staging replaces the 1,024 scattered 8-byte
  // cp.async weight copies per stage (bitwise; MMA bytes/order unchanged).
#ifdef DSPARK_V4_TMA_WEIGHTS
  if (stage.w13_weight_tma != nullptr) {
#ifdef DSPARK_ROUTED_W13_STAGE3
    dspark_w13_stage3::execute_w13_dg_tma(args, stage.w13_weight_tma);
#else
    dspark_w13::execute_w13_dg_tma(args, stage.w13_weight_tma);
#endif
  } else {
    dspark_w13::execute_w13_dg(args);
  }
#else
  dspark_w13::execute_w13_dg(args);
#endif
#else
  dspark_w13::execute_tcgen(args);
#endif
#else
  dspark_w13::execute_scalar(args);
#endif
}

__device__ void execute_layer0_routed_swiglu_half(
    uint32_t half_band,
    Layer0ExpertsStage stage,
    float* shared) {
  constexpr int kHalfTile = 256;
  constexpr int kElements =
      dspark_epi::kDraftRows * dspark_epi::kActivatedExperts
      * dspark_epi::kIntermediate;
  const int base = static_cast<int>(half_band) * kHalfTile;
#pragma unroll
  for (int lane_item = threadIdx.x; lane_item < kHalfTile;
       lane_item += kBlockThreads) {
    const int index = base + lane_item;
    if (index >= kElements) {
      continue;
    }
    const int route_row = index / dspark_epi::kIntermediate;
    const int feature = index % dspark_epi::kIntermediate;
    const int w13_base = route_row * 2 * dspark_epi::kIntermediate + feature;
    float gate = __bfloat162float(stage.routed_w13[w13_base]);
    float up = __bfloat162float(
        stage.routed_w13[w13_base + dspark_epi::kIntermediate]);
    gate = fminf(gate, dspark_epi::kSwiGluLimit);
    up = fminf(
        fmaxf(up, -dspark_epi::kSwiGluLimit),
        dspark_epi::kSwiGluLimit);
    const float value = gate / (1.0f + expf(-gate)) * up
        * stage.route_weights[route_row];
    stage.routed_swiglu[index] = __float2bfloat16_rn(value);
  }
  __syncthreads();
  dspark_epi::quantize_bf16_blocks(
      stage.routed_swiglu + base,
      stage.routed_swiglu_quantized + base,
      stage.routed_swiglu_scales + base / dspark_epi::kQuantBlock,
      kHalfTile / dspark_epi::kQuantBlock,
      shared);
}

__device__ __forceinline__ bool routed_group_leader(
    const int32_t* indices,
    int route_row) {
  const int expert = indices[route_row];
  int occurrences_before = 0;
  for (int row = 0; row < route_row; ++row) {
    occurrences_before += indices[row] == expert ? 1 : 0;
  }
  if (occurrences_before % 8 != 0) {
    return false;
  }
#if defined(DSPARK_V4_TP2_ABLATE)
  if (!dspark_tp2_ablate::owns_expert(expert)) {
    return false;
  }
#endif
  return true;
}

__device__ void execute_layer0_routed_w13(
    const ClaimedTask& task,
    Layer0ExpertsStage stage,
    float* shared) {
#ifdef DSPARK_V4_FINE_GRAINED_OVERLAP
  // One physical claim owns matching 256-feature gate/up bands. Keep the
  // historical four W13 work units for completion accounting, but execute
  // them as two aligned DG tile-pairs and immediately consume the result.
  const uint32_t route_row = task.ticket / 8;
  const uint32_t half = task.ticket % 8;
  if (!routed_group_leader(stage.indices, route_row)) {
    return;
  }
  ClaimedTask band_task = task;
  band_task.begin = route_row * dspark_w13::kCombinedTiles + half * 2;
  band_task.end = band_task.begin + 4;
  dspark_w13::Args args = routed_w13_args(band_task, stage);
  args.item_gap_begin = band_task.begin + 2;
  args.item_gap =
      dspark_w13::kIntermediate / dspark_w13::kOutputTile - 2;
  execute_layer0_routed_w13_args(args, stage);
  const int expert = stage.indices[route_row];
  int member_count = 0;
  constexpr int kRouteRows =
      dspark_w13::kDraftRows * dspark_w13::kActivatedExperts;
  for (int member_row = route_row;
       member_row < kRouteRows && member_count < 8;
       ++member_row) {
    if (stage.indices[member_row] != expert) {
      continue;
    }
    execute_layer0_routed_swiglu_half(member_row * 8 + half, stage, shared);
    ++member_count;
  }
#else
  execute_layer0_routed_w13_args(routed_w13_args(task, stage), stage);
#endif
}

__device__ void execute_layer0_shared_w13(
    const ClaimedTask& task,
    Layer0ExpertsStage stage) {
  dspark_shared::Args args;
  args.input = stage.input;
  args.input_scales = stage.input_scales;
  args.shared_w13 = stage.shared_w13;
  args.weight_arena = stage.weight_arena;
  args.weight_offsets = stage.weight_offsets;
  args.weight_base = generated::kSharedExpertWeightBases[stage.layer];
  args.w1_slot = generated::kExpertW1Slot;
  args.w3_slot = generated::kExpertW3Slot;
  args.w1_scale_slot = generated::kExpertW1ScaleSlot;
  args.w3_scale_slot = generated::kExpertW3ScaleSlot;
  args.begin = task.begin;
  args.end = task.end;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#if DSPARK_V4_BATCH == 1
#ifdef DSPARK_SHARED_BULK_STAGING
  for (uint32_t item = args.begin; item < args.end; ++item) {
    const auto plan = dspark_shared::item_plan(args, item);
    const uint32_t tile = plan.output_base / 128;
#ifdef DSPARK_SHARED_NATIVE_SCALE4
    dspark_fp8_scaled::execute<4, 4096, 2048, 5, 1, true, 4096>(
        {args.input, args.input_scales, plan.weight, plan.weight_scales, nullptr,
         args.shared_w13 + plan.branch * 2048, tile, tile + 1});
#else
    dspark_shared_bulk::execute<4, 4096, 2048, 5, 1, true, 4096>(
        {args.input, args.input_scales, plan.weight, plan.weight_scales, nullptr,
         args.shared_w13 + plan.branch * 2048, tile, tile + 1});
#endif
  }
#elif defined(DSPARK_SHARED_K128_RING)
  dspark_shared::execute_tcgen<4>(args);
#else
  dspark_shared::execute_tcgen(args);
#endif
#else
  // Batched serving (R4 band 7): the shared expert's 16 MiB W1/W3 pair is
  // read ONCE for the whole batch; the 5*batch flat rows ride the UMMA N mode
  // and the item count stays at the batch-invariant kCombinedTiles.
  dspark_shared::execute_tcgen_wide<
      dspark_batch::kWideRows, dspark_batch::kRows>(args);
#endif
#else
  dspark_shared::execute_scalar(args);
#endif
}

__device__ void execute_layer0_routed_swiglu(
    const ClaimedTask& task,
    Layer0ExpertsStage stage,
    float* shared) {
  // Body extracted verbatim into dspark_epi_phase.cuh (relaxed-drafter
  // stage 3, front/epilogue bands). Pure code motion; serves contract AND
  // relaxed builds.
  dspark_epi::SwigluArgs args;
  args.w13 = stage.routed_w13;
  args.route_weights = stage.route_weights;
  args.swiglu = stage.routed_swiglu;
  args.swiglu_quantized = stage.routed_swiglu_quantized;
  args.swiglu_scales = stage.routed_swiglu_scales;
  args.begin = task.begin;
  args.end = task.end;
#ifdef DSPARK_SWIGLU_WARP_QUANT
  dspark_swiglu::execute<true>(args);
#else
  dspark_epi::execute_routed_swiglu_reference(args, shared);
#endif
}

__device__ void execute_layer0_shared_swiglu(
    const ClaimedTask& task,
    Layer0ExpertsStage stage,
    float* shared) {
  // Body extracted verbatim into dspark_epi_phase.cuh (relaxed-drafter
  // stage 3, front/epilogue bands). Pure code motion; serves contract AND
  // relaxed builds.
  dspark_epi::SwigluArgs args;
  args.w13 = stage.shared_w13;
  args.route_weights = nullptr;
  args.swiglu = stage.shared_swiglu;
  args.swiglu_quantized = stage.shared_swiglu_quantized;
  args.swiglu_scales = stage.shared_swiglu_scales;
  args.begin = task.begin;
  args.end = task.end;
#ifdef DSPARK_SWIGLU_WARP_QUANT
  dspark_swiglu::execute<false>(args);
#else
  dspark_epi::execute_shared_swiglu_reference(args, shared);
#endif
}

__device__ void execute_layer0_routed_w2(
    const ClaimedTask& task,
    Layer0ExpertsStage stage) {
  dspark_w13::W2Args args;
#ifdef DSPARK_ROUTED_GROUP_READY
  args.group_ready = stage.group_ready;
#endif
  args.swiglu_quantized = stage.routed_swiglu_quantized;
  args.swiglu_scales = stage.routed_swiglu_scales;
  args.indices = stage.indices;
  args.output_partials = stage.routed_output_partials;
  args.weight_arena = stage.weight_arena;
  args.weight_offsets = stage.weight_offsets;
  args.weight_base = generated::kRoutedExpertWeightBases[stage.layer];
  args.expert_weight_slots = generated::kExpertWeightSlots;
  args.w2_slot = generated::kExpertW2Slot;
  args.w2_scale_slot = generated::kExpertW2ScaleSlot;
#if DSPARK_ROUTED_COMPACT_TILES
  args.begin = task.ticket;
  args.end = task.ticket + 1;
#else
  args.begin = task.begin;
  args.end = task.end;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#ifdef DSPARK_V4_RELAXED_DAG
  // B2: same tensor-map staging swap as the routed W13 body (bitwise).
#ifdef DSPARK_V4_TMA_WEIGHTS
  if (stage.w2_weight_tma != nullptr) {
    dspark_w13::execute_w2_dg_tma(args, stage.w2_weight_tma);
  } else {
    dspark_w13::execute_w2_dg(args);
  }
#else
  dspark_w13::execute_w2_dg(args);
#endif
#else
  dspark_w13::execute_w2_tcgen(args);
#endif
#else
  dspark_w13::execute_w2_scalar(args);
#endif
}

__device__ void execute_layer0_shared_w2(
    const ClaimedTask& task,
    Layer0ExpertsStage stage) {
  // Body extracted verbatim into dspark_epi_phase.cuh (relaxed-drafter
  // stage 3, front/epilogue bands) so the bench lane compiles exactly the
  // production implementation. Pure code motion (the shared_expert_weight
  // slot lookups become pre-offset pointers); serves contract AND relaxed
  // builds.
  dspark_epi::SharedW2Args args;
  args.swiglu_quantized = stage.shared_swiglu_quantized;
  args.swiglu_scales = stage.shared_swiglu_scales;
  args.weight = reinterpret_cast<const cutlass::float_e4m3_t*>(
      shared_expert_weight(stage, generated::kExpertW2Slot));
  args.weight_scales =
      shared_expert_weight(stage, generated::kExpertW2ScaleSlot);
  args.shared_output = stage.shared_output;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
#ifdef DSPARK_DENSE_BLOCK_SCALED
  dspark_fp8_scaled::execute<DSPARK_DENSE_BLOCK_SCALED, 2048, 4096, 5, 1, true>(
      {args.swiglu_quantized, args.swiglu_scales,
       args.weight, args.weight_scales,
       nullptr, args.shared_output, args.begin, args.end});
#else
  dspark_epi::execute_shared_w2_tcgen(args);
#endif
#else
  dspark_epi::execute_shared_w2_reference(args);
#endif
}

__device__ void execute_layer0_expert_combine(
    const ClaimedTask& task,
    Layer0ExpertsStage stage) {
  // Body extracted verbatim into dspark_epi_phase.cuh (relaxed-drafter
  // stage 3, front/epilogue bands). Pure code motion; serves contract AND
  // relaxed builds.
  dspark_epi::CombineArgs args;
  args.indices = stage.indices;
  args.routed_output_partials = stage.routed_output_partials;
  args.shared_output = stage.shared_output;
  args.residual_streams = stage.residual_streams;
  args.comb = stage.comb;
  args.post = stage.post;
  args.routed_output = stage.routed_output;
  args.streams = stage.streams;
  args.begin = task.begin;
  args.end = task.end;
  dspark_epi::execute_combine_reference(args);
}

__device__ DSPARK_EXPERT_ENTRY void execute_layer0_expert_phase(
    const ClaimedTask& task,
    Layer0ExpertsStage stage,
    float* shared) {
  if (!stage.enabled) {
    return;
  }
  for (int layer = 0; layer < 3; ++layer) {
    if (task.phase != generated::kRoutedW13Phases[layer]
        && task.phase != generated::kRoutedSwiGluPhases[layer]
        && task.phase != generated::kRoutedW2Phases[layer]
        && task.phase != generated::kSharedW13Phases[layer]
        && task.phase != generated::kSharedSwiGluPhases[layer]
        && task.phase != generated::kSharedW2Phases[layer]
        && task.phase != generated::kExpertCombinePhases[layer]) {
      continue;
    }
    if (!layer_enabled(stage.layer_mask, layer)) {
      return;
    }
    stage.layer = layer;
    if (task.phase == generated::kRoutedW13Phases[layer]) {
      execute_layer0_routed_w13(task, stage, shared);
    } else if (task.phase == generated::kRoutedSwiGluPhases[layer]) {
      execute_layer0_routed_swiglu(task, stage, shared);
    } else if (task.phase == generated::kRoutedW2Phases[layer]) {
      execute_layer0_routed_w2(task, stage);
    } else if (task.phase == generated::kSharedW13Phases[layer]) {
      execute_layer0_shared_w13(task, stage);
    } else if (task.phase == generated::kSharedSwiGluPhases[layer]) {
      execute_layer0_shared_swiglu(task, stage, shared);
    } else if (task.phase == generated::kSharedW2Phases[layer]) {
      execute_layer0_shared_w2(task, stage);
    } else {
      execute_layer0_expert_combine(task, stage);
    }
    return;
  }
}

__device__ void execute_head_reduce(
    const ClaimedTask& task,
    HeadStage stage,
    float* shared
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    ,
    const dspark_v4_tp2::DeviceContext& tp2_context
#endif
) {
  constexpr int kFlattened = kHcStreams * kMainHidden;
  constexpr float kHcEpsilon = 1.0e-6f;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    if (item >= kDraftRows) {
      continue;
    }
    const int row = static_cast<int>(item);
    const int row_base = row * kFlattened;
#ifndef DSPARK_HEAD_PREFETCH
    // Four 16-byte copies per thread cover the immutable 16KiB weight vector.
    // This region is disjoint from reduction/gates/hidden scratch and old prefetch slots.
    constexpr int kNormWeightByteOffset = 114688;
    static_assert(kNormWeightByteOffset % 16 == 0);
    static_assert(kNormWeightByteOffset >= (256 + 4) * 4 + 4096 * 2);
    static_assert(kNormWeightByteOffset + 4096 * 4 <= 153600);
    float* norm_weight_shared = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(shared) + kNormWeightByteOffset);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int offset = (threadIdx.x + j * kBlockThreads) * 4;
      const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(norm_weight_shared + offset));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(stage.norm_weight + offset) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
    // R1 leg 6, intra-node retile: the RMS pass and the four per-stream dots
    // all walk the SAME 16,384 ascending elements of the row, and the four
    // dots never feed each other. They ride ONE traversal instead of five
    // back-to-back ones.
    //
    // Every accumulator still sees exactly the ascending element sequence its
    // owning thread saw before, so local_square and each local_dot[stream]
    // are bit-for-bit the values the serialized shape produced; only the five
    // chains' instructions interleave. The band was latency-bound at one
    // dependent load per element with 147 of 152 SMs idle -- the fused pass
    // keeps five independent loads in flight per element instead.
#ifdef DSPARK_HEAD_PREFETCH
    float local_square = 0.0f;
    const auto* row_streams = dspark_head_prefetch::project(stage, row, shared);
#else
    const auto* row_streams = stage.streams + row_base;
    float local_square = 0.0f;
    float local_dot[kHcStreams];
#pragma unroll
    for (int stream = 0; stream < kHcStreams; ++stream) {
      local_dot[stream] = 0.0f;
    }
    #pragma unroll 8
    for (int index = threadIdx.x; index < kFlattened; index += kBlockThreads) {
      const float value = __bfloat162float(stage.streams[row_base + index]);
      local_square = fmaf(value, value, local_square);
#pragma unroll
      for (int stream = 0; stream < kHcStreams; ++stream) {
        local_dot[stream] =
            fmaf(value, stage.hc_fn[stream * kFlattened + index], local_dot[stream]);
      }
    }
    shared[threadIdx.x] = local_square;
    __syncthreads();
    dspark_reduction::sum_256(shared);
    const float inverse_rms = rsqrtf(shared[0] / kFlattened + kMainNormEpsilon);
    // The retired shape let the (long) dot loop stand in for this barrier
    // between reading shared[0] and thread 0 overwriting it; with the dots
    // hoisted out there is nothing left to hide the reuse.
    __syncthreads();

#pragma unroll
    for (int stream = 0; stream < kHcStreams; ++stream) {
      shared[threadIdx.x] = local_dot[stream];
      __syncthreads();
      dspark_reduction::sum_256(shared);
      if (threadIdx.x == 0) {
        const float mix = shared[0] * inverse_rms;
        const float logit = mix * stage.hc_scale[0] + stage.hc_base[stream];
        shared[kBlockThreads + stream] =
            1.0f / (1.0f + expf(-logit)) + kHcEpsilon;
      }
      __syncthreads();
    }

#endif

#ifdef DSPARK_V4_RELAXED_DAG
    __nv_bfloat16* head_hidden_scratch = reinterpret_cast<__nv_bfloat16*>(
        shared + kBlockThreads + kHcStreams);
    local_square = 0.0f;
#endif
    #pragma unroll 4
    for (int column = threadIdx.x; column < kMainHidden; column += kBlockThreads) {
      float value = 0.0f;
#pragma unroll
      for (int stream = 0; stream < kHcStreams; ++stream) {
        value = fmaf(
            shared[kBlockThreads + stream],
            __bfloat162float(
                row_streams[stream * kMainHidden + column]),
            value);
      }
      const __nv_bfloat16 stored = __float2bfloat16_rn(value);
      stage.hidden[row * kMainHidden + column] = stored;
#ifdef DSPARK_V4_RELAXED_DAG
      head_hidden_scratch[column] = stored;
      const float rounded = __bfloat162float(stored);
      local_square = fmaf(rounded, rounded, local_square);
#endif
    }
#ifndef DSPARK_V4_RELAXED_DAG
    __syncthreads();

    local_square = 0.0f;
    #pragma unroll 8
    for (int column = threadIdx.x; column < kMainHidden; column += kBlockThreads) {
      const float value =
          __bfloat162float(stage.hidden[row * kMainHidden + column]);
      local_square = fmaf(value, value, local_square);
    }
#endif
    shared[threadIdx.x] = local_square;
    __syncthreads();
    dspark_reduction::sum_256(shared);
    const float norm_inverse_rms =
        rsqrtf(shared[0] / kMainHidden + kMainNormEpsilon);
#ifndef DSPARK_HEAD_PREFETCH
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
#endif
    #pragma unroll 8
    for (int column = threadIdx.x; column < kMainHidden; column += kBlockThreads) {
#ifdef DSPARK_V4_RELAXED_DAG
      const float value = __bfloat162float(head_hidden_scratch[column]);
#else
      const float value =
          __bfloat162float(stage.hidden[row * kMainHidden + column]);
#endif
      stage.normalized[row * kMainHidden + column] = __float2bfloat16_rn(
#ifndef DSPARK_HEAD_PREFETCH
          value * norm_inverse_rms * norm_weight_shared[column]);
#else
          value * norm_inverse_rms * stage.norm_weight[column]);
#endif
    }
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    dspark_v4_tp2::rank0_publish_normalized_row(
        tp2_context, stage.normalized, row);
#endif
  }
}

__device__ void execute_lm_rows(const ClaimedTask& task, HeadStage stage) {
  dspark_lm::Args args;
  args.lm_head = stage.lm_head;
  args.normalized = stage.normalized;
  args.base_logits = stage.base_logits;
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_TP2_ABLATE)
  // LM head vocab split, rebalanced 505/505 (shipped static tail is 404/606).
  dspark_tp2_ablate::clip_half(
      1010, dspark_tp2_ablate::kBandVocab, args.begin, args.end);
#endif
  if (stage.lm_head_bf16 != nullptr) {
    args.lm_head_bf16 = stage.lm_head_bf16;
#if defined(DSPARK_V4_RELAXED_DAG) && defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 1000
    // Stage-3 iteration 3: descriptor-fed TCGen05 UMMA (0.326 ms isolated vs
    // 0.593 wmma); numerics identical to the wmma body to the last digit.
#if DSPARK_V4_BATCH == 1
#if defined(DSPARK_LM_STREAMED_K)
    dspark_lm_streamed::execute<DSPARK_LM_STREAMED_K>(args);
#elif defined(DSPARK_LM_K128_RING)
    // FALSIFIED APPARATUS: this source-proven K=128/two-stage hybrid ring
    // measured 1.555 ms versus 1.498 ms controls when composed with the best
    // routed body. It remains opt-in only. The hybrid walker indexes M64
    // units, while this phase's ABI is M128, so doubling the half-open range
    // produces only aligned interior items: identical rows/K order, half the
    // ring turns, but more per-slot issue/accumulator machinery.
    args.begin *= 2;
    args.end *= 2;
    dspark_lm::execute_umma_bf16_hybrid(args);
#else
    dspark_lm::execute_umma_bf16(args);
#endif
#else
    // Batched serving: the SAME 1.06 GiB weight stream, 5*batch activation
    // rows on the UMMA N mode. R3 measured 96.4% of this band's step to be
    // N-invariant and rows 0-4 BITWISE identical at every width up to 48.
    dspark_lm::execute_umma_bf16_batch<kUmmaBatchRows, kDraftRows>(args);
#endif
#else
    dspark_lm::execute_wmma_bf16(args);
#endif
    return;
  }
  dspark_lm::execute_scalar(args);
}

__device__ DSPARK_HEAD_ENTRY void execute_head_phase(
    const ClaimedTask& task,
    HeadStage stage,
    float* shared
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    ,
    const dspark_v4_tp2::DeviceContext& tp2_context
#endif
) {
  if (!stage.enabled) {
    return;
  }
  if (task.phase == generated::kHeadReducePhase) {
    execute_head_reduce(
        task,
        stage,
        shared
#ifdef DSPARK_V4_STATIC_TP2_TAIL
        ,
        tp2_context
#endif
    );
    return;
  }
  if (task.phase == generated::kLmRowPhases[0]) {
    execute_lm_rows(task, stage);
    return;
  }
  for (int row = 1; row < kDraftBlock; ++row) {
    if (task.phase == generated::kLmRowPhases[row]) {
      return;
    }
  }
}

// Packed (value, lowest-index-tiebreak) encoding for the fused tail argmax
// (#102). The float bits map monotonically onto uint32, the complemented
// index occupies the low word, so a single uint64 atomicMax reproduces the
// two-stage (max value, then lowest index) reduction bit-exactly under ANY
// arrival order.
__device__ inline uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ inline float float_from_ordered_bits(uint32_t ordered) {
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered & 0x7FFFFFFFu) : ~ordered;
  return __uint_as_float(bits);
}

__device__ inline unsigned long long pack_argmax(float value, int index) {
  return (static_cast<unsigned long long>(ordered_float_bits(value)) << 32)
      | static_cast<unsigned long long>(0xFFFFFFFFu - static_cast<uint32_t>(index));
}

// One 256-byte-separated slot per (step, batch element). softmax_partial_max
// is (batch, block, 512) floats = batch * 5 * 256 uint64 words, so the stride
// batch(256 / kBatch) keeps every slot inside the region and reproduces the
// frozen step * 256 layout exactly at batch 1.
__device__ inline unsigned long long* tail_argmax_slot(
    TailStage stage,
    int step,
    int element = 0) {
  constexpr int kSlotStride = 256 / kBatch;
  return reinterpret_cast<unsigned long long*>(stage.partial_max)
      + (step * kBatch + element) * kSlotStride;
}

__device__ void execute_tail_markov_gather(
    const ClaimedTask& task,
    TailStage stage,
    int step
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    ,
    const dspark_v4_tp2::DeviceContext& tp2_context
#endif
) {
  // One work item per batch element; the batch-1 program has exactly one.
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int element = static_cast<int>(item);
    const int previous_token = step == 0
        ? stage.anchor[element]
        : stage.output_ids[element * (kDraftBlock + 1) + step];
    if (threadIdx.x == 0) {
      if (step == 0) {
        stage.output_ids[element * (kDraftBlock + 1)] = previous_token;
      }
      // Reset this step's fused-argmax slot; the dependency chain
      // (markov_gather -> markov_w2 -> correct_logits_partial_max) orders
      // this write before any tile's atomicMax.
      *tail_argmax_slot(stage, step, element) = 0ULL;
    }
    stage.markov_embeddings[(element * kDraftBlock + step) * 256 + threadIdx.x] =
        stage.markov_w1[static_cast<int64_t>(previous_token) * 256 + threadIdx.x];
  }
#ifdef DSPARK_V4_STATIC_TP2_TAIL
  dspark_v4_tp2::rank0_publish_markov_embedding(
      tp2_context, stage.markov_embeddings, step);
#endif
}

#ifdef DSPARK_CONFIDENCE_HEAD_PREFIX
// The first 4096 bytes of greedy-unused softmax_partial_sum are explicitly
// owned by this producer for steps 1..4, one FP32 prefix per thread. Stochastic
// softmax phases are absent from the guarded generated program. Step 0 keeps
// its original computation because its gather overlaps the LM phase.
__device__ inline void execute_confidence_head_prefix(TailStage stage) {
  static_assert(kMainHidden == 4096 && kBlockThreads == 256 && kDraftBlock == 5);
  static_assert(generated::kSampleScanOffset - generated::kSoftmaxPartialSumOffset
                >= 4 * 256 * sizeof(float));
  #pragma unroll
  for (int step = 1; step < kDraftBlock; ++step) {
    float partial = 0.0f;
    #pragma unroll 8
    for (int column = threadIdx.x; column < kMainHidden; column += kBlockThreads) {
      partial = fmaf(
          __bfloat162float(stage.head_hidden[step * kMainHidden + column]),
          stage.confidence_weight[column],
          partial);
    }
    stage.partial_sum[(step - 1) * kBlockThreads + threadIdx.x] = partial;
  }
  // Each writer orders its own four stores before the existing outer body
  // join and thread-0 LM completion fence/counter. No added CTA join is needed.
  __threadfence();
}
#endif

__device__ void execute_tail_confidence(
    const ClaimedTask& task,
    TailStage stage,
    int step,
    float* shared) {
  if (task.begin != 0) {
    return;
  }
  float partial;
#ifdef DSPARK_CONFIDENCE_HEAD_PREFIX
  if (step > 0) {
    partial = stage.partial_sum[(step - 1) * kBlockThreads + threadIdx.x];
  } else
#endif
  {
    partial = 0.0f;
    #pragma unroll 8
    for (int column = threadIdx.x; column < kMainHidden; column += kBlockThreads) {
      partial = fmaf(
          __bfloat162float(stage.head_hidden[step * kMainHidden + column]),
          stage.confidence_weight[column],
          partial);
    }
  }
  #pragma unroll
  for (int column = threadIdx.x; column < 256; column += kBlockThreads) {
    partial = fmaf(
        __bfloat162float(stage.markov_embeddings[step * 256 + column]),
        stage.confidence_weight[kMainHidden + column],
        partial);
  }
  shared[threadIdx.x] = partial;
  __syncthreads();
  dspark_reduction::sum_256(shared);
  if (threadIdx.x == 0) {
    const float raw = shared[0];
    stage.confidence_logits[step] = raw;
    const float temperature =
        stage.calibration_enabled ? stage.sts_temperatures[step] : 1.0f;
    stage.calibrated_confidences[step] =
        1.0f / (1.0f + expf(-raw / temperature));
  }
}

__device__ void execute_tail_markov_w2(
    const ClaimedTask& task,
    TailStage stage,
    int step) {
  constexpr int kVocab = 129280;
  constexpr int kTile = 128;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int output = static_cast<int>(item) * kTile + threadIdx.x;
    if (threadIdx.x >= kTile || output >= kVocab) {
      continue;
    }
    float result = 0.0f;
    const int64_t weight_base = static_cast<int64_t>(output) * 256;
    const float4* weight_vectors =
        reinterpret_cast<const float4*>(stage.markov_w2 + weight_base);
#pragma unroll 8
    for (int column = 0; column < 256; column += 4) {
      const float4 weight_vector = weight_vectors[column / 4];
      const float* weights = reinterpret_cast<const float*>(&weight_vector);
#pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
        result = fmaf(
            __bfloat162float(
                stage.markov_embeddings[step * 256 + column + offset]),
            weights[offset],
            result);
      }
    }
    stage.markov_logits[output] = result;
  }
}

// Relaxed-numerics markov body: identical ascending-column FP32 chain, BF16
// weight stream (16-byte uint4 packets of 8 weights). The BF16 values are
// checkpoint-exact, so drift vs the FP32 body is zero — only traffic halves.
__device__ void execute_tail_markov_w2_bf16(
    const ClaimedTask& task,
    TailStage stage,
    int step) {
  constexpr int kVocab = 129280;
  constexpr int kTile = 128;
  for (uint32_t item = task.begin; item < task.end; ++item) {
    const int output = static_cast<int>(item) * kTile + threadIdx.x;
    if (threadIdx.x >= kTile || output >= kVocab) {
      continue;
    }
    float result = 0.0f;
    const int64_t weight_base = static_cast<int64_t>(output) * 256;
    const uint4* weight_vectors = reinterpret_cast<const uint4*>(
        stage.markov_w2_bf16 + weight_base);
#pragma unroll 4
    for (int column = 0; column < 256; column += 8) {
      const uint4 packet = weight_vectors[column / 8];
      const __nv_bfloat16* weights =
          reinterpret_cast<const __nv_bfloat16*>(&packet);
#pragma unroll
      for (int offset = 0; offset < 8; ++offset) {
        result = fmaf(
            __bfloat162float(
                stage.markov_embeddings[step * 256 + column + offset]),
            __bfloat162float(weights[offset]),
            result);
      }
    }
    stage.markov_logits[output] = result;
  }
}

#ifdef DSPARK_V4_RELAXED_DAG
// Stage-2 fused markov body: markov GEMV (BF16 stream when the relaxed slot
// is present, FP32 otherwise) + logit correction + the #102 packed argmax,
// all in one 1010-tile phase.
//
// The body itself now lives in dspark_markov_phase.cuh (pure code motion; the
// extracted execute_fused_reference is byte-for-byte the previous body with
// stage./task. renamed to args.), so the phase microbenchmark compiles the
// exact implementation the production kernel runs. This wrapper resolves the
// per-step pointer offsets and selects the body.
__device__ void execute_tail_markov_w2_fused(
    const ClaimedTask& task,
    TailStage stage,
    int step,
    float* shared) {
  constexpr int kVocab = 129280;
  dspark_markov::Args args;
  args.markov_w2 = stage.markov_w2;
  args.markov_w2_bf16 = stage.markov_w2_bf16;
  args.embedding = stage.markov_embeddings + step * 256;
  args.base_logits = stage.base_logits + static_cast<int64_t>(step) * kVocab;
  args.corrected_logits =
      stage.corrected_logits + static_cast<int64_t>(step) * kVocab;
  args.markov_logits = stage.markov_logits;
  args.argmax_slot = tail_argmax_slot(stage, step);
  args.begin = task.begin;
  args.end = task.end;
#if defined(DSPARK_V4_TP2_ABLATE)
  // markov_w2 vocab split, same 505/505 tiling as the LM head.
  dspark_tp2_ablate::clip_half(
      1010, dspark_tp2_ablate::kBandVocab, args.begin, args.end);
#endif
  // Batched serving: element e's row of this step sits one block-stride away
  // in every (batch, block, ...) region. The 66 MiB W2 stream is unchanged.
  args.embedding_stride = kDraftBlock * 256;
  args.logits_stride = static_cast<int64_t>(kDraftBlock) * kVocab;
  args.markov_logits_stride = kVocab;
  args.argmax_slot_stride = 256 / kBatch;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  if (stage.markov_w2_bf16 != nullptr) {
    // TCGen05 UMMA staging pipeline (the LM band's port, K = 256), wide-slot
    // instantiation (32 KB slots, 3-deep ring — the phase-local lane's best
    // variant). The arch guard matters: the host compilation pass cannot see
    // the gated body.
#if defined(DSPARK_MARKOV_BULK_K)
    static_assert(kBatch == 1, "bulk Markov staging specializes batch one");
    if constexpr (DSPARK_MARKOV_BULK_K == 128) {
      dspark_markov::execute_umma_impl<8, 3, 1, true>(args);
    } else {
      dspark_markov::execute_umma_impl<16, 2, 1, true>(args);
    }
#else
    dspark_markov::execute_umma_bf16_wide<kBatch>(args);
#endif
    return;
  }
#endif
  dspark_markov::execute_fused_reference(args, shared);
}
#endif  // DSPARK_V4_RELAXED_DAG

__device__ void execute_tail_correct_logits(
    const ClaimedTask& task,
    TailStage stage,
    int step,
    float* shared) {
  constexpr int kVocab = 129280;
  constexpr int kTile = 512;
  int* shared_index = reinterpret_cast<int*>(shared + kBlockThreads);
  for (uint32_t item = task.begin; item < task.end; ++item) {
    float local_max = -FLT_MAX;
    int local_index = kVocab;
#pragma unroll
    for (int lane_item = 0; lane_item < 2; ++lane_item) {
      const int vocab = static_cast<int>(item) * kTile
          + threadIdx.x + lane_item * kBlockThreads;
      if (vocab >= kVocab) {
        continue;
      }
      const int64_t offset = static_cast<int64_t>(step) * kVocab + vocab;
      const float corrected = stage.base_logits[offset] + stage.markov_logits[vocab];
      stage.corrected_logits[offset] = corrected;
      if (corrected > local_max || (corrected == local_max && vocab < local_index)) {
        local_max = corrected;
        local_index = vocab;
      }
    }
    shared[threadIdx.x] = local_max;
    shared_index[threadIdx.x] = local_index;
    __syncthreads();
    #pragma unroll
    for (int delta = kBlockThreads / 2; delta > 0; delta /= 2) {
      if (threadIdx.x < delta) {
        const float other = shared[threadIdx.x + delta];
        const int other_index = shared_index[threadIdx.x + delta];
        if (other > shared[threadIdx.x]
            || (other == shared[threadIdx.x]
                && other_index < shared_index[threadIdx.x])) {
          shared[threadIdx.x] = other;
          shared_index[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicMax(
          tail_argmax_slot(stage, step),
          pack_argmax(shared[0], shared_index[0]));
    }
    __syncthreads();
  }
}

__device__ void execute_tail_exp_sum(
    const ClaimedTask& task,
    TailStage stage,
    int step,
    float* shared) {
  if (stage.sampling_temperature < 1.0e-5f) {
    return;
  }
  constexpr int kVocab = 129280;
  constexpr int kTile = 512;
  const float maximum = float_from_ordered_bits(
      static_cast<uint32_t>(*tail_argmax_slot(stage, step) >> 32));
  for (uint32_t item = task.begin; item < task.end; ++item) {
    float local_sum = 0.0f;
#pragma unroll
    for (int lane_item = 0; lane_item < 2; ++lane_item) {
      const int vocab = static_cast<int>(item) * kTile
          + threadIdx.x + lane_item * kBlockThreads;
      if (vocab < kVocab) {
        const int64_t offset = static_cast<int64_t>(step) * kVocab + vocab;
        const float value = expf(
            (stage.corrected_logits[offset] - maximum)
            / stage.sampling_temperature);
        stage.probabilities[offset] = value;
        local_sum += value;
      }
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    dspark_reduction::sum_256(shared);
    if (threadIdx.x == 0) {
      stage.partial_sum[step * 512 + item] = shared[0];
    }
    __syncthreads();
  }
}

__device__ void execute_tail_sum_reduce(
    const ClaimedTask& task,
    TailStage stage,
    int step,
    float* shared) {
  if (stage.sampling_temperature < 1.0e-5f || task.begin != 0) {
    return;
  }
  constexpr int kPartials = (129280 + 511) / 512;
  float local_sum = 0.0f;
  #pragma unroll
  for (int item = threadIdx.x; item < kPartials; item += kBlockThreads) {
    local_sum += stage.partial_sum[step * 512 + item];
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  dspark_reduction::sum_256(shared);
  if (threadIdx.x == 0) {
    const float total = shared[0];
    const float threshold = stage.uniforms[step] * total;
    float prefix = 0.0f;
    int selected_tile = kPartials - 1;
    for (int item = 0; item < kPartials; ++item) {
      const float next = prefix + stage.partial_sum[step * 512 + item];
      if (next >= threshold) {
        selected_tile = item;
        break;
      }
      prefix = next;
    }
    stage.partial_sum[step * 512] = total;
    stage.sample_scan[step * 512] = static_cast<float>(selected_tile);
    stage.sample_scan[step * 512 + 1] = prefix;
  }
}

__device__ void execute_tail_normalize_sample(
    const ClaimedTask& task,
    TailStage stage,
    int step) {
  constexpr int kVocab = 129280;
  constexpr int kTile = 512;
  if (stage.sampling_temperature < 1.0e-5f) {
    const int selected = static_cast<int>(
        0xFFFFFFFFu
        - static_cast<uint32_t>(*tail_argmax_slot(stage, step) & 0xFFFFFFFFu));
    for (uint32_t item = task.begin; item < task.end; ++item) {
      for (int lane_item = 0; lane_item < 2; ++lane_item) {
        const int vocab = static_cast<int>(item) * kTile
            + threadIdx.x + lane_item * kBlockThreads;
        if (vocab < kVocab) {
          stage.probabilities[static_cast<int64_t>(step) * kVocab + vocab] =
              vocab == selected ? 1.0f : 0.0f;
        }
      }
      if (item == 0 && threadIdx.x == 0) {
        stage.output_ids[step + 1] = selected;
      }
    }
    return;
  }

#ifdef DSPARK_V4_RELAXED_DAG
  // Stage-2 fusion: sum_reduce's total + tile selection recompute redundantly
  // per task (253 float reads in registers) instead of costing a phase
  // boundary. Same ascending accumulation order as the retired phase.
  constexpr int kPartials = (kVocab + kTile - 1) / kTile;
  float total = 0.0f;
  for (int part = 0; part < kPartials; ++part) {
    total += stage.partial_sum[step * 512 + part];
  }
  const float inverse_total = 1.0f / total;
  const float threshold_all = stage.uniforms[step] * total;
  float selected_prefix = 0.0f;
  int selected_tile = kPartials - 1;
  {
    float prefix = 0.0f;
    for (int part = 0; part < kPartials; ++part) {
      const float next = prefix + stage.partial_sum[step * 512 + part];
      if (next >= threshold_all) {
        selected_tile = part;
        break;
      }
      prefix = next;
    }
    selected_prefix = prefix;
  }
#else
  const float total = stage.partial_sum[step * 512];
  const float inverse_total = 1.0f / total;
  const int selected_tile = static_cast<int>(stage.sample_scan[step * 512]);
  const float selected_prefix = stage.sample_scan[step * 512 + 1];
#endif
  for (uint32_t item = task.begin; item < task.end; ++item) {
    if (static_cast<int>(item) == selected_tile && threadIdx.x == 0) {
      const int begin = selected_tile * kTile;
      const int end = min(begin + kTile, kVocab);
      const float threshold = stage.uniforms[step] * total - selected_prefix;
      float cumulative = 0.0f;
      int selected = end - 1;
      for (int vocab = begin; vocab < end; ++vocab) {
        cumulative += stage.probabilities[
            static_cast<int64_t>(step) * kVocab + vocab];
        if (cumulative >= threshold) {
          selected = vocab;
          break;
        }
      }
      stage.output_ids[step + 1] = selected;
    }
    __syncthreads();
    for (int lane_item = 0; lane_item < 2; ++lane_item) {
      const int vocab = static_cast<int>(item) * kTile
          + threadIdx.x + lane_item * kBlockThreads;
      if (vocab < kVocab) {
        stage.probabilities[static_cast<int64_t>(step) * kVocab + vocab] *=
            inverse_total;
      }
    }
  }
}

__device__ void execute_tail_prefix_scheduler(
    const ClaimedTask& task,
    TailStage stage) {
  if (task.begin != 0 || threadIdx.x != 0) {
    return;
  }
#pragma unroll
  for (int step = 0; step < kDraftBlock; ++step) {
    stage.scheduler_read_mask[step] = 0;
  }
  if (!stage.prefix_enabled) {
    stage.scheduled_prefix_lengths[0] = kDraftBlock;
    stage.scheduler_summary[0] = 1.0f;
    stage.scheduler_summary[1] = 1.0f;
    stage.scheduler_summary[2] = 0.0f;
    stage.scheduler_summary[3] = 0.0f;
    return;
  }

  float best_tau = 1.0f;
  int best_batch = 1;
  float best_throughput = stage.steps_per_second[1];
  int best_length = 0;
  float current_tau = 1.0f;
  int current_batch = 1;
  float survival = 1.0f;
  int evaluated = 0;
#pragma unroll
  for (int step = 0; step < kDraftBlock; ++step) {
    stage.scheduler_read_mask[step] = 1;
    survival *= stage.calibrated_confidences[step];
    current_tau += survival;
    ++current_batch;
    ++evaluated;
    const float throughput =
        current_tau * stage.steps_per_second[current_batch];
    if (throughput <= best_throughput) {
      break;
    }
    best_throughput = throughput;
    best_tau = current_tau;
    best_batch = current_batch;
    best_length = step + 1;
  }
  stage.scheduled_prefix_lengths[0] = best_length;
  stage.scheduler_summary[0] = best_tau;
  stage.scheduler_summary[1] = static_cast<float>(best_batch);
  stage.scheduler_summary[2] = best_throughput;
  stage.scheduler_summary[3] = static_cast<float>(evaluated);
}

__device__ DSPARK_HEAD_ENTRY void execute_tail_phase(
    const ClaimedTask& task,
    TailStage stage,
    float* shared
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    ,
    const dspark_v4_tp2::DeviceContext& tp2_context
#endif
) {
  if (!stage.enabled) {
    return;
  }
  if (task.phase == generated::kPrefixSchedulerPhase) {
    execute_tail_prefix_scheduler(task, stage);
    return;
  }
#pragma unroll
  for (int step = 0; step < kDraftBlock; ++step) {
    if (task.phase == generated::kMarkovGatherPhases[step]) {
      execute_tail_markov_gather(
          task,
          stage,
          step
#ifdef DSPARK_V4_STATIC_TP2_TAIL
          ,
          tp2_context
#endif
      );
#ifdef DSPARK_V4_RELAXED_DAG
      // Stage-2 fusion: confidence rides the gather phase (its head_reduce
      // dependency moved onto the gather in the relaxed DAG).
      __syncthreads();
      execute_tail_confidence(task, stage, step, shared);
#endif
    } else if (task.phase == generated::kConfidencePhases[step]) {
      execute_tail_confidence(task, stage, step, shared);
    } else if (task.phase == generated::kMarkovW2Phases[step]) {
#ifdef DSPARK_V4_RELAXED_DAG
      execute_tail_markov_w2_fused(task, stage, step, shared);
#else
      if (stage.markov_w2_bf16 != nullptr) {
        execute_tail_markov_w2_bf16(task, stage, step);
      } else {
        execute_tail_markov_w2(task, stage, step);
      }
#endif
    } else if (task.phase == generated::kCorrectLogitPhases[step]) {
      execute_tail_correct_logits(task, stage, step, shared);
    } else if (task.phase == generated::kSoftmaxExpSumPhases[step]) {
      execute_tail_exp_sum(task, stage, step, shared);
    } else if (task.phase == generated::kSumReducePhases[step]) {
      execute_tail_sum_reduce(task, stage, step, shared);
    } else if (task.phase == generated::kNormalizeSamplePhases[step]) {
      execute_tail_normalize_sample(task, stage, step);
    } else {
      continue;
    }
    return;
  }
}

__device__ uint64_t initialize_epoch_counter(uint64_t* counter, uint32_t epoch) {
  uint64_t observed = atomicAdd(
      reinterpret_cast<unsigned long long*>(counter),
      0ULL);
  while (counter_epoch(observed) != epoch) {
    const uint64_t replacement = pack_epoch(epoch, 0);
    const uint64_t prior = atomicCAS(
        reinterpret_cast<unsigned long long*>(counter),
        observed,
        replacement);
    if (prior == observed) {
      return replacement;
    }
    observed = prior;
  }
  return observed;
}

// Same contract as initialize_epoch_counter, but the probe read is a relaxed
// load instead of atomicAdd(counter, 0). Correctness rests on the CAS, not on
// the probe: a stale observation simply fails the exchange and the loop
// retries with the true prior value.
__device__ uint64_t initialize_epoch_counter_lean(
    uint64_t* counter,
    uint32_t epoch) {
  uint64_t observed = load_relaxed_u64(counter);
  while (counter_epoch(observed) != epoch) {
    const uint64_t replacement = pack_epoch(epoch, 0);
    const uint64_t prior = atomicCAS(
        reinterpret_cast<unsigned long long*>(counter),
        observed,
        replacement);
    if (prior == observed) {
      return replacement;
    }
    observed = prior;
  }
  return observed;
}

__device__ __forceinline__ uint64_t* worker_word(
    SchedulerWorkspace workspace,
    int worker,
    int word) {
  return workspace.worker_state + static_cast<int64_t>(worker) * kWorkerWords + word;
}

__device__ __forceinline__ uint64_t* queue_word(
    SchedulerWorkspace workspace,
    int worker,
    int word) {
  return workspace.queue_state + static_cast<int64_t>(worker) * kQueueStateWords + word;
}

__device__ void queue_lock(SchedulerWorkspace workspace, int worker) {
  auto* lock = reinterpret_cast<unsigned long long*>(worker_word(workspace, worker, 0));
  while (atomicCAS(lock, 0ULL, 1ULL) != 0ULL) {
    // Intentionally no nanosleep: an owner holds this only for three scalar
    // operations, and sleeping would introduce an unobservable scheduler gap.
  }
}

__device__ __forceinline__ void queue_unlock(
    SchedulerWorkspace workspace,
    int worker) {
  __threadfence();
  atomicExch(
      reinterpret_cast<unsigned long long*>(worker_word(workspace, worker, 0)),
      0ULL);
}

__device__ bool queue_ready(
    SchedulerWorkspace workspace,
    int worker,
    uint32_t epoch) {
  const uint64_t observed = atomicAdd(
      reinterpret_cast<unsigned long long*>(worker_word(workspace, worker, 1)),
      0ULL);
  return static_cast<uint32_t>(observed) == epoch;
}

__device__ bool enqueue_local(
    SchedulerWorkspace workspace,
    int worker,
    int phase,
    uint32_t epoch) {
  if (!queue_ready(workspace, worker, epoch)) {
    return false;
  }
  queue_lock(workspace, worker);
  uint64_t head = *queue_word(workspace, worker, 0);
  uint64_t tail = *queue_word(workspace, worker, 1);
  bool inserted = false;
  if (tail - head < generated::kLocalQueueDepth) {
    workspace.ready_queues[
        static_cast<int64_t>(worker) * generated::kLocalQueueDepth
        + tail % generated::kLocalQueueDepth] = static_cast<uint32_t>(phase);
    *queue_word(workspace, worker, 1) = tail + 1;
    inserted = true;
  }
  queue_unlock(workspace, worker);
  return inserted;
}

__device__ int dequeue_local(
    SchedulerWorkspace workspace,
    int worker,
    uint32_t epoch) {
  if (!queue_ready(workspace, worker, epoch)) {
    return -1;
  }
  queue_lock(workspace, worker);
  uint64_t head = *queue_word(workspace, worker, 0);
  const uint64_t tail = *queue_word(workspace, worker, 1);
  int phase = -1;
  if (head < tail) {
    phase = static_cast<int>(workspace.ready_queues[
        static_cast<int64_t>(worker) * generated::kLocalQueueDepth
        + head % generated::kLocalQueueDepth]);
    *queue_word(workspace, worker, 0) = head + 1;
  }
  queue_unlock(workspace, worker);
  return phase;
}

__device__ void mark_queue_overflow(
    SchedulerWorkspace workspace,
    int worker) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(worker_word(workspace, worker, 3)),
      1ULL);
}

#ifdef DSPARK_V4_RELAXED_DAG
__device__ void broadcast_ready_hint(uint32_t epoch, int phase) {
  atomicExch(&g_ready_hint, pack_epoch(epoch, static_cast<uint32_t>(phase) + 1));
}
#endif

#ifdef DSPARK_V4_ROUTED_DYNAMIC_CLAIMS
__device__ __forceinline__ bool dynamically_claimed_routed_phase(int phase) {
#pragma unroll
  for (int layer = 0; layer < 3; ++layer) {
    if (phase == generated::kRoutedW13Phases[layer]
        || phase == generated::kRoutedW2Phases[layer]) {
      return true;
    }
  }
  return false;
}
#endif

__device__ void publish_phase(
    SchedulerWorkspace workspace,
    int phase,
    int preferred_worker,
    uint32_t epoch,
    bool remote,
    unsigned int sched_flags) {
#ifdef DSPARK_V4_STATIC_QUEUES
  // The AOT worker streams already name every claim. Only the completed-item
  // counter and acquire-visible ready epoch are live; next-tile ownership,
  // ready rings, queue locks, and publish odometers belong to the dynamic
  // scheduler and would be pure overhead here.
  store_relaxed_u64(
      &workspace.completed_tiles[phase], pack_epoch(epoch, 0));
#ifdef DSPARK_V4_ROUTED_DYNAMIC_CLAIMS
  // Sole publisher resets claims before making the phase acquire-visible.
  if (dynamically_claimed_routed_phase(phase)) {
    store_relaxed_u64(&workspace.next_tile[phase], pack_epoch(epoch, 0));
  }
#endif
#ifdef DSPARK_V4_FINE_GRAINED_OVERLAP
  // A streaming consumer is not released by the producer's phase-complete
  // successor walk. Initialize its completion counter before making the
  // producer claimable; fused producer CTAs credit its physical half-band
  // completions directly. This ordering prevents an early completion from
  // racing a late phase-level reset.
  const uint16_t streaming_consumer =
      generated::kStreamingConsumerForPhase[phase];
  if (streaming_consumer != generated::kNoStreamingConsumer) {
    store_relaxed_u64(
        &workspace.completed_tiles[streaming_consumer], pack_epoch(epoch, 0));
    store_release_u64(
        &workspace.publish_epoch[streaming_consumer], pack_epoch(epoch, 1));
  }
#endif
#ifdef DSPARK_V4_COMPLETION_RELEASE
  store_release_u64(&workspace.publish_epoch[phase], pack_epoch(epoch, 1));
#else
  atomicExch(
      reinterpret_cast<unsigned long long*>(&workspace.publish_epoch[phase]),
      pack_epoch(epoch, 1));
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  g_probe_phase[phase * kProbePhaseSlots + kProbePublishNs] = globaltimer_ns();
#endif
  (void)preferred_worker;
  (void)remote;
  (void)sched_flags;
#else
  const bool lean = (sched_flags & kSchedLeanPublish) != 0;
  if (lean) {
    // Sole writer at this instant (see kSchedLeanPublish): store, do not CAS.
    if ((sched_flags & kSchedPlainInit) != 0) {
      store_relaxed_u64(&workspace.next_tile[phase], pack_epoch(epoch, 0));
      store_relaxed_u64(&workspace.completed_tiles[phase], pack_epoch(epoch, 0));
    } else {
      atomicExch(
          reinterpret_cast<unsigned long long*>(&workspace.next_tile[phase]),
          pack_epoch(epoch, 0));
      atomicExch(
          reinterpret_cast<unsigned long long*>(&workspace.completed_tiles[phase]),
          pack_epoch(epoch, 0));
    }
  } else {
    initialize_epoch_counter(&workspace.next_tile[phase], epoch);
    initialize_epoch_counter(&workspace.completed_tiles[phase], epoch);
  }
#ifdef DSPARK_V4_COMPLETION_RELEASE
  store_release_u64(&workspace.publish_epoch[phase], pack_epoch(epoch, 1));
#else
  atomicExch(
      reinterpret_cast<unsigned long long*>(&workspace.publish_epoch[phase]),
      pack_epoch(epoch, 1));
#endif
#ifdef DSPARK_V4_RELAXED_DAG
  // Only the non-ready-ring discovery path ever reads g_ready_hint.
  if (!lean || (sched_flags & kSchedReadyRing) == 0) {
    broadcast_ready_hint(epoch, phase);
  }
#endif
  {
    // Publish odometer + ring. The odometer is what lets an idle CTA answer
    // "has anything become claimable since I last looked?" with ONE load,
    // instead of a 147-lock steal sweep plus a 94-phase scan every round.
    const unsigned long long ticket = atomicAdd(&g_ready_ring_cursor, 1ULL);
    if ((sched_flags & kSchedPlainRing) != 0) {
      store_relaxed_u64(
          reinterpret_cast<uint64_t*>(
              &g_ready_ring[ticket & (kReadyRingSlots - 1)]),
          pack_epoch(epoch, static_cast<uint32_t>(phase) + 1));
    } else {
      atomicExch(
          &g_ready_ring[ticket & (kReadyRingSlots - 1)],
          pack_epoch(epoch, static_cast<uint32_t>(phase) + 1));
    }
    __threadfence();
    atomicAdd(&g_publish_seq, 1ULL);
  }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  g_probe_phase[phase * kProbePhaseSlots + kProbePublishNs] = globaltimer_ns();
#endif
  if (enqueue_local(workspace, preferred_worker, phase, epoch)) {
    return;
  }
  for (int offset = 1; offset < generated::kWorkers; ++offset) {
    const int worker = (preferred_worker + offset) % generated::kWorkers;
    if (enqueue_local(workspace, worker, phase, epoch)) {
      return;
    }
  }
  // The ready epoch is the lossless fallback. Idle controllers scan it, so a
  // full queue can affect performance but cannot lose a phase.
  mark_queue_overflow(workspace, preferred_worker);
  (void)remote;
#endif
}

__device__ bool tiles_remain(
    SchedulerWorkspace workspace,
    int phase,
    uint32_t epoch,
    bool relaxed_probe) {
  const uint64_t next = relaxed_probe
      ? load_relaxed_u64(&workspace.next_tile[phase])
      : atomicAdd(
            reinterpret_cast<unsigned long long*>(&workspace.next_tile[phase]),
            0ULL);
  return counter_epoch(next) == epoch
      && counter_value(next) < generated::kWorkUnits[phase];
}

__device__ bool phase_is_ready(
    SchedulerWorkspace workspace,
    int phase,
    uint32_t epoch,
    bool relaxed_probe) {
  const uint64_t state = relaxed_probe
      ? load_relaxed_u64(&workspace.publish_epoch[phase])
      : atomicAdd(
            reinterpret_cast<unsigned long long*>(
                &workspace.publish_epoch[phase]),
            0ULL);
  if (counter_epoch(state) != epoch || counter_value(state) != 1) {
    return false;
  }
  return tiles_remain(workspace, phase, epoch, relaxed_probe);
}

__device__ int scan_ready_phase(
    SchedulerWorkspace workspace,
    int worker,
    uint32_t epoch,
    uint32_t round,
    bool relaxed_probe,
    bool lean) {
  const int start = (worker * 17 + round) % generated::kPhaseCount;
  for (int index = 0; index < generated::kPhaseCount; ++index) {
    const int phase = (start + index) % generated::kPhaseCount;
    // next_tile is epoch-stamped by publish_phase before publish_epoch is,
    // and publish_phase only runs once every dependency has arrived, so the
    // tiles-remain half alone is a sound (and half as expensive) probe.
    const bool ready = lean
        ? tiles_remain(workspace, phase, epoch, relaxed_probe)
        : phase_is_ready(workspace, phase, epoch, relaxed_probe);
    if (ready) {
      return phase;
    }
  }
  return -1;
}

// Recently published phases, newest first. Replaces the single-word
// g_ready_hint lookup when kSchedReadyRing is selected: a fan-out publish
// writes k successors back to back, and a one-word hint keeps only the last,
// which is exactly when idle CTAs fall back to a full deep sweep.
__device__ int ring_ready_phase(
    SchedulerWorkspace workspace,
    uint32_t epoch,
    bool express_narrow_only,
    bool relaxed_probe,
    unsigned long long* floor_slot,
    unsigned long long seq_cursor,
    bool lean) {
  const unsigned long long cursor = lean
      ? seq_cursor
      : load_relaxed_u64(
            reinterpret_cast<const uint64_t*>(&g_ready_ring_cursor));
  unsigned long long floor =
      cursor > static_cast<unsigned long long>(kReadyRingSlots)
      ? cursor - kReadyRingSlots
      : 0ULL;
  if (floor_slot != nullptr && *floor_slot > floor) {
    floor = *floor_slot;
  }
  for (unsigned long long index = cursor; index > floor; --index) {
    const unsigned long long ticket = index - 1;
    const uint64_t entry = load_relaxed_u64(reinterpret_cast<const uint64_t*>(
        &g_ready_ring[ticket & (kReadyRingSlots - 1)]));
    if (counter_epoch(entry) != epoch || counter_value(entry) == 0) {
      continue;
    }
    const int candidate = static_cast<int>(counter_value(entry)) - 1;
    if (candidate < 0 || candidate >= generated::kPhaseCount) {
      continue;
    }
    if (express_narrow_only && generated::kWorkUnits[candidate] > 256) {
      continue;
    }
    // A ring entry tagged with this epoch is itself proof that publish_epoch
    // was stored (publish_phase orders the two), so the lean probe checks
    // only the claimable-tiles half of phase_is_ready.
    const bool ready = lean
        ? tiles_remain(workspace, candidate, epoch, relaxed_probe)
        : phase_is_ready(workspace, candidate, epoch, relaxed_probe);
    if (ready) {
      // Deliberately do NOT advance the floor here: an OLDER slot may still
      // be handing out tiles, and excluding it would strand a wide phase
      // behind a newer narrow one (measured as a 73 us regression, run 3).
      return candidate;
    }
  }
  if (floor_slot != nullptr) {
    // Leave the newest slot in range: publish_phase advances the cursor a few
    // instructions before the slot write lands, and re-reading one slot is
    // cheaper than depending on the periodic deep sweep to recover it.
    *floor_slot = cursor > 0ULL ? cursor - 1ULL : 0ULL;
  }
  return -1;
}

#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
#define DSPARK_PROBE_ADD(slot, value) probe[(slot)] += (value)
#else
#define DSPARK_PROBE_ADD(slot, value) \
  do {                                \
  } while (0)
#endif

__device__ ClaimedTask claim_task(
    SchedulerWorkspace workspace,
    int worker,
    uint32_t epoch,
    uint32_t round,
    const V4SchedPolicy& policy,
    unsigned long long* state
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
    ,
    unsigned long long* probe
#endif
) {
  int source_worker = worker;
  int phase = -1;
  const bool relaxed_probe = (policy.flags & kSchedRelaxedProbe) != 0;
  const bool seq_gate = (policy.flags & kSchedSeqGate) != 0;
  const bool local_hint = (policy.flags & kSchedLocalHint) != 0;
  unsigned long long* seen_seq = &state[kSchedSeenSeq];
  unsigned long long* ring_floor = (policy.flags & kSchedRingWatermark) != 0
      ? &state[kSchedRingFloor]
      : nullptr;
  bool deep = true;
  bool look = true;
  unsigned long long seq = 0;
  if (seq_gate) {
    // One relaxed load answers "did anything become claimable since I last
    // looked?". A phase only becomes claimable at publish, and publish bumps
    // this odometer after the ring slot is visible, so an unchanged odometer
    // means a discovery sweep cannot find anything the previous sweep missed.
    seq = load_relaxed_u64(reinterpret_cast<const uint64_t*>(&g_publish_seq));
    deep = policy.deep_period != 0 && (round % policy.deep_period) == 0;
    look = deep || seq != *seen_seq;
    if (look) {
      // A publish can drop a phase straight into this CTA's ready queue.
      state[kSchedLocalMaybe] = 1ULL;
    }
  }
  DSPARK_PROBE_ADD(kProbeClaimCalls, 1ULL);
  if (deep) {
    DSPARK_PROBE_ADD(kProbeDeepSweeps, 1ULL);
  }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  uint64_t stamp = globaltimer_ns();
#endif
  (void)look;
#ifdef DSPARK_V4_RELAXED_DAG
  if (look) {
    if ((policy.flags & kSchedReadyRing) != 0) {
      phase = ring_ready_phase(
          workspace,
          epoch,
          static_cast<unsigned int>(worker) < policy.express_workers,
          relaxed_probe,
          ring_floor,
          seq,
          seq_gate && (policy.flags & kSchedRingLean) != 0);
    } else {
      const unsigned long long hint = atomicAdd(&g_ready_hint, 0ULL);
      if (counter_epoch(hint) == epoch && counter_value(hint) != 0) {
        const int candidate = static_cast<int>(counter_value(hint)) - 1;
        const bool narrow = generated::kWorkUnits[candidate] <= 256;
        if ((static_cast<unsigned int>(worker) >= policy.express_workers
             || narrow)
            && phase_is_ready(workspace, candidate, epoch, relaxed_probe)) {
          phase = candidate;
        }
      }
    }
  }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  {
    const uint64_t now = globaltimer_ns();
    probe[kProbeNsRing] += now - stamp;
    stamp = now;
  }
#endif
  if (phase >= 0) {
    DSPARK_PROBE_ADD(kProbeHitRing, 1ULL);
  }
  if (phase < 0) {
  // Express lane (contention re-aim): workers 0-3 serve ONLY narrow phases
  // (<=256 tiles), so critical-path glue never waits behind a claimed
  // 40-300 us wide tile. Publishes into their local queues are drained by
  // other workers' steals; when no narrow work is ready they idle-prefetch
  // and poll for termination like any other worker.
  const bool express =
      static_cast<unsigned int>(worker) < policy.express_workers;
  if (express) {
    if (deep) {
      for (int candidate = 0; candidate < generated::kPhaseCount; ++candidate) {
        if (generated::kWorkUnits[candidate] <= 256
            && phase_is_ready(workspace, candidate, epoch, relaxed_probe)) {
          phase = candidate;
          break;
        }
      }
    }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
    {
      const uint64_t now = globaltimer_ns();
      probe[kProbeNsExpress] += now - stamp;
      stamp = now;
    }
#endif
    if (phase < 0) {
      if (seq_gate) {
        *seen_seq = seq;
      }
      return {-1, 0, 0, 0, false};
    }
    DSPARK_PROBE_ADD(kProbeHitExpress, 1ULL);
  }
  }
#endif
  if (phase < 0 && (!local_hint || state[kSchedLocalMaybe] != 0ULL)) {
    phase = dequeue_local(workspace, worker, epoch);
    if (phase < 0) {
      state[kSchedLocalMaybe] = 0ULL;
    }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
    {
      const uint64_t now = globaltimer_ns();
      probe[kProbeNsLocal] += now - stamp;
      stamp = now;
    }
#endif
    if (phase >= 0) {
      DSPARK_PROBE_ADD(kProbeHitLocal, 1ULL);
    }
  }
  if (phase < 0 && deep && (policy.flags & kSchedStealSweep) != 0) {
    // Lock-based victim sweep. Retained as a selectable policy (it is the
    // pre-2026-08-08 behaviour) but it is pure redundancy for discovery: a
    // phase with unclaimed tiles is always visible to scan_ready_phase,
    // whether or not it also sits in someone's local queue.
    for (int offset = 1; offset < generated::kWorkers; ++offset) {
      const int victim = (worker + offset + round) % generated::kWorkers;
      phase = dequeue_local(workspace, victim, epoch);
      if (phase >= 0) {
        source_worker = victim;
        break;
      }
    }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
    {
      const uint64_t now = globaltimer_ns();
      probe[kProbeNsSteal] += now - stamp;
      stamp = now;
    }
#endif
    if (phase >= 0) {
      DSPARK_PROBE_ADD(kProbeHitSteal, 1ULL);
    }
  }
  if (phase < 0 && deep) {
    phase = scan_ready_phase(
        workspace,
        worker,
        epoch,
        round,
        relaxed_probe,
        (policy.flags & kSchedRingLean) != 0);
    source_worker = worker;
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
    {
      const uint64_t now = globaltimer_ns();
      probe[kProbeNsScan] += now - stamp;
      stamp = now;
    }
#endif
    if (phase >= 0) {
      DSPARK_PROBE_ADD(kProbeHitScan, 1ULL);
    }
  }
  const bool lean_final = (policy.flags & kSchedRingLean) != 0;
  if (phase < 0
      || !(lean_final ? tiles_remain(workspace, phase, epoch, relaxed_probe)
                      : phase_is_ready(workspace, phase, epoch, relaxed_probe))) {
    if (seq_gate) {
      // Nothing claimable was found: the odometer value we just observed is a
      // truthful "already looked at this" watermark.
      *seen_seq = seq;
    }
    return {-1, 0, 0, 0, false};
  }
  if (seq_gate) {
    // A candidate existed. Force one more look next round: the ring may hold
    // other freshly published phases behind this one, and a lost claim race
    // must not be mistaken for "nothing new".
    *seen_seq = 0ULL;
  }
  uint32_t claim_chunk = generated::kClaimChunks[phase];
#if defined(DSPARK_ROUTED_CLAIM_CHUNK)
  // FALSIFIED dynamic-scheduler experiment: reprice routed load balancing
  // after the K=128 -> K=512 body change. The AOT static task stream encodes
  // its claim
  // indices using generated::kClaimChunks and therefore deliberately rejects
  // this define at the extension boundary. Chunk 2 measured 1.973664 ms
  // versus 1.914792 ms mean chunk-4 controls (+58.872 us / 3.07%), so the
  // greater wave count does not repay doubled claim/scheduling work.
  bool routed_phase = false;
  CUTE_UNROLL
  for (int layer = 0; layer < 3; ++layer) {
    routed_phase = routed_phase
        || phase == generated::kRoutedW13Phases[layer]
        || phase == generated::kRoutedW2Phases[layer];
  }
  if (routed_phase) {
    claim_chunk = DSPARK_ROUTED_CLAIM_CHUNK;
  }
#endif
  const uint64_t claimed = atomicAdd(
      reinterpret_cast<unsigned long long*>(&workspace.next_tile[phase]),
      static_cast<unsigned long long>(claim_chunk));
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  probe[kProbeNsClaimAtomic] += globaltimer_ns() - stamp;
  probe[kProbeClaimAtomics] += 1ULL;
  atomicAdd(&g_probe_phase[phase * kProbePhaseSlots + kProbeClaimTries], 1ULL);
#endif
  if (counter_epoch(claimed) != epoch) {
    return {-1, 0, 0, 0, false};
  }
  const uint32_t begin = counter_value(claimed);
  const uint32_t work_units = generated::kWorkUnits[phase];
  if (begin >= work_units) {
    return {-1, 0, 0, 0, false};
  }
  const uint32_t end = begin + claim_chunk < work_units
      ? begin + claim_chunk
      : work_units;
  if (end < work_units) {
    if (enqueue_local(workspace, worker, phase, epoch)) {
      state[kSchedLocalMaybe] = 1ULL;
    } else {
      mark_queue_overflow(workspace, worker);
    }
  }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  {
    const unsigned long long now = globaltimer_ns();
    probe[kProbeClaimHits] += 1ULL;
    atomicMin(
        &g_probe_phase[phase * kProbePhaseSlots + kProbeFirstClaimNs], now);
    atomicMax(
        &g_probe_phase[phase * kProbePhaseSlots + kProbeLastClaimNs], now);
  }
#endif
  return {
      phase,
      begin,
      end,
      begin / claim_chunk,
      source_worker != worker,
  };
}

__device__ void record_trace(
    int64_t* trace,
    int32_t* trace_counts,
    int worker,
    int phase,
    TraceRole role,
    TraceSegment segment,
    uint64_t start,
    uint64_t end,
    uint32_t work_item,
    uint32_t ticket,
    int64_t flags) {
  if constexpr (!kTraceEnabled) {
    return;
  }
  int count = trace_counts[worker];
  if (count >= generated::kTraceCapacity) {
    return;
  }
  int64_t* event = trace
      + (static_cast<int64_t>(worker) * generated::kTraceCapacity + count)
          * generated::kTraceColumns;
  event[0] = phase;
  event[1] = static_cast<int64_t>(role);
  event[2] = static_cast<int64_t>(segment);
  event[3] = static_cast<int64_t>(start);
  event[4] = static_cast<int64_t>(end);
  event[5] = work_item;
  event[6] = ticket;
  event[7] = flags;
  trace_counts[worker] = count + 1;
}

__device__ void finalize_proposal_if_terminal(
    SchedulerWorkspace workspace,
    const ClaimedTask& task,
    uint32_t epoch,
    uint64_t* launch_audit) {
  if (task.phase != generated::kPhaseCount - 1) {
    return;
  }
#ifndef DSPARK_V4_RELAXED_DAG
  uint64_t checksum = 0;
#endif
  uint64_t overflows = 0;
  for (int index = 0; index < generated::kWorkers; ++index) {
#ifndef DSPARK_V4_RELAXED_DAG
    checksum += *worker_word(workspace, index, 2);
#endif
    overflows += *worker_word(workspace, index, 3);
  }
  launch_audit[3] = globaltimer_ns();
  launch_audit[4] = generated::kPhaseCount;
  launch_audit[5] = overflows;
#ifndef DSPARK_V4_RELAXED_DAG
  launch_audit[6] = checksum;
#endif
  launch_audit[7] =
      g_idle_prefetch_cursor < generated::kWeightArenaBytes
      ? g_idle_prefetch_cursor
      : generated::kWeightArenaBytes;
  __threadfence_system();
  atomicExch(
      reinterpret_cast<unsigned long long*>(worker_word(workspace, 0, 7)),
      pack_epoch(epoch, 1));
}

__device__ void finish_phase(
    SchedulerWorkspace workspace,
    const ClaimedTask& task,
    int worker,
    uint32_t epoch,
    uint64_t* launch_audit,
    unsigned int sched_flags
#ifdef DSPARK_V4_GREEDY_TAIL
    ,
    TailStage tail_stage
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    ,
    const dspark_v4_tp2::DeviceContext& tp2_context
#endif
#endif
) {
  const uint32_t count = task.end - task.begin;
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  // Entry stamp: the boundary between "this task's body ended" and "the
  // scheduler began reacting to it". Stored into every successor this call
  // goes on to publish, so each published phase carries the timestamp of the
  // completion that actually unblocked it.
  const uint64_t finish_entry_ns = globaltimer_ns();
#endif
#ifdef DSPARK_V4_GREEDY_TAIL
  int greedy_markov_step = -1;
#pragma unroll
  for (int step = 0; step < kDraftBlock; ++step) {
    if (tail_stage.enabled
        && task.phase == generated::kMarkovW2Phases[step]) {
      greedy_markov_step = step;
    }
  }
  if (greedy_markov_step >= 0) {
    // Every Markov CTA must publish its packed-argmax contribution before
    // incrementing the completion counter.
    __threadfence();
  }
#endif
#ifdef DSPARK_PRIVATE_COMPLETION_RED_0908
  if (greedy_markov_step < 0 && task.phase != generated::kPhaseCount - 1) {
    // The caller has joined all producer lanes. Consumers acquire the exact
    // completed total through this release RMW chain. Ordinary Direct2 phases
    // neither publish a second flag nor perform a last-CTA output epilogue.
    // Markov and terminal retain the original returning atomic path below.
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;"
                 : : "l"(&workspace.completed_tiles[task.phase]),
                     "l"(static_cast<uint64_t>(count)) : "memory");
    return;
  }
#endif
#ifdef DSPARK_V4_COMPLETION_RELEASE
  const uint64_t prior = complete_items_release(
      &workspace.completed_tiles[task.phase], count);
#else
  const uint64_t prior = atomicAdd(
      reinterpret_cast<unsigned long long*>(&workspace.completed_tiles[task.phase]),
      static_cast<unsigned long long>(count));
#endif
  if (counter_epoch(prior) != epoch
      || counter_value(prior) + count != generated::kWorkUnits[task.phase]) {
    return;
  }
#ifdef DSPARK_V4_COMPLETION_RELEASE
#if DSPARK_V4_DIRECT_DEPENDENCIES == 2
  if (greedy_markov_step >= 0)
#endif
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
#ifdef DSPARK_V4_GREEDY_TAIL
  if (greedy_markov_step >= 0) {
    const uint64_t packed = *tail_argmax_slot(tail_stage, greedy_markov_step);
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    if (!dspark_v4_tp2::rank0_merge_step(
            tp2_context,
            packed,
            greedy_markov_step,
            &tail_stage.output_ids[greedy_markov_step + 1])) {
      // Do not publish the failed Markov phase: its successor gather would
      // otherwise index Markov W1 with poisoned/stale output_ids.  Mark the
      // proposal terminal so every CTA exits once its already-claimed safe
      // work drains; the protocol status remains the authoritative failure.
      __threadfence_system();
      atomicExch(
          reinterpret_cast<unsigned long long*>(worker_word(workspace, 0, 7)),
          pack_epoch(epoch, 1));
      return;
    }
#else
    tail_stage.output_ids[greedy_markov_step + 1] = static_cast<int32_t>(
        0xFFFFFFFFu - static_cast<uint32_t>(packed));
    // Batched serving: one packed argmax per (step, batch element). Element 0
    // was published above so the batch-1 path is byte-for-byte the frozen one.
    for (int element = 1; element < kBatch; ++element) {
      const uint64_t element_packed =
          *tail_argmax_slot(tail_stage, greedy_markov_step, element);
      tail_stage.output_ids
          [element * (kDraftBlock + 1) + greedy_markov_step + 1] =
              static_cast<int32_t>(
                  0xFFFFFFFFu - static_cast<uint32_t>(element_packed));
    }
#endif
    // In the greedy build output_ids[step + 1] feeds the next Markov gather.
    // Order that store before the dependency-arrival publication below.
    __threadfence();
  }
#endif
#ifdef DSPARK_V4_COMPLETION_RELEASE
#if DSPARK_V4_DIRECT_DEPENDENCIES == 2
  if (greedy_markov_step >= 0)
#endif
    store_release_u64(&workspace.publish_epoch[task.phase], pack_epoch(epoch, 2));
#else
  atomicExch(
      reinterpret_cast<unsigned long long*>(&workspace.publish_epoch[task.phase]),
      pack_epoch(epoch, 2));
#endif
#ifndef DSPARK_V4_DIRECT_DEPENDENCIES
  const int begin = generated::kSuccessorOffsets[task.phase];
  const int end = generated::kSuccessorOffsets[task.phase + 1];
  const bool lean = (sched_flags & kSchedLeanPublish) != 0;
  const bool solo_dep = (sched_flags & kSchedSoloDep) != 0;
  for (int index = begin; index < end; ++index) {
    const int successor = generated::kSuccessors[index];
    // Sole predecessor: this completion IS the successor's readiness, so the
    // arrival counter has nothing to contribute. Publish straight through.
    const bool solo =
        solo_dep && generated::kDependencyCounts[successor] == 1;
    if (!solo) {
      if (lean) {
        initialize_epoch_counter_lean(
            &workspace.dependency_arrivals[successor], epoch);
      } else {
        initialize_epoch_counter(
            &workspace.dependency_arrivals[successor], epoch);
      }
      const uint64_t dependency_prior = atomicAdd(
          reinterpret_cast<unsigned long long*>(
              &workspace.dependency_arrivals[successor]),
          1ULL);
      if (counter_epoch(dependency_prior) != epoch
          || counter_value(dependency_prior) + 1
              != generated::kDependencyCounts[successor]) {
        continue;
      }
    }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
    g_probe_phase[successor * kProbePhaseSlots + kProbeTriggerEndNs] =
        finish_entry_ns;
    g_probe_phase[successor * kProbePhaseSlots + kProbeTriggerPhase] =
        static_cast<unsigned long long>(task.phase) + 1ULL;
    g_probe_phase[successor * kProbePhaseSlots + kProbeArriveNs] =
        globaltimer_ns();
#endif
    publish_phase(workspace, successor, worker, epoch, true, sched_flags);
  }
#endif
  finalize_proposal_if_terminal(workspace, task, epoch, launch_audit);
}

__device__ bool proposal_done(
    SchedulerWorkspace workspace,
    uint32_t epoch) {
  if (!queue_ready(workspace, 0, epoch)) {
    return false;
  }
  const uint64_t done = atomicAdd(
      reinterpret_cast<unsigned long long*>(worker_word(workspace, 0, 7)),
      0ULL);
  return counter_epoch(done) == epoch && counter_value(done) == 1;
}

// R15: the launch bound is the register ceiling. DSPARK_V4_LAUNCH_MAX_THREADS
// defaults to 0, which resolves to generated::kThreads -- the shipped
// __launch_bounds__(256, 1) -- so an undefined build is byte-identical.
constexpr int kLaunchMaxThreads =
    DSPARK_V4_LAUNCH_MAX_THREADS ? DSPARK_V4_LAUNCH_MAX_THREADS
                                 : generated::kThreads;
static_assert(
    kLaunchMaxThreads >= generated::kThreads,
    "the launch bound is a promise about the MAXIMUM block size");

__global__ __launch_bounds__(kLaunchMaxThreads, 1) void dspark_v4_scheduler_kernel(
    const uint8_t* weight_arena,
    const int64_t* weight_offsets,
    const __nv_bfloat16* main_hidden,
    __nv_bfloat16* main_output,
    const int32_t* anchor,
    __nv_bfloat16* embedding_output,
    const float* rope_cos_sin,
    __nv_bfloat16* kv_cache,
    int32_t* output_ids,
    float* corrected_logits,
    float* probabilities,
    float* confidence_logits,
    float* calibrated_confidences,
    int32_t* scheduled_prefix_lengths,
    uint8_t* scheduler_read_mask,
    float* scheduler_summary,
    const float* uniforms,
    const float* sts_temperatures,
    const float* steps_per_second,
    float sampling_temperature,
    bool calibration_enabled,
    bool prefix_enabled,
    int start_pos,
    uint8_t* raw_workspace,
    int64_t* trace,
    int32_t* trace_counts,
    uint64_t* launch_audit,
    uint32_t epoch,
    uint32_t execution_mask,
    uint8_t draft_layer_mask,
    int extra_offset_slots
#ifdef DSPARK_V4_TMA_WEIGHTS
    ,
    const CUtensorMap* w13_weight_tma,
    const CUtensorMap* w2_weight_tma
#endif
#ifdef DSPARK_V4_STATIC_TP2_TAIL
    ,
    dspark_v4_tp2::DeviceContext tp2_context
#endif
) {
#if defined(DSPARK_V4_COMPILED_EXECUTION_MASK)
  static_assert(
      (DSPARK_V4_COMPILED_EXECUTION_MASK & ~kExecuteAll) == 0,
      "compiled execution mask contains unsupported phase bits");
  static_assert(
      DSPARK_V4_COMPILED_DRAFT_LAYER_MASK >= 1
          && DSPARK_V4_COMPILED_DRAFT_LAYER_MASK <= 0x7,
      "compiled draft-layer mask must select one to three layers");
  execution_mask = DSPARK_V4_COMPILED_EXECUTION_MASK;
  draft_layer_mask = DSPARK_V4_COMPILED_DRAFT_LAYER_MASK;
#endif
#ifdef DSPARK_V4_INTEGRATED_PROPOSAL_IO
  static_assert(generated::kWorkers == 152 && generated::kThreads == 256);
  if (start_pos != -1 || epoch != UINT32_MAX) __trap();
  auto* io = reinterpret_cast<dspark_proposal_io::IntegrationContext*>(
      raw_workspace + generated::kWorkspaceBytes + 2 * sizeof(int64_t));
  // Every CTA latches the old epoch before the preparation quorum. Use the
  // returned next epoch; rereading the global epoch would race CTA0's update.
  epoch = dspark_proposal_io::prepare(
      io, const_cast<__nv_bfloat16*>(main_hidden), const_cast<int32_t*>(anchor),
      const_cast<float*>(rope_cos_sin), kv_cache,
      reinterpret_cast<int64_t*>(raw_workspace + generated::kWorkspaceBytes),
      reinterpret_cast<int64_t*>(raw_workspace + generated::kWorkspaceBytes + sizeof(int64_t)));
#endif
#ifdef DSPARK_V4_TMA_WEIGHTS
  if (w13_weight_tma != nullptr && w2_weight_tma != nullptr) {
    // The tensor maps live in global memory and were written through the
    // generic proxy (host memcpy at prepare time). Every thread acquires
    // its own tensormap-proxy ordering before any cp.async.bulk.tensor can
    // consume the descriptors; one thread prefetches them.
    cute::tma_descriptor_fence_acquire(w13_weight_tma);
    cute::tma_descriptor_fence_acquire(w2_weight_tma);
    if (threadIdx.x == 0) {
      cute::prefetch_tma_descriptor(w13_weight_tma);
      cute::prefetch_tma_descriptor(w2_weight_tma);
    }
  }
#endif
#ifdef DSPARK_V4_STATIC_TP2_TAIL
  bool tp2_start_pos_valid = true;
#endif
#ifdef DSPARK_V4_RELAXED_DAG
  if (start_pos == -1) {
    static_assert(
        generated::kWorkspaceBytes % alignof(int64_t) == 0,
        "device start position must be naturally aligned");
    const int64_t device_start_pos = *reinterpret_cast<const int64_t*>(
        raw_workspace + generated::kWorkspaceBytes);
    if (device_start_pos <= 0 || device_start_pos > 0x7fffffffLL) {
#ifdef DSPARK_V4_STATIC_TP2_TAIL
      if (tp2_context.enabled && tp2_context.serving) {
        // Serving must fail closed through the coordinated TP2 request so
        // rank 1 retires and proposal outputs are cleared.  Use a harmless
        // placeholder while stage descriptors are assembled; no root can be
        // published after rank0_begin() rejects the device input below.
        tp2_start_pos_valid = false;
        start_pos = 1;
      } else {
        __trap();
      }
#else
      __trap();
#endif
    } else {
      start_pos = static_cast<int>(device_start_pos);
    }
  }
#ifdef DSPARK_V4_FULL_LOOP_DEVICE_EPOCH
  if (epoch == UINT32_MAX) {
    static_assert(
        generated::kWorkspaceBytes % alignof(int64_t) == 0,
        "device proposal epoch must be naturally aligned");
    const int64_t device_epoch = *reinterpret_cast<const int64_t*>(
        raw_workspace + generated::kWorkspaceBytes + sizeof(int64_t));
    if (device_epoch <= 0 || device_epoch >= UINT32_MAX) {
      __trap();
    }
    epoch = static_cast<uint32_t>(device_epoch);
  }
#endif
#endif
  SchedulerWorkspace workspace{
      reinterpret_cast<uint64_t*>(raw_workspace + generated::kDependencyOffset),
      reinterpret_cast<uint64_t*>(raw_workspace + generated::kNextTileOffset),
      reinterpret_cast<uint64_t*>(raw_workspace + generated::kCompletedTileOffset),
      reinterpret_cast<uint64_t*>(raw_workspace + generated::kPublishEpochOffset),
      reinterpret_cast<uint32_t*>(raw_workspace + generated::kReadyQueueOffset),
      reinterpret_cast<uint64_t*>(raw_workspace + generated::kQueueStateOffset),
      reinterpret_cast<uint64_t*>(raw_workspace + generated::kWorkerStateOffset),
  };
#ifdef DSPARK_V4_SHARED_STAGE_CONTEXT
  // Uniform phase descriptors are initialized once per CTA. Keeping them in
  // shared memory avoids retaining every phase's pointer set in each thread
  // across the persistent dispatch loop. Phase functions still get local copies.
  __shared__ MainStage main_stage;
  __shared__ EmbeddingStage embedding_stage;
  __shared__ MainKvStage main_kv_stage;
  __shared__ Layer0AttnHcStage layer0_attn_hc_stage;
  __shared__ Layer0AttnProjectionStage layer0_attn_projection_stage;
  __shared__ Layer0SparseAttentionStage layer0_sparse_attention_stage;
  __shared__ Layer0AttentionOutputStage layer0_attention_output_stage;
  __shared__ Layer0AttnHcStage layer0_ffn_hc_stage;
  __shared__ Layer0RouterStage layer0_router_stage;
  __shared__ Layer0ExpertsStage layer0_experts_stage;
  __shared__ HeadStage head_stage;
  __shared__ TailStage tail_stage;
#define DSPARK_INIT_STAGE(type, name) name = type
  if (threadIdx.x == 0) {
#else
#define DSPARK_INIT_STAGE(type, name) type name
#endif
  DSPARK_INIT_STAGE(MainStage, main_stage){
      main_hidden,
      main_output,
      reinterpret_cast<cutlass::float_e4m3_t*>(
          raw_workspace + generated::kMainQuantizedOffset),
      raw_workspace + generated::kMainScaleOffset,
      reinterpret_cast<float*>(raw_workspace + generated::kMainPartialOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kMainFp32Offset),
      reinterpret_cast<float*>(raw_workspace + generated::kMainRmsPartialOffset),
      execution_mask & kExecuteMain
          ? reinterpret_cast<const cutlass::float_e4m3_t*>(
                weight_arena + weight_offsets[generated::kMainProjWeightSlot])
          : nullptr,
      execution_mask & kExecuteMain
          ? weight_arena + weight_offsets[generated::kMainProjScaleSlot]
          : nullptr,
      execution_mask & kExecuteMain
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kMainNormWeightSlot])
          : nullptr,
      static_cast<bool>(execution_mask & kExecuteMain),
  };
  DSPARK_INIT_STAGE(EmbeddingStage, embedding_stage){
      anchor,
      embedding_output,
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kHiddenStreamsOffset),
      execution_mask & kExecuteEmbedding
          ? reinterpret_cast<const __nv_bfloat16*>(
                weight_arena + weight_offsets[generated::kEmbeddingWeightSlot])
          : nullptr,
      static_cast<bool>(execution_mask & kExecuteEmbedding),
  };
  DSPARK_INIT_STAGE(MainKvStage, main_kv_stage){
      main_output,
      reinterpret_cast<cutlass::float_e4m3_t*>(
          raw_workspace + generated::kMainProjectedQuantizedOffset),
      raw_workspace + generated::kMainProjectedScaleOffset,
      reinterpret_cast<float*>(raw_workspace + generated::kMainKvPartialOffset),
      {nullptr, nullptr, nullptr},
      {nullptr, nullptr, nullptr},
      {nullptr, nullptr, nullptr},
      rope_cos_sin,
      kv_cache,
      start_pos,
      static_cast<bool>(execution_mask & kExecuteMainKv),
  };
  if (execution_mask & kExecuteMainKv) {
#pragma unroll
    for (int layer = 0; layer < 3; ++layer) {
      main_kv_stage.weight[layer] = reinterpret_cast<const cutlass::float_e4m3_t*>(
          weight_arena + weight_offsets[generated::kMainKvWeightSlots[layer]]);
      main_kv_stage.weight_scales[layer] =
          weight_arena + weight_offsets[generated::kMainKvScaleSlots[layer]];
      main_kv_stage.norm_weight[layer] = reinterpret_cast<const float*>(
          weight_arena + weight_offsets[generated::kMainKvNormWeightSlots[layer]]);
    }
  }
  DSPARK_INIT_STAGE(Layer0AttnHcStage, layer0_attn_hc_stage){
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kHiddenStreamsOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kHcMixOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kHcPreOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kHcPostOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kHcCombOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kNormalizedHiddenOffset),
      reinterpret_cast<cutlass::float_e4m3_t*>(
          raw_workspace + generated::kAttnInputQuantizedOffset),
      raw_workspace + generated::kAttnInputScaleOffset,
      execution_mask & kExecuteLayer0AttnHc
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0AttnHcFnSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttnHc
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0AttnHcBaseSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttnHc
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0AttnHcScaleSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttnHc
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0AttnNormWeightSlot])
          : nullptr,
      weight_arena,
      weight_offsets,
      draft_layer_mask,
      static_cast<bool>(execution_mask & kExecuteLayer0AttnHc),
  };
  DSPARK_INIT_STAGE(Layer0AttnProjectionStage, layer0_attn_projection_stage){
      reinterpret_cast<const cutlass::float_e4m3_t*>(
          raw_workspace + generated::kAttnInputQuantizedOffset),
      raw_workspace + generated::kAttnInputScaleOffset,
      reinterpret_cast<float*>(raw_workspace + generated::kQaPartialOffset),
      reinterpret_cast<__nv_bfloat16*>(raw_workspace + generated::kQloraOffset),
      reinterpret_cast<cutlass::float_e4m3_t*>(
      raw_workspace + generated::kQloraQuantizedOffset),
      raw_workspace + generated::kQloraScaleOffset,
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kQueryProjectionOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kQueryInverseRmsOffset),
      reinterpret_cast<__nv_bfloat16*>(raw_workspace + generated::kQueriesOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kDraftKvPartialOffset),
      reinterpret_cast<__nv_bfloat16*>(raw_workspace + generated::kDraftKvOffset),
      execution_mask & kExecuteLayer0AttnProjections
          ? reinterpret_cast<const cutlass::float_e4m3_t*>(
                weight_arena + weight_offsets[generated::kLayer0WqaWeightSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttnProjections
          ? weight_arena + weight_offsets[generated::kLayer0WqaScaleSlot]
          : nullptr,
      execution_mask & kExecuteLayer0AttnProjections
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0QaNormWeightSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttnProjections
          ? reinterpret_cast<const cutlass::float_e4m3_t*>(
                weight_arena + weight_offsets[generated::kLayer0WqbWeightSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttnProjections
          ? weight_arena + weight_offsets[generated::kLayer0WqbScaleSlot]
          : nullptr,
      execution_mask & kExecuteLayer0AttnProjections
          ? reinterpret_cast<const cutlass::float_e4m3_t*>(
                weight_arena + weight_offsets[generated::kLayer0WkvWeightSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttnProjections
          ? weight_arena + weight_offsets[generated::kLayer0WkvScaleSlot]
          : nullptr,
      execution_mask & kExecuteLayer0AttnProjections
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0KvNormWeightSlot])
          : nullptr,
      rope_cos_sin,
      weight_arena,
      weight_offsets,
      draft_layer_mask,
      static_cast<bool>(execution_mask & kExecuteLayer0AttnProjections),
  };
  DSPARK_INIT_STAGE(Layer0SparseAttentionStage, layer0_sparse_attention_stage){
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kQueriesOffset),
      kv_cache,
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kDraftKvOffset),
      reinterpret_cast<float*>(
          raw_workspace + generated::kAttentionAccumulatorOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kAttentionRawOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kAttentionValuesOffset),
      execution_mask & kExecuteLayer0SparseAttention
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0AttnSinkWeightSlot])
          : nullptr,
      rope_cos_sin,
      start_pos,
      weight_arena,
      weight_offsets,
      draft_layer_mask,
      static_cast<bool>(execution_mask & kExecuteLayer0SparseAttention),
  };
  DSPARK_INIT_STAGE(Layer0AttentionOutputStage, layer0_attention_output_stage){
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kAttentionValuesOffset),
      reinterpret_cast<__nv_bfloat16*>(raw_workspace + generated::kOutputLoraOffset),
      reinterpret_cast<cutlass::float_e4m3_t*>(
          raw_workspace + generated::kOutputLoraQuantizedOffset),
      raw_workspace + generated::kOutputLoraScaleOffset,
      reinterpret_cast<float*>(
          raw_workspace + generated::kAttentionOutputPartialOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kAttentionOutputOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kHiddenStreamsOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kAttentionHiddenStreamsOffset),
      reinterpret_cast<const float*>(raw_workspace + generated::kHcPostOffset),
      reinterpret_cast<const float*>(raw_workspace + generated::kHcCombOffset),
      execution_mask & kExecuteLayer0AttentionOutput
          ? reinterpret_cast<const __nv_bfloat16*>(
                weight_arena + weight_offsets[generated::kLayer0WoaWeightSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttentionOutput
          ? reinterpret_cast<const cutlass::float_e4m3_t*>(
                weight_arena + weight_offsets[generated::kLayer0WobWeightSlot])
          : nullptr,
      execution_mask & kExecuteLayer0AttentionOutput
          ? weight_arena + weight_offsets[generated::kLayer0WobScaleSlot]
          : nullptr,
      weight_arena,
      weight_offsets,
      draft_layer_mask,
      static_cast<bool>(execution_mask & kExecuteLayer0AttentionOutput),
  };
  DSPARK_INIT_STAGE(Layer0AttnHcStage, layer0_ffn_hc_stage){
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kAttentionHiddenStreamsOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kFfnHcMixOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kFfnHcPreOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kFfnHcPostOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kFfnHcCombOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kFfnNormalizedHiddenOffset),
      reinterpret_cast<cutlass::float_e4m3_t*>(
          raw_workspace + generated::kFfnInputQuantizedOffset),
      raw_workspace + generated::kFfnInputScaleOffset,
      execution_mask & kExecuteLayer0FfnRouter
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0FfnHcFnSlot])
          : nullptr,
      execution_mask & kExecuteLayer0FfnRouter
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0FfnHcBaseSlot])
          : nullptr,
      execution_mask & kExecuteLayer0FfnRouter
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0FfnHcScaleSlot])
          : nullptr,
      execution_mask & kExecuteLayer0FfnRouter
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0FfnNormWeightSlot])
          : nullptr,
      weight_arena,
      weight_offsets,
      draft_layer_mask,
      static_cast<bool>(execution_mask & kExecuteLayer0FfnRouter),
  };
  DSPARK_INIT_STAGE(Layer0RouterStage, layer0_router_stage){
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kFfnNormalizedHiddenOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kRouterScoreOffset),
      reinterpret_cast<int32_t*>(raw_workspace + generated::kRouterIndexOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kRouterWeightOffset),
      execution_mask & kExecuteLayer0FfnRouter
          ? reinterpret_cast<const __nv_bfloat16*>(
                weight_arena + weight_offsets[generated::kLayer0RouterWeightSlot])
          : nullptr,
      execution_mask & kExecuteLayer0FfnRouter
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLayer0RouterBiasSlot])
          : nullptr,
      weight_arena,
      weight_offsets,
      draft_layer_mask,
      static_cast<bool>(execution_mask & kExecuteLayer0FfnRouter),
  };
  DSPARK_INIT_STAGE(Layer0ExpertsStage, layer0_experts_stage){
      reinterpret_cast<const cutlass::float_e4m3_t*>(
          raw_workspace + generated::kFfnInputQuantizedOffset),
      raw_workspace + generated::kFfnInputScaleOffset,
      reinterpret_cast<const int32_t*>(
          raw_workspace + generated::kRouterIndexOffset),
      reinterpret_cast<const float*>(
          raw_workspace + generated::kRouterWeightOffset),
      reinterpret_cast<__nv_bfloat16*>(raw_workspace + generated::kRoutedW13Offset),
      reinterpret_cast<__nv_bfloat16*>(raw_workspace + generated::kSharedW13Offset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kRoutedSwiGluOffset),
      reinterpret_cast<cutlass::float_e4m3_t*>(
          raw_workspace + generated::kRoutedSwiGluQuantizedOffset),
      raw_workspace + generated::kRoutedSwiGluScaleOffset,
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kSharedSwiGluOffset),
      reinterpret_cast<cutlass::float_e4m3_t*>(
          raw_workspace + generated::kSharedSwiGluQuantizedOffset),
      raw_workspace + generated::kSharedSwiGluScaleOffset,
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kRoutedOutputPartialOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kRoutedOutputOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kSharedOutputOffset),
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kAttentionHiddenStreamsOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kHiddenStreamsOffset),
      reinterpret_cast<const float*>(raw_workspace + generated::kFfnHcPostOffset),
      reinterpret_cast<const float*>(raw_workspace + generated::kFfnHcCombOffset),
      weight_arena,
      weight_offsets,
      -1,
      draft_layer_mask,
      static_cast<bool>(execution_mask & kExecuteLayer0Experts),
  };
#ifdef DSPARK_ROUTED_GROUP_READY
  layer0_router_stage.group_ready = reinterpret_cast<uint32_t*>(raw_workspace + generated::kRoutedGroupReadyOffset);
  layer0_experts_stage.group_ready = layer0_router_stage.group_ready;
#endif
#ifdef DSPARK_V4_TMA_WEIGHTS
  layer0_experts_stage.w13_weight_tma = w13_weight_tma;
  layer0_experts_stage.w2_weight_tma = w2_weight_tma;
#endif
  const bool head_enabled =
      (execution_mask & kExecuteHead) && draft_layer_mask == 0x7;
  DSPARK_INIT_STAGE(HeadStage, head_stage){
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kHiddenStreamsOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kHeadHiddenOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kHeadNormalizedOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kBaseLogitsOffset),
      head_enabled
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kHcHeadFnWeightSlot])
          : nullptr,
      head_enabled
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kHcHeadBaseWeightSlot])
          : nullptr,
      head_enabled
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kHcHeadScaleWeightSlot])
          : nullptr,
      head_enabled
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kFinalNormWeightSlot])
          : nullptr,
      head_enabled
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kLmHeadWeightSlot])
          : nullptr,
      head_enabled,
  };
  const bool tail_enabled =
      (execution_mask & kExecuteTail) && draft_layer_mask == 0x7
      && output_ids != nullptr && corrected_logits != nullptr
      && probabilities != nullptr && confidence_logits != nullptr
      && calibrated_confidences != nullptr
      && scheduled_prefix_lengths != nullptr && scheduler_read_mask != nullptr
      && scheduler_summary != nullptr && uniforms != nullptr
      && sts_temperatures != nullptr && steps_per_second != nullptr;
  DSPARK_INIT_STAGE(TailStage, tail_stage){
      anchor,
      reinterpret_cast<const __nv_bfloat16*>(
          raw_workspace + generated::kHeadHiddenOffset),
      reinterpret_cast<const float*>(
          raw_workspace + generated::kBaseLogitsOffset),
      reinterpret_cast<__nv_bfloat16*>(
          raw_workspace + generated::kMarkovEmbeddingOffset),
      reinterpret_cast<float*>(
          raw_workspace + generated::kMarkovLogitsOffset),
      reinterpret_cast<float*>(
          raw_workspace + generated::kSoftmaxPartialMaxOffset),
      reinterpret_cast<float*>(
          raw_workspace + generated::kSoftmaxPartialSumOffset),
      reinterpret_cast<float*>(raw_workspace + generated::kSampleScanOffset),
      output_ids,
      corrected_logits,
      probabilities,
      confidence_logits,
      calibrated_confidences,
      scheduled_prefix_lengths,
      scheduler_read_mask,
      scheduler_summary,
      uniforms,
      sts_temperatures,
      steps_per_second,
      tail_enabled
          ? reinterpret_cast<const __nv_bfloat16*>(
                weight_arena + weight_offsets[generated::kMarkovW1WeightSlot])
          : nullptr,
      tail_enabled
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kMarkovW2WeightSlot])
          : nullptr,
      tail_enabled
          ? reinterpret_cast<const float*>(
                weight_arena + weight_offsets[generated::kConfidenceWeightSlot])
          : nullptr,
      sampling_temperature,
      calibration_enabled,
      prefix_enabled,
      tail_enabled,
  };
  // Relaxed-numerics slots live past the 4,724 frozen offsets; -1 sentinels
  // keep each body independently switchable. Contract mode never sees them.
  if (extra_offset_slots >= 2) {
    if (head_enabled && weight_offsets[generated::kWeightOffsetSlots] >= 0) {
      head_stage.lm_head_bf16 = reinterpret_cast<const __nv_bfloat16*>(
          weight_arena + weight_offsets[generated::kWeightOffsetSlots]);
    }
    if (tail_enabled
        && weight_offsets[generated::kWeightOffsetSlots + 1] >= 0) {
      tail_stage.markov_w2_bf16 = reinterpret_cast<const __nv_bfloat16*>(
          weight_arena + weight_offsets[generated::kWeightOffsetSlots + 1]);
    }
  }
#ifdef DSPARK_V4_SHARED_STAGE_CONTEXT
  }
  __syncthreads();
#endif
#undef DSPARK_INIT_STAGE
  extern __shared__ float dynamic_shared[];
  __shared__ ClaimedTask task;
  __shared__ uint64_t task_start;
  __shared__ uint64_t task_end;
  __shared__ uint64_t idle_start;
  __shared__ int idle_prefetched;
  __shared__ int should_exit;
  __shared__ int task_body_enabled;
  const int worker = static_cast<int>(blockIdx.x);

#ifdef DSPARK_V4_STATIC_TP2_TAIL
  bool tp2_serving_ready = true;
  if (tp2_context.enabled) {
    const bool tp2_ready = dspark_v4_tp2::rank0_begin(
        tp2_context, anchor, worker, tp2_start_pos_valid);
    if (tp2_context.serving) {
      tp2_serving_ready = tp2_ready;
    }
  }
#endif

  if (threadIdx.x == 0) {
    *worker_word(workspace, worker, 0) = 0;
    *queue_word(workspace, worker, 0) = 0;
    *queue_word(workspace, worker, 1) = 0;
#ifndef DSPARK_V4_RELAXED_DAG
    *worker_word(workspace, worker, 2) = 0;
#endif
    *worker_word(workspace, worker, 3) = 0;
    *worker_word(workspace, worker, 7) = worker == 0 ? pack_epoch(epoch, 0) : 0;
    if constexpr (kTraceEnabled) {
      trace_counts[worker] = 0;
    }
    __threadfence();
    *worker_word(workspace, worker, 1) = epoch;
    idle_start = 0;
    idle_prefetched = 0;
    if (worker == 0) {
      g_idle_prefetch_cursor = 0;
#ifdef DSPARK_V4_RELAXED_DAG
      g_ready_hint = 0;
#endif
      launch_audit[0] = kAuditMagic;
      launch_audit[1] = epoch;
      launch_audit[2] = globaltimer_ns();
      launch_audit[3] = 0;
      launch_audit[4] = 0;
      launch_audit[5] = 0;
      launch_audit[6] = 0;
      launch_audit[7] = 0;
    }
  }
  __syncthreads();

#ifdef DSPARK_V4_STATIC_TP2_TAIL
  if (!tp2_serving_ready) {
    const int lane = worker * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int index = lane; index < kDraftBlock + 1; index += stride) {
      output_ids[index] = 0;
    }
    for (int index = lane; index < kDraftBlock * dspark_v4_tp2::kVocab;
         index += stride) {
      corrected_logits[index] = 0.0f;
      probabilities[index] = 0.0f;
    }
    for (int index = lane; index < kDraftBlock; index += stride) {
      confidence_logits[index] = 0.0f;
      calibrated_confidences[index] = 0.0f;
      scheduler_read_mask[index] = 0;
    }
    for (int index = lane; index < 4; index += stride) {
      scheduler_summary[index] = 0.0f;
    }
    if (lane == 0) {
      scheduled_prefix_lengths[0] = 0;
      launch_audit[3] = globaltimer_ns();
      launch_audit[4] = 0;
      __threadfence_system();
      atomicExch(
          reinterpret_cast<unsigned long long*>(worker_word(workspace, 0, 7)),
          pack_epoch(epoch, 1));
    }
    return;
  }
#endif

#ifndef DSPARK_V4_STATIC_QUEUES
  // One load per CTA; the policy is immutable for the life of a proposal.
  const V4SchedPolicy policy = g_sched_policy;
  __shared__ unsigned long long sched_state[kSchedStateWords];
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  __shared__ unsigned long long probe[kProbeWorkerSlots];
#endif
  if (threadIdx.x == 0) {
#ifndef DSPARK_V4_STATIC_QUEUES
    sched_state[kSchedSeenSeq] = 0ULL;
    sched_state[kSchedRingFloor] = 0ULL;
    sched_state[kSchedLocalMaybe] = 1ULL;
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
    for (int slot = 0; slot < kProbeWorkerSlots; ++slot) {
      probe[slot] = 0ULL;
    }
#endif
  }
  __syncthreads();

#ifdef DSPARK_V4_DIRECT_DEPENDENCIES
  static_assert(generated::kPhaseCount <= generated::kWorkers);
  if (threadIdx.x == 0 && worker < generated::kPhaseCount) {
    store_relaxed_u64(&workspace.completed_tiles[worker], pack_epoch(epoch, 0));
    store_relaxed_u64(&workspace.next_tile[worker], pack_epoch(epoch, 0));
    store_release_u64(&workspace.publish_epoch[worker], pack_epoch(epoch, 1));
  }
#elif defined(DSPARK_V4_STATIC_QUEUES)
  if (threadIdx.x == 0 && worker < generated::kRootCount) {
    const int root = generated::kRoots[worker];
    initialize_epoch_counter(&workspace.dependency_arrivals[root], epoch);
    publish_phase(
        workspace,
        root,
        worker,
        epoch,
        false,
        kSchedDefaultFlags
    );
  }
#else
  if (threadIdx.x == 0 && worker < generated::kRootCount) {
    const int root = generated::kRoots[worker];
    initialize_epoch_counter(&workspace.dependency_arrivals[root], epoch);
    publish_phase(
        workspace,
        root,
        worker,
        epoch,
        false,
        policy.flags
    );
  }
#endif
  __syncthreads();

#ifdef DSPARK_V4_STATIC_QUEUES
  static_assert(kBatch == 1, "static queues are specialized for batch 1");
#ifdef DSPARK_V4_DIRECT_DEPENDENCIES
  int direct_ready_cache = -1;
#endif
  int static_task_index = generated::kStaticWorkerTaskOffsets[worker];
  const int static_task_end = generated::kStaticWorkerTaskOffsets[worker + 1];
#ifdef DSPARK_V4_QUEUE_LOOKAHEAD
  uint32_t static_retired = 0;
#endif
  const bool compact_masked_schedule =
      execution_mask != kExecuteAll || draft_layer_mask != 0x7;
#else
  uint32_t round = 0;
  uint32_t idle_rounds = 0;
#endif
#if DSPARK_ROUTED_REUSE_TMEM
  dspark_tmem::initialize_routed_reservation();
#endif
#ifdef DSPARK_V4_PERSISTENT_TMEM
  dspark_tmem::reserve();
#endif
  while (true) {
    if (threadIdx.x == 0) {
      uint64_t controller_start = 0;
      if constexpr (kTraceEnabled) {
        controller_start = globaltimer_ns();
      }
#ifdef DSPARK_V4_STATIC_QUEUES
      task = {-1, 0, 0, 0, false};
      task_body_enabled = 0;
#ifdef DSPARK_V4_QUEUE_LOOKAHEAD
      task = claim_ready_static_task(workspace, epoch, static_task_index,
          static_task_end, static_retired, direct_ready_cache);
      task_body_enabled = task.phase >= 0;
#elif defined(DSPARK_V4_ROUTED_DYNAMIC_CLAIMS)
      // Keep AOT ordering between phases, but let any participating CTA drain
      // the routed phase's claim counter. Runtime expert grouping makes these
      // claims bimodal: duplicates are no-ops while leaders execute full GEMMs.
      while (static_task_index < static_task_end) {
        const int slot = static_task_index;
        const int phase = generated::kStaticTaskPhases[slot];
        const uint32_t static_claim = generated::kStaticTaskClaims[slot];
        const bool body_enabled = !compact_masked_schedule
            || phase_body_enabled(phase, execution_mask, draft_layer_mask);
        if (!body_enabled && static_claim != 0) {
          ++static_task_index;
          continue;
        }
        if (body_enabled && dynamically_claimed_routed_phase(phase)) {
#ifdef DSPARK_V4_DIRECT_DEPENDENCIES
          const uint64_t published = wait_direct_dependencies(
              workspace, phase, epoch, direct_ready_cache);
#elif (DSPARK_SCHEDULER_HANDOFF & 1)
          const uint64_t published = wait_published_phase<true>(
              &workspace.publish_epoch[phase], epoch);
#else
          uint64_t published;
          do {
            published = load_acquire_u64(&workspace.publish_epoch[phase]);
            if (counter_epoch(published) == epoch
                && counter_value(published) >= 1) {
              break;
            }
            __nanosleep(32);
          } while (true);
#endif
          // Another CTA can finish the entire phase before this worker reaches
          // its AOT slot. Epoch state 2 is therefore ready-and-already-finished.
          if (counter_value(published) == 2) {
            ++static_task_index;
            continue;
          }
#if defined(DSPARK_ROUTED_CLAIM_CHUNK)
          // AOT slots decide participating workers; actual routed ranges and
          // completion credits follow this GPU claim counter's granularity.
#ifdef DSPARK_ROUTED_W2_WORKERS
          // Exactly128 W2 stream tickets, four physical credits each.
          const uint32_t chunk = (phase == generated::kRoutedW2Phases[0] || phase == generated::kRoutedW2Phases[1] || phase == generated::kRoutedW2Phases[2]) ? 4 : DSPARK_ROUTED_CLAIM_CHUNK;
#else
          const uint32_t chunk = DSPARK_ROUTED_CLAIM_CHUNK;
#endif
#else
          const uint32_t chunk = generated::kClaimChunks[phase];
#endif
          const uint64_t claimed = atomicAdd(
              reinterpret_cast<unsigned long long*>(&workspace.next_tile[phase]),
              static_cast<unsigned long long>(chunk));
          const uint32_t begin = counter_value(claimed);
          if (counter_epoch(claimed) != epoch) {
            __trap();
          }
          if (begin >= generated::kWorkUnits[phase]) {
            ++static_task_index;
            continue;
          }
          const uint32_t end = begin + chunk < generated::kWorkUnits[phase]
              ? begin + chunk : generated::kWorkUnits[phase];
          task = {phase, begin, end, begin / chunk, false};
          task_body_enabled = 1;
          // Retain this AOT slot until all claims have been taken. A fast CTA
          // may claim repeatedly; each actual item is still completed once.
          break;
        }
        ++static_task_index;
#ifdef DSPARK_V4_DIRECT_DEPENDENCIES
        wait_direct_dependencies(workspace, phase, epoch, direct_ready_cache);
#elif (DSPARK_SCHEDULER_HANDOFF & 1)
        wait_published_phase<false>(&workspace.publish_epoch[phase], epoch);
#else
        const uint64_t ready = pack_epoch(epoch, 1);
        while (load_acquire_u64(&workspace.publish_epoch[phase]) != ready) {
          __nanosleep(32);
        }
#endif
        const uint32_t chunk = generated::kClaimChunks[phase];
        const uint32_t begin = static_claim * chunk;
        const uint32_t end = begin + chunk < generated::kWorkUnits[phase]
            ? begin + chunk : generated::kWorkUnits[phase];
        task = body_enabled
            ? ClaimedTask{phase, begin, end, static_claim, false}
            : ClaimedTask{phase, 0, generated::kWorkUnits[phase], 0, false};
        task_body_enabled = body_enabled;
        break;
      }
#else
      if (!compact_masked_schedule && static_task_index < static_task_end) {
        const int slot = static_task_index;
        const int phase = generated::kStaticTaskPhases[slot];
        const uint32_t claim = generated::kStaticTaskClaims[slot];
        ++static_task_index;
#if (DSPARK_SCHEDULER_HANDOFF & 1)
        wait_published_phase<false>(&workspace.publish_epoch[phase], epoch);
#else
        const uint64_t ready = pack_epoch(epoch, 1);
        while (load_acquire_u64(&workspace.publish_epoch[phase]) != ready) {
          __nanosleep(32);
        }
#endif
        const uint32_t chunk = generated::kClaimChunks[phase];
        const uint32_t begin = claim * chunk;
        const uint32_t end = begin + chunk < generated::kWorkUnits[phase]
            ? begin + chunk
            : generated::kWorkUnits[phase];
        task = {phase, begin, end, claim, false};
        task_body_enabled = 1;
      } else {
        while (static_task_index < static_task_end) {
          const int slot = static_task_index;
          const int phase = generated::kStaticTaskPhases[slot];
          const uint32_t claim = generated::kStaticTaskClaims[slot];
          ++static_task_index;
          const bool body_enabled =
              phase_body_enabled(phase, execution_mask, draft_layer_mask);
          // An omitted phase still participates in the original DAG, but one
          // claim can retire all of its logical work. Every other no-op claim
          // is skipped before the block-wide barriers. The all-enabled path
          // above retains the frozen task-acquisition sequence.
          if (!body_enabled && claim != 0) {
            continue;
          }
#if (DSPARK_SCHEDULER_HANDOFF & 1)
          wait_published_phase<false>(&workspace.publish_epoch[phase], epoch);
#else
          const uint64_t ready = pack_epoch(epoch, 1);
          while (load_acquire_u64(&workspace.publish_epoch[phase]) != ready) {
            __nanosleep(32);
          }
#endif
          if (body_enabled) {
            const uint32_t chunk = generated::kClaimChunks[phase];
            const uint32_t begin = claim * chunk;
            const uint32_t end = begin + chunk < generated::kWorkUnits[phase]
                ? begin + chunk
                : generated::kWorkUnits[phase];
            task = {phase, begin, end, claim, false};
            task_body_enabled = 1;
          } else {
            task = {phase, 0, generated::kWorkUnits[phase], 0, false};
          }
          break;
        }
      }
#endif
#else
      task = claim_task(
          workspace,
          worker,
          epoch,
          round++,
          policy,
          sched_state
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
          ,
          probe
#endif
      );
      task_body_enabled = 1;
#endif
      uint64_t controller_end = 0;
      if constexpr (kTraceEnabled) {
        controller_end = globaltimer_ns();
      }
      if (task.phase >= 0) {
        if constexpr (kTraceEnabled) {
          if (idle_start != 0) {
            if (idle_prefetched != 0) {
              // The idle window streamed weight-arena L2 prefetches: record
              // the TMA role's span with the dedicated Prefetch segment (the
              // other roles keep their RoleWait accounting). Prefetch is not
              // Useful: occupancy renders it as its own band, never as
              // painted-over idle.
              record_trace(
                  trace,
                  trace_counts,
                  worker,
                  generated::kPhaseCount - 1,
                  TraceRole::Tma,
                  TraceSegment::Prefetch,
                  idle_start,
                  controller_start,
                  0,
                  task.ticket,
                  kTraceNone);
            }
            record_trace(
                trace,
                trace_counts,
                worker,
                task.phase,
                TraceRole::Controller,
                TraceSegment::QueueEmptyWait,
                idle_start,
                controller_start,
                0,
                task.ticket,
                kTraceNone);
            for (int role = 1; role < 4; ++role) {
              if (role == static_cast<int>(TraceRole::Tma)
                  && idle_prefetched != 0) {
                continue;
              }
              record_trace(
                  trace,
                  trace_counts,
                  worker,
                  task.phase,
                  static_cast<TraceRole>(role),
                  TraceSegment::RoleWait,
                  idle_start,
                  controller_start,
                  0,
                  task.ticket,
                  kTraceNone);
            }
            idle_prefetched = 0;
            idle_start = 0;
          }
          record_trace(
              trace,
              trace_counts,
              worker,
              task.phase,
              TraceRole::Controller,
              TraceSegment::Controller,
              controller_start,
              controller_end,
              task.begin,
              task.ticket,
              task.stolen ? kTraceStolen : kTraceNone);
          task_start = controller_end;
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
          // Device-side first body start for this phase. The host derives the
          // same quantity from the trace ring, but only from USEFUL segments;
          // comparing the two is what separates real wake latency from an
          // artifact of which segments the ring happens to carry.
          atomicMin(
              &g_probe_phase[task.phase * kProbePhaseSlots + kProbeUsefulStartNs],
              static_cast<unsigned long long>(controller_end));
#endif
        }
      } else if constexpr (kTraceEnabled) {
        if (idle_start == 0) {
          idle_start = controller_start;
        }
      }
    }
    __syncthreads();

    if (task.phase < 0) {
#ifdef DSPARK_V4_STATIC_QUEUES
      // This CTA has retired its complete AOT instruction stream. Other CTAs
      // do not need it for a grid barrier and may finish independently.
      break;
#else
      if (threadIdx.x == 0) {
        should_exit = proposal_done(workspace, epoch) ? 1 : 0;
        if constexpr (kTraceEnabled) {
          if (should_exit && idle_start != 0) {
            const uint64_t now = globaltimer_ns();
            record_trace(
                trace,
                trace_counts,
                worker,
                generated::kPhaseCount - 1,
                TraceRole::Controller,
                TraceSegment::QueueEmptyWait,
                idle_start,
                now,
                0,
                0,
                kTraceNone);
          }
        }
      }
      __syncthreads();
      if (should_exit) {
        break;
      }
      // With the publish odometer in play an idle round costs one relaxed
      // load instead of a 33 us sweep, so the idle-window L2 prefetch fires
      // orders of magnitude more often than it did when it was designed.
      // prefetch_period throttles it (1 == every idle round, the historical
      // behaviour; 0 == off).
      if (policy.prefetch_period != 0u
          && (idle_rounds % policy.prefetch_period) == 0u) {
        __shared__ unsigned long long prefetch_base;
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
        const uint64_t prefetch_started =
            threadIdx.x == 0 ? globaltimer_ns() : 0ULL;
#endif
        if (threadIdx.x == 0) {
          prefetch_base =
              g_idle_prefetch_cursor >= generated::kWeightArenaBytes
                  ? generated::kWeightArenaBytes
                  : atomicAdd(&g_idle_prefetch_cursor, kIdlePrefetchBytes);
          if constexpr (kTraceEnabled) {
            if (prefetch_base < generated::kWeightArenaBytes) {
              idle_prefetched = 1;
            }
          }
        }
        __syncthreads();
        if (prefetch_base < generated::kWeightArenaBytes) {
          const unsigned long long limit =
              prefetch_base + kIdlePrefetchBytes
                      < generated::kWeightArenaBytes
                  ? prefetch_base + kIdlePrefetchBytes
                  : generated::kWeightArenaBytes;
          for (unsigned long long offset =
                   prefetch_base + threadIdx.x * 128ULL;
               offset < limit;
               offset += static_cast<unsigned long long>(blockDim.x) * 128ULL) {
            const void* address = weight_arena + offset;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(address));
          }
        }
        __syncthreads();
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
        if (threadIdx.x == 0) {
          probe[kProbeNsPrefetch] += globaltimer_ns() - prefetch_started;
          probe[kProbeCountPrefetch] += 1ULL;
        }
#endif
      }
      // Citizenship under co-residency: when no work is claimable, back off
      // with a parked-warp sleep instead of hammering the phase counters and
      // the prefetch stream. In serving, the verify graph's blocks co-reside
      // on these SMs and every idle rescan/prefetch steals execution slots
      // and L2/atomic bandwidth from them (measured as the bolted serving
      // cadence collapse, 2026-08-08). The backoff is now a POLICY: the
      // compiled default (512/4096 ns after 2 idle rounds) is exactly the
      // serving behaviour, and a standalone lane may select a shorter or
      // zero backoff via set_scheduler_policy / DSPARK_V4_IDLE_SLEEP_NS.
      if (++idle_rounds > policy.idle_sleep_after) {
        const unsigned int nap = idle_rounds > 8 ? policy.idle_sleep_long_ns
                                                 : policy.idle_sleep_ns;
        if (nap != 0u) {
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
          const uint64_t nap_started =
              threadIdx.x == 0 ? globaltimer_ns() : 0ULL;
#endif
          __nanosleep(nap);
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
          if (threadIdx.x == 0) {
            probe[kProbeNsSleep] += globaltimer_ns() - nap_started;
            probe[kProbeCountSleep] += 1ULL;
          }
#endif
        }
      }
      continue;
#endif
    }

#ifndef DSPARK_V4_STATIC_QUEUES
    idle_rounds = 0;
#endif
#ifdef DSPARK_V4_STATIC_QUEUES
    // The AOT stream already knows the exact owner of every phase. Dispatch
    // once instead of invoking all twelve phase families and making eleven of
    // them reject the same task. The generated table is checked for complete
    // ownership, and this changes neither task order nor any phase body.
    if (task_body_enabled) switch (generated::kExecutionGroups[task.phase]) {
      case generated::kExecutionMain:
        execute_main_phase(task, main_stage, dynamic_shared);
        break;
      case generated::kExecutionEmbedding:
        execute_embedding_phase(task, embedding_stage);
        break;
      case generated::kExecutionMainKv:
        execute_main_kv_phase(task, main_kv_stage, dynamic_shared);
        break;
      case generated::kExecutionAttnHc:
        execute_layer0_attn_hc_phase(task, layer0_attn_hc_stage, dynamic_shared);
        break;
      case generated::kExecutionAttnProjection:
        execute_layer0_attn_projection_phase(
            task,
            layer0_attn_projection_stage,
            dynamic_shared);
        break;
      case generated::kExecutionSparseAttention:
        execute_layer0_sparse_attention_phase(
            task,
            layer0_sparse_attention_stage,
            dynamic_shared);
        break;
      case generated::kExecutionAttentionOutput:
        execute_layer0_attention_output_phase(
            task,
            layer0_attention_output_stage,
            dynamic_shared);
        break;
      case generated::kExecutionFfnHc:
        execute_layer0_ffn_hc_phase(task, layer0_ffn_hc_stage, dynamic_shared);
        break;
      case generated::kExecutionRouter:
        execute_layer0_router_phase(task, layer0_router_stage, dynamic_shared);
        break;
      case generated::kExecutionExpert:
        execute_layer0_expert_phase(task, layer0_experts_stage, dynamic_shared);
        break;
      case generated::kExecutionHead:
        execute_head_phase(task, head_stage, dynamic_shared);
#ifdef DSPARK_CONFIDENCE_HEAD_PREFIX
        // Unique short LM owner: two tiles [1008,1010), all other owners seven.
        // Prefix publication is included in its unchanged LM completion credit.
        if (task.phase == generated::kLmRowPhases[0] && worker == 144
            && task.begin == 1008 && task.end == 1010) {
          execute_confidence_head_prefix(tail_stage);
        }
#endif
        break;
      case generated::kExecutionTail:
        execute_tail_phase(task, tail_stage, dynamic_shared);
        break;
    }
#else
    execute_main_phase(task, main_stage, dynamic_shared);
    execute_embedding_phase(task, embedding_stage);
    execute_main_kv_phase(task, main_kv_stage, dynamic_shared);
    execute_layer0_attn_hc_phase(task, layer0_attn_hc_stage, dynamic_shared);
    execute_layer0_attn_projection_phase(
        task,
        layer0_attn_projection_stage,
        dynamic_shared);
    execute_layer0_sparse_attention_phase(
        task,
        layer0_sparse_attention_stage,
        dynamic_shared);
    execute_layer0_attention_output_phase(
        task,
        layer0_attention_output_stage,
        dynamic_shared);
    execute_layer0_ffn_hc_phase(task, layer0_ffn_hc_stage, dynamic_shared);
    execute_layer0_router_phase(task, layer0_router_stage, dynamic_shared);
    execute_layer0_expert_phase(task, layer0_experts_stage, dynamic_shared);
    execute_head_phase(
        task,
        head_stage,
        dynamic_shared
#ifdef DSPARK_V4_STATIC_TP2_TAIL
        ,
        tp2_context
#endif
    );
    execute_tail_phase(
        task,
        tail_stage,
        dynamic_shared
#ifdef DSPARK_V4_STATIC_TP2_TAIL
        ,
        tp2_context
#endif
    );
#endif

#ifndef DSPARK_V4_RELAXED_DAG
    // The contract scheduler smoke uses this synthetic body to prove all
    // lanes executed. Numerical relaxed builds already ran real phase bodies,
    // so repeating the diagnostic would add millions of dead integer ops and
    // tens of thousands of global atomics to every serving proposal.
    uint64_t value =
        (static_cast<uint64_t>(task.phase + 1) << 48)
        ^ (static_cast<uint64_t>(task.begin + 1) << 24)
        ^ static_cast<uint64_t>(threadIdx.x + 1);
#pragma unroll
    for (int iteration = 0; iteration < 8; ++iteration) {
      value ^= value << 13;
      value ^= value >> 7;
      value ^= value << 17;
    }
    if ((threadIdx.x & 31) == 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(worker_word(workspace, worker, 2)),
          value);
    }
#endif
    __syncthreads();
    if (threadIdx.x == 0) {
      if constexpr (kTraceEnabled) {
        task_end = globaltimer_ns();
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
        // Device-side phase completion, recorded for EVERY phase including the
        // Controller-bodied ones the host trace cannot see as USEFUL. Note
        // this atomic, and the record_trace calls below, sit between this
        // stamp and finish_phase -- so the T1 term derived from it is an
        // upper bound that includes instrumentation the shipped kernel omits.
        atomicMax(
            &g_probe_phase[task.phase * kProbePhaseSlots + kProbeBodyEndNs],
            static_cast<unsigned long long>(task_end));
#endif
        const uint8_t role_mask = generated::kRoleMasks[task.phase];
        int64_t flags = task.stolen ? kTraceStolen : kTraceNone;
        if (task.end == generated::kWorkUnits[task.phase]) {
          flags |= kTraceLastTile;
        }
        if (role_mask == 1) {
          record_trace(
              trace,
              trace_counts,
              worker,
              task.phase,
              TraceRole::Controller,
              TraceSegment::Controller,
              task_start,
              task_end,
              task.begin,
              task.ticket,
              flags);
        } else {
          for (int role = 1; role < 4; ++role) {
            if (role_mask & (1 << role)) {
              record_trace(
                  trace,
                  trace_counts,
                  worker,
                  task.phase,
                  static_cast<TraceRole>(role),
                  TraceSegment::Useful,
                  task_start,
                  task_end,
                  task.begin,
                  task.ticket,
                  flags);
            }
          }
        }
      }
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
      const uint64_t finish_started = globaltimer_ns();
      probe[kProbeNsBody] += finish_started - task_start;
      probe[kProbeCountBody] += 1ULL;
#endif
      finish_phase(
          workspace,
          task,
          worker,
          epoch,
          launch_audit,
#ifdef DSPARK_V4_STATIC_QUEUES
          kSchedDefaultFlags
#else
          policy.flags
#endif
#ifdef DSPARK_V4_GREEDY_TAIL
          ,
          tail_stage
#ifdef DSPARK_V4_STATIC_TP2_TAIL
          ,
          tp2_context
#endif
#endif
      );
#ifdef DSPARK_V4_FINE_GRAINED_OVERLAP
      const uint16_t fused_consumer =
          generated::kStreamingConsumerForPhase[task.phase];
      if (fused_consumer != generated::kNoStreamingConsumer) {
        const int route_row = task.ticket / 8;
        if (routed_group_leader(layer0_experts_stage.indices, route_row)) {
          const int half = task.ticket % 8;
          const int expert = layer0_experts_stage.indices[route_row];
          int member_count = 0;
          constexpr int kRouteRows =
              dspark_w13::kDraftRows * dspark_w13::kActivatedExperts;
          for (int member_row = route_row;
               member_row < kRouteRows && member_count < 8;
               ++member_row) {
            if (layer0_experts_stage.indices[member_row] != expert) {
              continue;
            }
            const uint32_t consumer_claim = member_row * 8 + half;
            const ClaimedTask consumer_task{
                static_cast<int>(fused_consumer),
                consumer_claim,
                consumer_claim + 1,
                consumer_claim,
                false,
            };
            finish_phase(
                workspace,
                consumer_task,
                worker,
                epoch,
                launch_audit,
                kSchedDefaultFlags,
                tail_stage);
            ++member_count;
          }
        }
      }
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
      probe[kProbeNsFinish] += globaltimer_ns() - finish_started;
      probe[kProbeCountFinish] += 1ULL;
#endif
    }
#if !(DSPARK_SCHEDULER_HANDOFF & 2)
    __syncthreads();
#endif
    // With bit 2, the next dispatch barrier also closes this handoff. The
    // body-completion barrier above remains: all body reads/writes finish
    // before thread zero publishes completion or overwrites the shared task.
    // Only thread zero uses task between that barrier and the next dispatch.
  }

#ifdef DSPARK_V4_INTEGRATED_PROPOSAL_IO
  // Static queues retire at local exhaustion, which need not be terminal.
  // Only the publication CTA waits for the complete proposal; other workers
  // may retire normally. The barrier carries its acquire to the copy lanes.
  if (worker == 0) {
    if (threadIdx.x == 0) {
      while (load_acquire_u64(worker_word(workspace, 0, 7)) != pack_epoch(epoch, 1)) {
        __nanosleep(32);
      }
    }
    __syncthreads();
    dspark_proposal_io::publish(io, output_ids);
  }
#endif
#if DSPARK_ROUTED_REUSE_TMEM
  dspark_tmem::release_routed_reservation();
#endif
#ifdef DSPARK_V4_PERSISTENT_TMEM
  dspark_tmem::release();
#endif
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  if (threadIdx.x == 0) {
    for (int slot = 0; slot < kProbeWorkerSlots; ++slot) {
      g_probe_worker[worker * kProbeWorkerSlots + slot] = probe[slot];
    }
  }
#endif
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

#ifdef DSPARK_V4_TMA_WEIGHTS
// ---------------------------------------------------------------------------
// B2 host side: packed-subbyte tensor maps over the routed FP4 weight
// matrices, ported from the bench lane's prepare_expert_tma_maps (the B1
// parity oracle). Both maps view the SAME production arena: the device body
// resolves each expert matrix as a byte offset from the arena base, and the
// 4D map is a pure linear reinterpretation of the arena (64 B cells ->
// 512 B planes -> row_bytes rows), so any 64 B-aligned matrix offset is
// addressed exactly. Encoded once per bound arena and uploaded to device
// memory; proposal launches pass two stable device pointers and never touch
// a descriptor on the hot path.

constexpr uint64_t kTmaW13RowBytes = kMainHidden / 2;         // packed FP4
constexpr uint32_t kTmaW13Planes = 4;
constexpr uint64_t kTmaW2RowBytes = kExpertIntermediate / 2;  // packed FP4
constexpr uint32_t kTmaW2Planes = 2;

bool encode_expert_weight_tma(
    const torch::Tensor& arena,
    uint64_t row_bytes,
    uint32_t planes,
    CUtensorMap* descriptor) {
  if (!arena.is_cuda() || arena.scalar_type() != torch::kUInt8
      || !arena.is_contiguous() || arena.numel() <= 0) {
    return false;
  }
  const uintptr_t address =
      reinterpret_cast<uintptr_t>(arena.const_data_ptr<uint8_t>());
  if (address % 128 != 0 || row_bytes != 64ULL * 8ULL * planes
      || arena.numel() % 64 != 0) {
    return false;
  }
  // Exclude a partial final logical row rather than advertising bytes
  // beyond the allocation; the audited production routed matrices end well
  // before this floor boundary (validated per offset below).
  const uint64_t outer = static_cast<uint64_t>(arena.numel()) / row_bytes;
  if (outer < 128
      || outer
          > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
#ifdef DSPARK_ROUTED_BULK_WEIGHTS
  // Four adjacent 64-byte cells per outer index allow every 256-byte-aligned
  // arena entry to address a contiguous K512 stage in one tensor-map load.
  const cuuint64_t global_dims[3] = {
      128, 4, static_cast<uint64_t>(arena.numel()) / 256};
  const cuuint64_t global_strides[2] = {64, 256};
  const cuuint32_t box_dims[3] = {128, 4, 128};
  constexpr int rank = 3;
#else
  const cuuint64_t global_dims[4] = {128, 8, planes, outer};
  const cuuint64_t global_strides[3] = {64, 512, row_bytes};
  const cuuint32_t box_dims[4] = {128, 1, 1, 128};
  constexpr int rank = 4;
#endif
  const cuuint32_t element_strides[4] = {1, 1, 1, 1};
  *descriptor = {};
  const CUresult result = cuTensorMapEncodeTiled(
      descriptor,
      CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
      rank,
      const_cast<uint8_t*>(arena.const_data_ptr<uint8_t>()),
      global_dims,
      global_strides,
      box_dims,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS;
}

// Every routed W1/W3/W2 matrix must start 64 B-aligned (one tensor-map
// cell) and end inside the map's floor boundary. Debug lanes with synthetic
// arenas can fail this; they fall back to the retained cp.async body.
bool routed_expert_offsets_are_tma_addressable(
    const torch::Tensor& weight_offsets,
    int64_t arena_bytes) {
  constexpr int kLayers = static_cast<int>(
      sizeof(generated::kRoutedExpertWeightBases)
      / sizeof(generated::kRoutedExpertWeightBases[0]));
  uint16_t bases[kLayers] = {};
  if (cudaMemcpyFromSymbol(
          bases, generated::kRoutedExpertWeightBases, sizeof(bases))
      != cudaSuccess) {
    return false;
  }
  const torch::Tensor host_offsets =
      weight_offsets.to(torch::kCPU).contiguous();
  const int64_t* offsets = host_offsets.const_data_ptr<int64_t>();
  const int64_t slot_count = host_offsets.numel();
  constexpr int64_t kW13MatrixBytes =
      static_cast<int64_t>(kExpertIntermediate) * kTmaW13RowBytes;
  constexpr int64_t kW2MatrixBytes =
      static_cast<int64_t>(kMainHidden) * kTmaW2RowBytes;
  const int64_t w13_floor = arena_bytes
      / static_cast<int64_t>(kTmaW13RowBytes)
      * static_cast<int64_t>(kTmaW13RowBytes);
  const int64_t w2_floor = arena_bytes
      / static_cast<int64_t>(kTmaW2RowBytes)
      * static_cast<int64_t>(kTmaW2RowBytes);
  struct MatrixCheck {
    int slot;
    int64_t bytes;
    int64_t floor;
  };
  const MatrixCheck matrices[3] = {
      {generated::kExpertW1Slot, kW13MatrixBytes, w13_floor},
      {generated::kExpertW3Slot, kW13MatrixBytes, w13_floor},
      {generated::kExpertW2Slot, kW2MatrixBytes, w2_floor},
  };
  for (int layer = 0; layer < kLayers; ++layer) {
    for (int expert = 0; expert < kRoutedExperts; ++expert) {
      const int64_t table_base = static_cast<int64_t>(bases[layer])
          + static_cast<int64_t>(expert) * generated::kExpertWeightSlots;
      for (const MatrixCheck& matrix : matrices) {
        const int64_t slot = table_base + matrix.slot;
        if (slot < 0 || slot >= slot_count) {
          return false;
        }
        const int64_t offset = offsets[slot];
#ifdef DSPARK_ROUTED_BULK_WEIGHTS
        constexpr int64_t required_alignment = 256;
#else
        constexpr int64_t required_alignment = 64;
#endif
        if (offset < 0 || offset % required_alignment != 0
            || offset + matrix.bytes > matrix.floor) {
          return false;
        }
      }
    }
  }
  return true;
}

struct ExpertWeightTmaCache {
  const void* arena_base = nullptr;
  int64_t arena_numel = 0;
  const void* offsets_base = nullptr;
  int64_t offsets_numel = 0;
  int device = -1;
  bool prepared = false;
  const CUtensorMap* device_maps = nullptr;  // [0] = W13, [1] = W2
};

ExpertWeightTmaCache g_expert_weight_tma_cache;

// Device pointer to the {W13, W2} tensor maps for this arena, or nullptr
// when the arena/offset table cannot be addressed by the packed tensor-map
// geometry (production arenas always qualify). Cold path runs once per
// bound arena; the caller must hold the device guard.
const CUtensorMap* prepare_expert_weight_tma_maps(
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets) {
  ExpertWeightTmaCache& cache = g_expert_weight_tma_cache;
  const void* arena_base = weight_arena.const_data_ptr();
  const void* offsets_base = weight_offsets.const_data_ptr();
  if (cache.prepared && cache.arena_base == arena_base
      && cache.arena_numel == weight_arena.numel()
      && cache.offsets_base == offsets_base
      && cache.offsets_numel == weight_offsets.numel()
      && cache.device == weight_arena.get_device()) {
    return cache.device_maps;
  }
  cache = ExpertWeightTmaCache{};
  cache.arena_base = arena_base;
  cache.arena_numel = weight_arena.numel();
  cache.offsets_base = offsets_base;
  cache.offsets_numel = weight_offsets.numel();
  cache.device = weight_arena.get_device();
  cache.prepared = true;

  alignas(CUtensorMap) CUtensorMap host_maps[2] = {};
  bool eligible = encode_expert_weight_tma(
          weight_arena, kTmaW13RowBytes, kTmaW13Planes, &host_maps[0])
      && encode_expert_weight_tma(
          weight_arena, kTmaW2RowBytes, kTmaW2Planes, &host_maps[1])
      && routed_expert_offsets_are_tma_addressable(
          weight_offsets, weight_arena.numel());
  if (eligible) {
    void* device_maps = nullptr;
    if (cudaMalloc(&device_maps, sizeof(host_maps)) != cudaSuccess) {
      eligible = false;
    } else if (
        cudaMemcpy(
            device_maps, host_maps, sizeof(host_maps),
            cudaMemcpyHostToDevice)
        != cudaSuccess) {
      cudaFree(device_maps);
      eligible = false;
    } else {
      // Intentionally never freed: one 256-byte allocation per bound arena
      // keeps the descriptors valid for every later launch in the process.
      cache.device_maps = static_cast<const CUtensorMap*>(device_maps);
    }
  }
  std::printf(
      "dspark_v4_tma_weights status=%s arena_bytes=%lld\n",
      cache.device_maps != nullptr ? "enabled" : "fallback",
      static_cast<long long>(weight_arena.numel()));
  std::fflush(stdout);
  return cache.device_maps;
}
#endif  // DSPARK_V4_TMA_WEIGHTS

// ---------------------------------------------------------------------------
// Scheduler-policy host plumbing. The defaults below reproduce the retained
// serving-co-residency behaviour byte for byte; a lane opts into a different
// idle-discovery policy through the environment or set_scheduler_policy().
// ---------------------------------------------------------------------------

unsigned int env_uint(const char* name, unsigned int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(raw, &end, 10);
  if (end == raw) {
    return fallback;
  }
  return static_cast<unsigned int>(parsed);
}

V4SchedPolicy& host_scheduler_policy() {
  static V4SchedPolicy policy = [] {
    V4SchedPolicy value;
    value.idle_sleep_ns = env_uint("DSPARK_V4_IDLE_SLEEP_NS", 64u);
    value.idle_sleep_long_ns = env_uint("DSPARK_V4_IDLE_SLEEP_LONG_NS", 64u);
    value.idle_sleep_after = env_uint("DSPARK_V4_IDLE_SLEEP_AFTER", 2u);
    value.flags = env_uint("DSPARK_V4_SCHED_FLAGS", kSchedDefaultFlags);
    value.deep_period = env_uint("DSPARK_V4_DEEP_PERIOD", 128u);
    value.express_workers = env_uint("DSPARK_V4_EXPRESS_WORKERS", 0u);
    value.prefetch_period = env_uint("DSPARK_V4_PREFETCH_PERIOD", 0u);
    return value;
  }();
  return policy;
}

bool& scheduler_policy_dirty() {
  static bool dirty = true;
  return dirty;
}

void push_scheduler_policy(cudaStream_t stream) {
  if (!scheduler_policy_dirty()) {
    return;
  }
  const V4SchedPolicy policy = host_scheduler_policy();
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      g_sched_policy,
      &policy,
      sizeof(V4SchedPolicy),
      0,
      cudaMemcpyHostToDevice,
      stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  scheduler_policy_dirty() = false;
}

#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
void reset_scheduler_probes(cudaStream_t stream) {
  static std::vector<unsigned long long> seed;
  if (seed.empty()) {
    seed.assign(generated::kPhaseCount * kProbePhaseSlots, 0ULL);
    for (int phase = 0; phase < generated::kPhaseCount; ++phase) {
      seed[phase * kProbePhaseSlots + kProbeFirstClaimNs] = ~0ULL;
      // atomicMin target: must start saturated or every phase reports 0.
      seed[phase * kProbePhaseSlots + kProbeUsefulStartNs] = ~0ULL;
    }
  }
  void* worker_address = nullptr;
  C10_CUDA_CHECK(cudaGetSymbolAddress(&worker_address, g_probe_worker));
  C10_CUDA_CHECK(cudaMemsetAsync(
      worker_address,
      0,
      sizeof(unsigned long long) * generated::kWorkers * kProbeWorkerSlots,
      stream));
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      g_probe_phase,
      seed.data(),
      sizeof(unsigned long long) * seed.size(),
      0,
      cudaMemcpyHostToDevice,
      stream));
}
#endif

// ---------------------------------------------------------------------------
// R14 STAGE 1: THE CLUSTER LAUNCH PRICED ALONE.
//
// `DSPARK_V4_CLUSTER_DIM` (host environment, read per launch) switches the one
// scheduler launch between the shipped `<<<>>>` form and `cudaLaunchKernelEx`
// with a `cudaLaunchAttributeClusterDimension` of (N,1,1). NOTHING ELSE MOVES:
// the cubin is byte-identical, every phase body is what it is today, and the
// grid is still generated::kWorkers CTAs of generated::kThreads. That makes the
// delta between dim 1 and dim 2 the PRICE OF CLUSTERING and nothing else --
// which is exactly the go/no-go for a cta_group::2 routed MMA, because that
// change has to pay this before it can win anything.
//
// The env is read on every launch rather than cached so one process can sweep
// both points on one GPU (the box drifts 4-11% between sessions). getenv is
// tens of nanoseconds against a ~2 ms kernel and BOTH points pay it.
//
// The persistent scheduler is NOT cooperative -- it is a DAG scheduler over
// global counters with work stealing, so a CTA that is not resident cannot
// deadlock the grid, it only slows it down. The design assumes all 152 CTAs
// run concurrently on the 152 SMs, and a cluster must fit inside one GPC,
// so a cluster dim that does not divide the per-GPC SM count silently strands
// SMs. check_cluster_residency() refuses to launch in that case instead of
// reporting a slow number as if it were the cost of clustering.
// ---------------------------------------------------------------------------
int dspark_v4_cluster_dim() {
  const char* value = std::getenv("DSPARK_V4_CLUSTER_DIM");
  if (value == nullptr || value[0] == '\0') {
    return 1;
  }
  const int dim = std::atoi(value);
  return dim > 1 ? dim : 1;
}

void check_cluster_residency(int cluster_dim) {
  TORCH_CHECK(
      generated::kWorkers % cluster_dim == 0,
      "cluster dim ",
      cluster_dim,
      " does not divide the ",
      generated::kWorkers,
      "-CTA grid");
  cudaLaunchConfig_t probe = {};
  probe.gridDim = dim3(generated::kWorkers);
  probe.blockDim = dim3(generated::kThreads);
  probe.dynamicSmemBytes = kLaunchSharedBytes;
  cudaLaunchAttribute probe_attr = {};
  probe_attr.id = cudaLaunchAttributeClusterDimension;
  probe_attr.val.clusterDim.x = static_cast<unsigned>(cluster_dim);
  probe_attr.val.clusterDim.y = 1;
  probe_attr.val.clusterDim.z = 1;
  probe.attrs = &probe_attr;
  probe.numAttrs = 1;
  int max_clusters = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
      &max_clusters,
      reinterpret_cast<const void*>(dspark_v4_scheduler_kernel),
      &probe));
  const int needed = generated::kWorkers / cluster_dim;
  static int reported = 0;
  if (reported != cluster_dim) {
    reported = cluster_dim;
    std::fprintf(
        stderr,
        "dspark_v4_cluster dim=%d clusters_needed=%d max_active_clusters=%d "
        "resident_ctas=%d\n",
        cluster_dim,
        needed,
        max_clusters,
        max_clusters * cluster_dim);
    std::fflush(stderr);
  }
  const char* allow_partial = std::getenv("DSPARK_V4_CLUSTER_ALLOW_PARTIAL");
  if (allow_partial == nullptr || allow_partial[0] == '\0') {
    TORCH_CHECK(
        max_clusters >= needed,
        "cluster dim ",
        cluster_dim,
        " strands SMs: only ",
        max_clusters,
        " of the required ",
        needed,
        " clusters are co-resident (",
        max_clusters * cluster_dim,
        " of ",
        generated::kWorkers,
        " CTAs). Set DSPARK_V4_CLUSTER_ALLOW_PARTIAL=1 to measure anyway.");
  }
}

void launch_v4_scheduler(
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const __nv_bfloat16* main_hidden,
    __nv_bfloat16* main_output,
    const int32_t* anchor,
    __nv_bfloat16* embedding_output,
    const float* rope_cos_sin,
    __nv_bfloat16* kv_cache,
    int32_t* output_ids,
    float* corrected_logits,
    float* probabilities,
    float* confidence_logits,
    float* calibrated_confidences,
    int32_t* scheduled_prefix_lengths,
    uint8_t* scheduler_read_mask,
    float* scheduler_summary,
    const float* uniforms,
    const float* sts_temperatures,
    const float* steps_per_second,
    float sampling_temperature,
    bool calibration_enabled,
    bool prefix_enabled,
    int start_pos,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    int64_t proposal_epoch,
    uint32_t execution_mask,
    uint8_t draft_layer_mask) {
  const c10::cuda::CUDAGuard device_guard(workspace.device());
#ifdef DSPARK_V4_INTEGRATED_PROPOSAL_IO
  TORCH_CHECK(
      workspace.numel() >= generated::kWorkspaceBytes + 2 * sizeof(int64_t)
          + sizeof(dspark_proposal_io::IntegrationContext),
      "integrated proposal IO needs the admitted context after the device scalars");
  TORCH_CHECK(start_pos == -1 && proposal_epoch == -1,
      "integrated proposal IO owns the device position and epoch");
  TORCH_CHECK(execution_mask == kExecuteAll && draft_layer_mask == 7,
      "integrated proposal IO requires all three complete draft layers");
#endif
  const int device = workspace.get_device();
  int major = 0;
  int minor = 0;
  int sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  TORCH_CHECK(
      major == 10 && minor == 3 && sms == 152 && generated::kWorkers == 152,
      "DeepSeek-V4 scheduler requires one CTA on every GB300/sm_103 SM "
      "(device sm_",
      major,
      minor,
      ", sms=",
      sms,
      ", workers=",
      generated::kWorkers,
      ")");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      dspark_v4_scheduler_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kLaunchSharedBytes));
  int occupancy = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy,
      dspark_v4_scheduler_kernel,
      generated::kThreads,
      kLaunchSharedBytes));
  TORCH_CHECK(occupancy == 1, "V4 scheduler must reside as exactly one CTA per SM");
  const int cluster_dim = dspark_v4_cluster_dim();
#ifdef DSPARK_V4_STATIC_QUEUES
  // An AOT worker may wait on a task assigned to any other worker. Ask CUDA's
  // cluster occupancy API to prove all 152 dim-1 clusters are co-resident,
  // and fail closed before launch if they are not.
  check_cluster_residency(1);
#endif
  if (cluster_dim > 1) {
    check_cluster_residency(cluster_dim);
  }
  const auto stream = at::cuda::getCurrentCUDAStream(device);
  push_scheduler_policy(stream.stream());
#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
  reset_scheduler_probes(stream.stream());
#endif
#ifdef DSPARK_V4_TMA_WEIGHTS
  const CUtensorMap* expert_weight_tma_maps =
      prepare_expert_weight_tma_maps(weight_arena, weight_offsets);
#endif
#ifdef DSPARK_V4_STATIC_TP2_TAIL
  dspark_v4_tp2::DeviceContext tp2_context;
  if (execution_mask & kExecuteTail) {
    tp2_context = dspark_v4_tp2::prepare_rank0_launch(
        static_cast<uint32_t>(proposal_epoch),
        stream.stream(),
        weight_arena.data_ptr<uint8_t>(),
        weight_offsets.data_ptr<int64_t>());
  }
#endif
// The scheduler argument list is spelled once and shared by the shipped
// `<<<>>>` launch and the R14 cluster launch, so the two cannot drift apart.
#ifdef DSPARK_V4_TMA_WEIGHTS
#define DSPARK_V4_SCHEDULER_TMA_ARGS                                  \
  , expert_weight_tma_maps,                                           \
      (expert_weight_tma_maps == nullptr ? nullptr                    \
                                         : expert_weight_tma_maps + 1)
#else
#define DSPARK_V4_SCHEDULER_TMA_ARGS
#endif
#ifdef DSPARK_V4_STATIC_TP2_TAIL
#define DSPARK_V4_SCHEDULER_TP2_ARGS , tp2_context
#else
#define DSPARK_V4_SCHEDULER_TP2_ARGS
#endif
#define DSPARK_V4_SCHEDULER_ARGS                                        \
  weight_arena.data_ptr<uint8_t>(), weight_offsets.data_ptr<int64_t>(), \
      main_hidden, main_output, anchor, embedding_output, rope_cos_sin, \
      kv_cache, output_ids, corrected_logits, probabilities,            \
      confidence_logits, calibrated_confidences, scheduled_prefix_lengths, \
      scheduler_read_mask, scheduler_summary, uniforms, sts_temperatures, \
      steps_per_second, sampling_temperature, calibration_enabled,      \
      prefix_enabled, start_pos, workspace.data_ptr<uint8_t>(),         \
      trace_records.data_ptr<int64_t>(), trace_counts.data_ptr<int32_t>(), \
      launch_audit.data_ptr<uint64_t>(),                                \
      static_cast<uint32_t>(proposal_epoch), execution_mask,            \
      draft_layer_mask,                                                 \
      static_cast<int>(                                                 \
          weight_offsets.numel() - generated::kWeightOffsetSlots)       \
          DSPARK_V4_SCHEDULER_TMA_ARGS DSPARK_V4_SCHEDULER_TP2_ARGS
  if (cluster_dim > 1) {
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(generated::kWorkers);
    config.blockDim = dim3(generated::kThreads);
    config.dynamicSmemBytes = kLaunchSharedBytes;
    config.stream = stream.stream();
    cudaLaunchAttribute cluster_attr = {};
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim.x = static_cast<unsigned>(cluster_dim);
    cluster_attr.val.clusterDim.y = 1;
    cluster_attr.val.clusterDim.z = 1;
    config.attrs = &cluster_attr;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, dspark_v4_scheduler_kernel, DSPARK_V4_SCHEDULER_ARGS));
  } else {
    dspark_v4_scheduler_kernel<<<
        generated::kWorkers,
        generated::kThreads,
        kLaunchSharedBytes,
        stream.stream()>>>(DSPARK_V4_SCHEDULER_ARGS);
  }
#undef DSPARK_V4_SCHEDULER_ARGS
#undef DSPARK_V4_SCHEDULER_TP2_ARGS
#undef DSPARK_V4_SCHEDULER_TMA_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#ifdef DSPARK_V4_STATIC_TP2_TAIL
  if (tp2_context.enabled && !tp2_context.serving) {
    dspark_v4_tp2::record_rank0_launch_complete(
        static_cast<uint32_t>(proposal_epoch), stream.stream());
  }
#endif
}

void check_scheduler_abi(
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    int64_t proposal_epoch) {
  check_cuda_contiguous(weight_arena, "weight_arena");
  check_cuda_contiguous(weight_offsets, "weight_offsets");
  check_cuda_contiguous(workspace, "workspace");
  check_cuda_contiguous(trace_records, "trace_records");
  check_cuda_contiguous(trace_counts, "trace_counts");
  check_cuda_contiguous(launch_audit, "launch_audit");
  TORCH_CHECK(weight_arena.scalar_type() == torch::kUInt8, "weight_arena must be uint8");
  TORCH_CHECK(weight_offsets.scalar_type() == torch::kInt64, "weight_offsets must be int64");
#ifdef DSPARK_V4_STATIC_TP2_TAIL
  TORCH_CHECK(
      weight_offsets.numel() == generated::kWeightOffsetSlots + 2,
      "static TP2 tail requires the 4724 contract offsets plus BF16 LM/W2 offsets");
#else
  TORCH_CHECK(
      weight_offsets.numel() == generated::kWeightOffsetSlots
          || weight_offsets.numel() == generated::kWeightOffsetSlots + 2,
      "weight offset table shape mismatch (4724 contract slots, optionally "
      "+2 relaxed-numerics slots: BF16 lm_head, BF16 markov_w2)");
#endif
  TORCH_CHECK(workspace.scalar_type() == torch::kUInt8, "workspace must be uint8");
  TORCH_CHECK(
      workspace.numel() >= static_cast<int64_t>(generated::kWorkspaceBytes),
      "V4 scheduler workspace is too small");
  TORCH_CHECK(trace_records.scalar_type() == torch::kInt64, "trace must be int64");
  TORCH_CHECK(
      trace_records.numel()
          >= static_cast<int64_t>(generated::kWorkers)
              * generated::kTraceCapacity * generated::kTraceColumns,
      "V4 trace ring is too small");
  TORCH_CHECK(trace_counts.scalar_type() == torch::kInt32, "trace counts must be int32");
  TORCH_CHECK(trace_counts.numel() == generated::kWorkers, "trace count shape mismatch");
  TORCH_CHECK(launch_audit.scalar_type() == torch::kUInt64, "launch audit must be uint64");
  TORCH_CHECK(launch_audit.numel() >= 8, "launch audit is too small");
#ifdef DSPARK_V4_FULL_LOOP_DEVICE_EPOCH
  TORCH_CHECK(
      proposal_epoch == -1
          || (proposal_epoch > 0 && proposal_epoch < UINT32_MAX),
      "invalid proposal epoch");
#else
  TORCH_CHECK(proposal_epoch > 0 && proposal_epoch <= UINT32_MAX, "invalid proposal epoch");
#endif
  TORCH_CHECK(weight_arena.device() == workspace.device(), "V4 tensors must share a device");
}

}  // namespace

void dspark_v4_set_scheduler_policy(
    int64_t idle_sleep_ns,
    int64_t idle_sleep_long_ns,
    int64_t idle_sleep_after,
    int64_t flags,
    int64_t deep_period,
    int64_t express_workers,
    int64_t prefetch_period) {
  V4SchedPolicy& policy = host_scheduler_policy();
  policy.idle_sleep_ns = static_cast<unsigned int>(idle_sleep_ns);
  policy.idle_sleep_long_ns = static_cast<unsigned int>(idle_sleep_long_ns);
  policy.idle_sleep_after = static_cast<unsigned int>(idle_sleep_after);
  policy.flags = static_cast<unsigned int>(flags);
  policy.deep_period = static_cast<unsigned int>(deep_period);
  policy.express_workers = static_cast<unsigned int>(express_workers);
  policy.prefetch_period = static_cast<unsigned int>(prefetch_period);
  scheduler_policy_dirty() = true;
}

#if defined(DSPARK_V4_TP2_ABLATE)
// Select which rank's half of which bands this build executes. No numerical
// contract: the ablation build deliberately produces a partial result, and
// exists only to time one rank's workload.
void dspark_v4_set_tp2_ablation(int64_t rank, int64_t bands) {
  const uint32_t rank_value = static_cast<uint32_t>(rank);
  const uint32_t band_value = static_cast<uint32_t>(bands);
  C10_CUDA_CHECK(cudaMemcpyToSymbol(
      dspark_tp2_ablate::g_rank, &rank_value, sizeof(rank_value)));
  C10_CUDA_CHECK(cudaMemcpyToSymbol(
      dspark_tp2_ablate::g_bands, &band_value, sizeof(band_value)));
}
#endif

std::vector<int64_t> dspark_v4_get_scheduler_policy() {
  const V4SchedPolicy policy = host_scheduler_policy();
  return {
      static_cast<int64_t>(policy.idle_sleep_ns),
      static_cast<int64_t>(policy.idle_sleep_long_ns),
      static_cast<int64_t>(policy.idle_sleep_after),
      static_cast<int64_t>(policy.flags),
      static_cast<int64_t>(policy.deep_period),
      static_cast<int64_t>(policy.express_workers),
      static_cast<int64_t>(policy.prefetch_period),
  };
}

#if defined(DSPARK_V4_ENABLE_DEVICE_TRACE)
std::vector<torch::Tensor> dspark_v4_scheduler_probe_dump() {
  auto worker = torch::empty(
      {generated::kWorkers, kProbeWorkerSlots},
      torch::dtype(torch::kInt64).device(torch::kCPU));
  auto phase = torch::empty(
      {generated::kPhaseCount, kProbePhaseSlots},
      torch::dtype(torch::kInt64).device(torch::kCPU));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(
      worker.data_ptr<int64_t>(),
      g_probe_worker,
      sizeof(unsigned long long) * generated::kWorkers * kProbeWorkerSlots,
      0,
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(
      phase.data_ptr<int64_t>(),
      g_probe_phase,
      sizeof(unsigned long long) * generated::kPhaseCount * kProbePhaseSlots,
      0,
      cudaMemcpyDeviceToHost));
  return {worker, phase};
}
#endif

void dspark_v4_scheduler_smoke_cuda(
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    int64_t proposal_epoch) {
  check_scheduler_abi(
      weight_arena,
      weight_offsets,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch);
  launch_v4_scheduler(
      weight_arena,
      weight_offsets,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0.0f,
      false,
      false,
      0,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch,
      0,
      0);
}

void dspark_v4_main_stage_cuda(
    const torch::Tensor& main_hidden,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& output,
    int64_t proposal_epoch) {
  check_scheduler_abi(
      weight_arena,
      weight_offsets,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch);
  check_cuda_contiguous(main_hidden, "main_hidden");
  check_cuda_contiguous(output, "main_output");
  TORCH_CHECK(main_hidden.scalar_type() == torch::kBFloat16, "main_hidden must be BF16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "main_output must be BF16");
  TORCH_CHECK(main_hidden.numel() == kMainFeatureWidth, "main_hidden shape mismatch");
  TORCH_CHECK(output.numel() == kMainHidden, "main_output shape mismatch");
  TORCH_CHECK(main_hidden.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(output.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(
      weight_arena.numel() >= 14206958848LL,
      "main stage requires the complete converted V4 weight arena");
  launch_v4_scheduler(
      weight_arena,
      weight_offsets,
      reinterpret_cast<const __nv_bfloat16*>(main_hidden.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0.0f,
      false,
      false,
      0,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch,
      kExecuteMain,
      0);
}

void dspark_v4_front_stage_cuda(
    const torch::Tensor& anchor,
    const torch::Tensor& main_hidden,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& main_output,
    const torch::Tensor& embedding_output,
    int64_t proposal_epoch) {
  check_scheduler_abi(
      weight_arena,
      weight_offsets,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch);
  check_cuda_contiguous(anchor, "anchor");
  check_cuda_contiguous(main_hidden, "main_hidden");
  check_cuda_contiguous(main_output, "main_output");
  check_cuda_contiguous(embedding_output, "embedding_output");
  TORCH_CHECK(anchor.scalar_type() == torch::kInt32, "anchor must be int32");
  TORCH_CHECK(anchor.numel() == 1, "anchor must contain one token");
  TORCH_CHECK(main_hidden.scalar_type() == torch::kBFloat16, "main_hidden must be BF16");
  TORCH_CHECK(main_output.scalar_type() == torch::kBFloat16, "main_output must be BF16");
  TORCH_CHECK(
      embedding_output.scalar_type() == torch::kBFloat16,
      "embedding_output must be BF16");
  TORCH_CHECK(main_hidden.numel() == kMainFeatureWidth, "main_hidden shape mismatch");
  TORCH_CHECK(main_output.numel() == kMainHidden, "main_output shape mismatch");
  TORCH_CHECK(
      embedding_output.numel() == kDraftBlock * kHcStreams * kMainHidden,
      "embedding_output shape mismatch");
  TORCH_CHECK(anchor.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(main_hidden.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(main_output.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(
      embedding_output.device() == workspace.device(),
      "V4 tensors must share a device");
  TORCH_CHECK(
      weight_arena.numel() >= 14206958848LL,
      "front stage requires the complete converted V4 weight arena");
  launch_v4_scheduler(
      weight_arena,
      weight_offsets,
      reinterpret_cast<const __nv_bfloat16*>(main_hidden.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(main_output.data_ptr<at::BFloat16>()),
      anchor.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(embedding_output.data_ptr<at::BFloat16>()),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0.0f,
      false,
      false,
      0,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch,
      kExecuteMain | kExecuteEmbedding,
      0);
}

void dspark_v4_front_kv_stage_masked_cuda(
    const torch::Tensor& anchor,
    const torch::Tensor& main_hidden,
    const torch::Tensor& rope_cos_sin,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& main_output,
    const torch::Tensor& embedding_output,
    const torch::Tensor& kv_cache,
    const torch::Tensor& output_ids,
    const torch::Tensor& corrected_logits,
    const torch::Tensor& probabilities,
    const torch::Tensor& confidence_logits,
    const torch::Tensor& calibrated_confidences,
    const torch::Tensor& scheduled_prefix_lengths,
    const torch::Tensor& scheduler_read_mask,
    const torch::Tensor& scheduler_summary,
    const torch::Tensor& uniforms,
    const torch::Tensor& sts_temperatures,
    const torch::Tensor& steps_per_second,
    double sampling_temperature,
    bool calibration_enabled,
    bool prefix_enabled,
    int64_t start_pos,
    int64_t draft_layer_count,
    int64_t proposal_epoch,
    int64_t execution_mask,
    int64_t draft_layer_mask) {
  check_scheduler_abi(
      weight_arena,
      weight_offsets,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch);
  check_cuda_contiguous(anchor, "anchor");
  check_cuda_contiguous(main_hidden, "main_hidden");
  check_cuda_contiguous(rope_cos_sin, "rope_cos_sin");
  check_cuda_contiguous(main_output, "main_output");
  check_cuda_contiguous(embedding_output, "embedding_output");
  check_cuda_contiguous(kv_cache, "kv_cache");
  check_cuda_contiguous(output_ids, "output_ids");
  check_cuda_contiguous(corrected_logits, "corrected_logits");
  check_cuda_contiguous(probabilities, "draft_probabilities");
  check_cuda_contiguous(confidence_logits, "confidence_logits");
  check_cuda_contiguous(calibrated_confidences, "calibrated_confidences");
  check_cuda_contiguous(scheduled_prefix_lengths, "scheduled_prefix_lengths");
  check_cuda_contiguous(scheduler_read_mask, "scheduler_read_mask");
  check_cuda_contiguous(scheduler_summary, "scheduler_summary");
  check_cuda_contiguous(uniforms, "uniforms");
  check_cuda_contiguous(sts_temperatures, "sts_temperatures");
  check_cuda_contiguous(steps_per_second, "steps_per_second");
  // Batched serving: every per-sequence tensor is (batch, ...) contiguous.
  TORCH_CHECK(
      anchor.scalar_type() == torch::kInt32 && anchor.numel() == kBatch,
      "bad anchor");
  TORCH_CHECK(
      main_hidden.scalar_type() == torch::kBFloat16
          && main_hidden.numel() == kBatch * kMainFeatureWidth,
      "bad main_hidden");
  TORCH_CHECK(
      rope_cos_sin.scalar_type() == torch::kFloat32 && rope_cos_sin.numel() == 6 * 64,
      "rope_cos_sin must be float32 [6, 32, 2]");
  TORCH_CHECK(
      main_output.scalar_type() == torch::kBFloat16
          && main_output.numel() == kBatch * kMainHidden,
      "bad main_output");
  TORCH_CHECK(
      embedding_output.scalar_type() == torch::kBFloat16
          && embedding_output.numel() == kDraftRows * kHcStreams * kMainHidden,
      "bad embedding_output");
  TORCH_CHECK(
      kv_cache.scalar_type() == torch::kBFloat16
          && kv_cache.numel() == 3 * kBatch * 128 * 512,
      "kv_cache must be BF16 [3, batch, 128, 512]");
  TORCH_CHECK(
      output_ids.scalar_type() == torch::kInt32
          && output_ids.numel() == kBatch * 6,
      "output_ids must be int32 [batch, 6]");
  TORCH_CHECK(
      corrected_logits.scalar_type() == torch::kFloat32
          && corrected_logits.numel() == kBatch * 5 * 129280,
      "corrected_logits must be float32 [batch, 5, 129280]");
  TORCH_CHECK(
      probabilities.scalar_type() == torch::kFloat32
          && probabilities.numel() == kBatch * 5 * 129280,
      "draft_probabilities must be float32 [batch, 5, 129280]");
  TORCH_CHECK(
      confidence_logits.scalar_type() == torch::kFloat32
          && confidence_logits.numel() == kBatch * 5,
      "confidence_logits must be float32 [batch, 5]");
  TORCH_CHECK(
      calibrated_confidences.scalar_type() == torch::kFloat32
          && calibrated_confidences.numel() == kBatch * 5,
      "calibrated_confidences must be float32 [batch, 5]");
  TORCH_CHECK(
      scheduled_prefix_lengths.scalar_type() == torch::kInt32
          && scheduled_prefix_lengths.numel() == kBatch,
      "scheduled_prefix_lengths must be int32 [batch]");
  TORCH_CHECK(
      scheduler_read_mask.scalar_type() == torch::kUInt8
          && scheduler_read_mask.numel() == kBatch * 5,
      "scheduler_read_mask must be uint8 [batch, 5]");
  TORCH_CHECK(
      scheduler_summary.scalar_type() == torch::kFloat32
          && scheduler_summary.numel() == 4,
      "scheduler_summary must be float32 [4]");
  TORCH_CHECK(
      uniforms.scalar_type() == torch::kFloat32
          && uniforms.numel() == kBatch * 5,
      "uniforms must be float32 [batch, 5]");
  TORCH_CHECK(
      sts_temperatures.scalar_type() == torch::kFloat32
          && sts_temperatures.numel() == 5,
      "sts_temperatures must be float32 [5]");
  TORCH_CHECK(
      steps_per_second.scalar_type() == torch::kFloat32
          && steps_per_second.numel() == kBatch * 6 + 1,
      "steps_per_second must be float32 [batch * (block + 1) + 1]");
  TORCH_CHECK(sampling_temperature >= 0.0, "sampling_temperature must be non-negative");
#ifdef DSPARK_V4_GREEDY_TAIL
  TORCH_CHECK(
      sampling_temperature < 1.0e-5,
      "greedy-tail extension requires sampling_temperature < 1e-5");
#endif
#ifdef DSPARK_V4_RELAXED_DAG
  TORCH_CHECK(
      start_pos > 0 || start_pos == -1,
      "front KV stage requires a positive host position or the relaxed "
      "device-position sentinel");
  if (start_pos == -1) {
    TORCH_CHECK(
        workspace.numel()
            >= static_cast<int64_t>(generated::kWorkspaceBytes + sizeof(int64_t)),
        "relaxed device start position requires 8 extra workspace bytes");
  }
#ifdef DSPARK_V4_FULL_LOOP_DEVICE_EPOCH
  if (proposal_epoch == -1) {
    TORCH_CHECK(
        workspace.numel()
            >= static_cast<int64_t>(generated::kWorkspaceBytes + 2 * sizeof(int64_t)),
        "relaxed device proposal epoch requires 16 extra workspace bytes");
  }
#endif
#else
  TORCH_CHECK(start_pos > 0, "front KV stage is decode-only");
#endif
  TORCH_CHECK(
      draft_layer_count >= 1 && draft_layer_count <= 3,
      "draft_layer_count must be in [1, 3]");
  TORCH_CHECK(
      execution_mask >= 0 && (execution_mask & ~kExecuteAll) == 0,
      "execution_mask contains unsupported phase bits");
  TORCH_CHECK(
      draft_layer_mask >= 1 && draft_layer_mask <= 0x7,
      "draft_layer_mask must select at least one of three draft layers");
#if defined(DSPARK_V4_COMPILED_EXECUTION_MASK)
  TORCH_CHECK(
      execution_mask == DSPARK_V4_COMPILED_EXECUTION_MASK
          && draft_layer_mask == DSPARK_V4_COMPILED_DRAFT_LAYER_MASK,
      "runtime phase masks must match this compiled scheduler specialization");
#endif
#ifdef DSPARK_V4_STATIC_TP2_TAIL
  TORCH_CHECK(
      draft_layer_count == 3,
      "static TP2 tail requires draft_layer_count == 3");
  TORCH_CHECK(
      !(execution_mask & kExecuteTail) || draft_layer_mask == 0x7,
      "static TP2 tail requires all three draft layers");
#endif
  TORCH_CHECK(anchor.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(main_hidden.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(rope_cos_sin.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(main_output.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(
      embedding_output.device() == workspace.device(),
      "V4 tensors must share a device");
  TORCH_CHECK(kv_cache.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(output_ids.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(
      corrected_logits.device() == workspace.device(),
      "V4 tensors must share a device");
  TORCH_CHECK(probabilities.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(
      confidence_logits.device() == workspace.device(),
      "V4 tensors must share a device");
  TORCH_CHECK(
      calibrated_confidences.device() == workspace.device(),
      "V4 tensors must share a device");
  TORCH_CHECK(
      scheduled_prefix_lengths.device() == workspace.device(),
      "V4 tensors must share a device");
  TORCH_CHECK(
      scheduler_read_mask.device() == workspace.device(),
      "V4 tensors must share a device");
  TORCH_CHECK(scheduler_summary.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(uniforms.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(sts_temperatures.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(steps_per_second.device() == workspace.device(), "V4 tensors must share a device");
  TORCH_CHECK(
      weight_arena.numel() >= 14206958848LL,
      "front KV stage requires the complete converted V4 weight arena");
  launch_v4_scheduler(
      weight_arena,
      weight_offsets,
      reinterpret_cast<const __nv_bfloat16*>(main_hidden.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(main_output.data_ptr<at::BFloat16>()),
      anchor.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(embedding_output.data_ptr<at::BFloat16>()),
      rope_cos_sin.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr<at::BFloat16>()),
      output_ids.data_ptr<int32_t>(),
      corrected_logits.data_ptr<float>(),
      probabilities.data_ptr<float>(),
      confidence_logits.data_ptr<float>(),
      calibrated_confidences.data_ptr<float>(),
      scheduled_prefix_lengths.data_ptr<int32_t>(),
      scheduler_read_mask.data_ptr<uint8_t>(),
      scheduler_summary.data_ptr<float>(),
      uniforms.data_ptr<float>(),
      sts_temperatures.data_ptr<float>(),
      steps_per_second.data_ptr<float>(),
      static_cast<float>(sampling_temperature),
      calibration_enabled,
      prefix_enabled,
      static_cast<int>(start_pos),
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      proposal_epoch,
      static_cast<uint32_t>(execution_mask),
      static_cast<uint8_t>(draft_layer_mask));
}

void dspark_v4_front_kv_stage_cuda(
    const torch::Tensor& anchor,
    const torch::Tensor& main_hidden,
    const torch::Tensor& rope_cos_sin,
    const torch::Tensor& weight_arena,
    const torch::Tensor& weight_offsets,
    const torch::Tensor& workspace,
    const torch::Tensor& trace_records,
    const torch::Tensor& trace_counts,
    const torch::Tensor& launch_audit,
    const torch::Tensor& main_output,
    const torch::Tensor& embedding_output,
    const torch::Tensor& kv_cache,
    const torch::Tensor& output_ids,
    const torch::Tensor& corrected_logits,
    const torch::Tensor& probabilities,
    const torch::Tensor& confidence_logits,
    const torch::Tensor& calibrated_confidences,
    const torch::Tensor& scheduled_prefix_lengths,
    const torch::Tensor& scheduler_read_mask,
    const torch::Tensor& scheduler_summary,
    const torch::Tensor& uniforms,
    const torch::Tensor& sts_temperatures,
    const torch::Tensor& steps_per_second,
    double sampling_temperature,
    bool calibration_enabled,
    bool prefix_enabled,
    int64_t start_pos,
    int64_t draft_layer_count,
    int64_t proposal_epoch) {
  dspark_v4_front_kv_stage_masked_cuda(
      anchor,
      main_hidden,
      rope_cos_sin,
      weight_arena,
      weight_offsets,
      workspace,
      trace_records,
      trace_counts,
      launch_audit,
      main_output,
      embedding_output,
      kv_cache,
      output_ids,
      corrected_logits,
      probabilities,
      confidence_logits,
      calibrated_confidences,
      scheduled_prefix_lengths,
      scheduler_read_mask,
      scheduler_summary,
      uniforms,
      sts_temperatures,
      steps_per_second,
      sampling_temperature,
      calibration_enabled,
      prefix_enabled,
      start_pos,
      draft_layer_count,
      proposal_epoch,
      kExecuteAll,
      static_cast<int64_t>((1U << draft_layer_count) - 1U));
}
