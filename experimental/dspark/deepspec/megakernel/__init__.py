"""Single-launch DSpark proposal kernels and observability utilities."""

from deepspec.megakernel.contract import (
    DeepSeekV4MegaKernelSpec,
    DSparkMegaKernelSpec,
    UnsupportedMegaKernelConfig,
)
from deepspec.megakernel.extension import (
    V4ProposalOutputs,
    V4SchedulerWorkspace,
    allocate_v4_proposal_outputs,
    allocate_v4_scheduler_workspace,
    run_v4_front_kv_stage,
    run_v4_front_stage,
    run_v4_main_stage,
    run_v4_scheduler_smoke,
    view_v4_workspace_region,
)
from deepspec.megakernel.scheduler import (
    AsyncPrefixSchedule,
    AsyncTwoStepPrefixScheduler,
    PrefixSchedule,
    VerificationCapacity,
    schedule_capacity_unconstrained,
    schedule_prefixes_causal,
    schedule_prefixes_from_sequences,
    select_current_top_k,
)
from deepspec.megakernel.tail_reference import (
    DSparkTailOutput,
    dspark_tail_reference,
)
from deepspec.megakernel.trace import (
    BubbleSummary,
    DeviceTrace,
    FullDeviceTrace,
    TracePhase,
)
from deepspec.megakernel.v4_abi import (
    V4LaunchABI,
    V4LaunchShape,
    build_v4_launch_abi,
)
from deepspec.megakernel.v4_schedule import (
    V4Phase,
    V4PhaseKind,
    V4WorkerRole,
    build_v4_phase_program,
)
from deepspec.megakernel.v4_trace import (
    V4DeviceTrace,
    V4TraceEvent,
    V4TraceFlag,
    V4TraceSegment,
    V4TraceSummary,
)
from deepspec.megakernel.v4_weights import (
    V4PackedWeights,
    V4WeightArenaPlan,
    expected_v4_weights,
    pack_v4_weights,
    plan_v4_weight_arena,
)

__all__ = [
    "AsyncPrefixSchedule",
    "AsyncTwoStepPrefixScheduler",
    "BubbleSummary",
    "DSparkMegaKernelSpec",
    "DSparkTailOutput",
    "DeepSeekV4MegaKernelSpec",
    "DeviceTrace",
    "FullDeviceTrace",
    "PrefixSchedule",
    "TracePhase",
    "UnsupportedMegaKernelConfig",
    "VerificationCapacity",
    "V4LaunchABI",
    "V4LaunchShape",
    "V4DeviceTrace",
    "V4Phase",
    "V4PhaseKind",
    "V4PackedWeights",
    "V4ProposalOutputs",
    "V4SchedulerWorkspace",
    "V4WorkerRole",
    "V4WeightArenaPlan",
    "V4TraceEvent",
    "V4TraceFlag",
    "V4TraceSegment",
    "V4TraceSummary",
    "build_v4_launch_abi",
    "build_v4_phase_program",
    "allocate_v4_proposal_outputs",
    "allocate_v4_scheduler_workspace",
    "expected_v4_weights",
    "pack_v4_weights",
    "plan_v4_weight_arena",
    "run_v4_scheduler_smoke",
    "run_v4_main_stage",
    "run_v4_front_stage",
    "run_v4_front_kv_stage",
    "view_v4_workspace_region",
    "dspark_tail_reference",
    "schedule_capacity_unconstrained",
    "schedule_prefixes_causal",
    "schedule_prefixes_from_sequences",
    "select_current_top_k",
]
