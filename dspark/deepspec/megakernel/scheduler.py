from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ConfidenceReader = Callable[[int, int], float]
StepsPerSecond = Callable[[int], float]


@dataclass(frozen=True)
class PrefixSchedule:
    lengths: tuple[int, ...]
    expected_tokens: float
    target_batch_size: int
    throughput: float
    candidates_evaluated: int


@dataclass(frozen=True)
class VerificationCapacity:
    draft_token_budget: int
    expected_tokens: float
    target_batch_size: int
    throughput: float


@dataclass(frozen=True)
class AsyncPrefixSchedule:
    lengths: tuple[int, ...]
    draft_token_budget: int
    capacity_source_step: int | None


def _checked_probability(value: float, request: int, position: int) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence[{request}, {position}]={value} is outside [0, 1]")
    return value


def _checked_sps(sps: StepsPerSecond, batch_size: int) -> float:
    value = float(sps(batch_size))
    if value < 0.0:
        raise ValueError(f"SPS({batch_size})={value} must be non-negative")
    return value


def schedule_prefixes_causal(
    *,
    num_requests: int,
    max_draft_tokens: int,
    confidence_at: ConfidenceReader,
    steps_per_second: StepsPerSecond,
) -> PrefixSchedule:
    """Implement paper Algorithm 1 without eagerly reading future confidence.

    Each request supplies a monotonically non-increasing survival stream. A
    heap performs a lazy global merge. Critically, the next conditional
    confidence is read only after the current candidate improves throughput.
    """

    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if max_draft_tokens < 0:
        raise ValueError("max_draft_tokens must be non-negative")

    base_batch = int(num_requests)
    best_tau = float(num_requests)
    best_batch = base_batch
    best_theta = best_tau * _checked_sps(steps_per_second, best_batch)
    best_lengths = [0] * num_requests
    if max_draft_tokens == 0:
        return PrefixSchedule(
            lengths=tuple(best_lengths),
            expected_tokens=best_tau,
            target_batch_size=best_batch,
            throughput=best_theta,
            candidates_evaluated=0,
        )

    # Heap entries are (-survival, request, one-based prefix position).
    frontier: list[tuple[float, int, int]] = []
    for request in range(num_requests):
        survival = _checked_probability(confidence_at(request, 0), request, 0)
        if survival > 0.0:
            heapq.heappush(frontier, (-survival, request, 1))

    current_lengths = [0] * num_requests
    current_tau = float(num_requests)
    current_batch = base_batch
    evaluated = 0
    while frontier:
        neg_survival, request, position = heapq.heappop(frontier)
        survival = -neg_survival
        current_lengths[request] = position
        current_tau += survival
        current_batch += 1
        evaluated += 1
        theta = current_tau * _checked_sps(steps_per_second, current_batch)
        if theta <= best_theta:
            break

        best_theta = theta
        best_tau = current_tau
        best_batch = current_batch
        best_lengths = current_lengths.copy()

        # Expose a continuation-dependent confidence only after committing the
        # candidate that precedes it. This ordering is the losslessness gate.
        if position < max_draft_tokens:
            conditional = _checked_probability(confidence_at(request, position), request, position)
            next_survival = survival * conditional
            if next_survival > 0.0:
                heapq.heappush(
                    frontier,
                    (-next_survival, request, position + 1),
                )

    return PrefixSchedule(
        lengths=tuple(best_lengths),
        expected_tokens=best_tau,
        target_batch_size=best_batch,
        throughput=best_theta,
        candidates_evaluated=evaluated,
    )


def schedule_prefixes_from_sequences(
    confidences: Sequence[Sequence[float]],
    *,
    steps_per_second: StepsPerSecond,
) -> PrefixSchedule:
    if not confidences:
        raise ValueError("confidences must contain at least one request")
    width = len(confidences[0])
    if any(len(row) != width for row in confidences):
        raise ValueError("all confidence sequences must have equal length")
    return schedule_prefixes_causal(
        num_requests=len(confidences),
        max_draft_tokens=width,
        confidence_at=lambda request, position: confidences[request][position],
        steps_per_second=steps_per_second,
    )


def _cumulative_survivals(
    confidences: Sequence[Sequence[float]],
) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for request, row in enumerate(confidences):
        survival = 1.0
        for position, confidence in enumerate(row):
            survival *= _checked_probability(confidence, request, position)
            candidates.append((survival, request, position + 1))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates


def schedule_capacity_unconstrained(
    historical_confidences: Sequence[Sequence[float]],
    *,
    steps_per_second: StepsPerSecond,
) -> VerificationCapacity:
    """Production Section 5.2 capacity search over two-step-old signals.

    Unlike Algorithm 1, this search does not stop at the first throughput
    decline. That retrospective search is causal here only because callers
    must supply historical, rather than current-step, confidence values.
    """

    if not historical_confidences:
        raise ValueError("historical_confidences must contain at least one request")
    width = len(historical_confidences[0])
    if any(len(row) != width for row in historical_confidences):
        raise ValueError("all confidence sequences must have equal length")
    requests = len(historical_confidences)
    best_budget = 0
    best_tau = float(requests)
    best_batch = requests
    best_throughput = best_tau * _checked_sps(steps_per_second, best_batch)
    tau = float(requests)
    for budget, (survival, _, _) in enumerate(
        _cumulative_survivals(historical_confidences),
        start=1,
    ):
        tau += survival
        batch = requests + budget
        throughput = tau * _checked_sps(steps_per_second, batch)
        if throughput > best_throughput:
            best_budget = budget
            best_tau = tau
            best_batch = batch
            best_throughput = throughput
    return VerificationCapacity(
        draft_token_budget=best_budget,
        expected_tokens=best_tau,
        target_batch_size=best_batch,
        throughput=best_throughput,
    )


def select_current_top_k(
    current_confidences: Sequence[Sequence[float]],
    *,
    draft_token_budget: int,
) -> tuple[int, ...]:
    """Rank current tokens by current cumulative confidence under a fixed K."""

    if not current_confidences:
        raise ValueError("current_confidences must contain at least one request")
    width = len(current_confidences[0])
    if any(len(row) != width for row in current_confidences):
        raise ValueError("all confidence sequences must have equal length")
    maximum = len(current_confidences) * width
    if not 0 <= draft_token_budget <= maximum:
        raise ValueError(f"draft_token_budget={draft_token_budget} must be in [0, {maximum}]")
    lengths = [0] * len(current_confidences)
    for _, request, position in _cumulative_survivals(current_confidences)[:draft_token_budget]:
        lengths[request] = max(lengths[request], position)
    if sum(lengths) != draft_token_budget:
        raise AssertionError("top-K selection violated prefix closure")
    return tuple(lengths)


class AsyncTwoStepPrefixScheduler:
    """Two-step-lag production scheduler from DSpark Section 5.2."""

    def __init__(self, *, steps_per_second: StepsPerSecond):
        self._steps_per_second = steps_per_second
        self._history: list[tuple[tuple[float, ...], ...]] = []
        self._step = 0

    def schedule(
        self,
        current_confidences: Sequence[Sequence[float]],
    ) -> AsyncPrefixSchedule:
        if not current_confidences:
            raise ValueError("current_confidences must contain at least one request")
        normalized = tuple(tuple(float(value) for value in row) for row in current_confidences)
        width = len(normalized[0])
        if any(len(row) != width for row in normalized):
            raise ValueError("all confidence sequences must have equal length")
        source_step: int | None = None
        if self._step >= 2:
            source_step = self._step - 2
            capacity = schedule_capacity_unconstrained(
                self._history[source_step],
                steps_per_second=self._steps_per_second,
            )
            budget = capacity.draft_token_budget
        else:
            budget = len(normalized) * width
        lengths = select_current_top_k(
            normalized,
            draft_token_budget=budget,
        )
        self._history.append(normalized)
        result = AsyncPrefixSchedule(
            lengths=lengths,
            draft_token_budget=budget,
            capacity_source_step=source_step,
        )
        self._step += 1
        return result
