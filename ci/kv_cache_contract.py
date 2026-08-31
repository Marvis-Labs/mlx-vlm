from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence


class StorageProfile(str, Enum):
    DENSE = "dense"
    WINDOWED = "windowed"
    SEGMENTED = "segmented"
    RECURRENT = "recurrent"
    POOLING = "pooling"
    QUANTIZED = "quantized"
    PREFIX = "prefix"


class CacheCapability(str, Enum):
    UPDATE = "update"
    TRIM = "trim"
    RESET = "reset"
    SNAPSHOT_RESTORE = "snapshot_restore"
    EXTRACT = "extract"
    MERGE = "merge"
    FILTER = "filter"
    BATCH_LIFECYCLE = "batch_lifecycle"
    ADVANCE = "advance"
    QUANTIZE = "quantize"


class CacheCharacteristic(str, Enum):
    CONTENT = "content"
    VISIBILITY = "visibility"
    POSITION = "position"
    SHAPE = "shape"
    DTYPE = "dtype"
    BATCH_LAYOUT = "batch_layout"
    METADATA = "metadata"
    STORAGE = "storage"
    APPROXIMATION = "approximation"


class CacheOperationKind(str, Enum):
    UPDATE = "update"
    TRIM = "trim"
    RESET = "reset"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"
    EXTRACT = "extract"
    MERGE = "merge"
    FILTER = "filter"
    PREPARE_BATCH = "prepare_batch"
    FINALIZE_BATCH = "finalize_batch"
    ADVANCE = "advance"
    QUANTIZE = "quantize"


OPERATION_CAPABILITIES = {
    CacheOperationKind.UPDATE: CacheCapability.UPDATE,
    CacheOperationKind.TRIM: CacheCapability.TRIM,
    CacheOperationKind.RESET: CacheCapability.RESET,
    CacheOperationKind.SNAPSHOT: CacheCapability.SNAPSHOT_RESTORE,
    CacheOperationKind.RESTORE: CacheCapability.SNAPSHOT_RESTORE,
    CacheOperationKind.EXTRACT: CacheCapability.EXTRACT,
    CacheOperationKind.MERGE: CacheCapability.MERGE,
    CacheOperationKind.FILTER: CacheCapability.FILTER,
    CacheOperationKind.PREPARE_BATCH: CacheCapability.BATCH_LIFECYCLE,
    CacheOperationKind.FINALIZE_BATCH: CacheCapability.BATCH_LIFECYCLE,
    CacheOperationKind.ADVANCE: CacheCapability.ADVANCE,
    CacheOperationKind.QUANTIZE: CacheCapability.QUANTIZE,
}


@dataclass(frozen=True)
class CacheOperation:
    kind: CacheOperationKind
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperationSequence:
    name: str
    operations: tuple[CacheOperation, ...]


@dataclass(frozen=True)
class CacheObservation:
    logical_keys: Any = None
    logical_values: Any = None
    visible_positions: Any = None
    offset: Any = None
    size: Any = None
    shape: Any = None
    dtype: Any = None
    batch_size: Any = None
    metadata: Any = None
    allocated_bytes: Any = None
    approximation_error: Any = None

    def characteristic(self, characteristic: CacheCharacteristic) -> Any:
        values = {
            CacheCharacteristic.CONTENT: (self.logical_keys, self.logical_values),
            CacheCharacteristic.VISIBILITY: self.visible_positions,
            CacheCharacteristic.POSITION: (self.offset, self.size),
            CacheCharacteristic.SHAPE: self.shape,
            CacheCharacteristic.DTYPE: self.dtype,
            CacheCharacteristic.BATCH_LAYOUT: self.batch_size,
            CacheCharacteristic.METADATA: self.metadata,
            CacheCharacteristic.STORAGE: self.allocated_bytes,
            CacheCharacteristic.APPROXIMATION: self.approximation_error,
        }
        return values[characteristic]


@dataclass(frozen=True)
class Tolerance:
    absolute: float = 0.0
    relative: float = 0.0


class CacheAdapter(Protocol):
    @property
    def capabilities(self) -> frozenset[CacheCapability]: ...

    def apply(self, operation: CacheOperation) -> CacheObservation: ...


class CacheOracle(CacheAdapter, Protocol):
    pass


Comparator = Callable[[Any, Any, Tolerance], str | None]


@dataclass(frozen=True)
class CacheContractCase:
    name: str
    profile: StorageProfile
    subject_factory: Callable[[], CacheAdapter]
    oracle_factory: Callable[[], CacheOracle]
    capabilities: frozenset[CacheCapability]
    characteristics: frozenset[CacheCharacteristic]
    sequences: tuple[OperationSequence, ...]
    tolerance: Tolerance = Tolerance()
    comparators: Mapping[CacheCharacteristic, Comparator] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractFailure:
    sequence: str
    step: int
    operation: str
    characteristic: str | None
    reason: str
    expected: str | None = None
    actual: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "sequence": self.sequence,
                "step": self.step,
                "operation": self.operation,
                "characteristic": self.characteristic,
                "reason": self.reason,
                "expected": self.expected,
                "actual": self.actual,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class ContractRun:
    sequence: str
    operations: int
    executed_operations: int
    checks: int
    failures: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "verdict": "passed" if self.failures == 0 else "test_failure",
            "operations": self.operations,
            "executed_operations": self.executed_operations,
            "checks": self.checks,
            "failures": self.failures,
        }


@dataclass(frozen=True)
class ContractResult:
    case: str
    profile: str
    checks: int
    failures: tuple[ContractFailure, ...]
    runs: tuple[ContractRun, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "profile": self.profile,
            "verdict": "passed" if self.passed else "test_failure",
            "checks": self.checks,
            "runs": [run.to_dict() for run in self.runs],
            "failures": [failure.to_dict() for failure in self.failures],
        }


class ContractRunner:
    """Compare a cache implementation with an independent semantic oracle."""

    def run(self, case: CacheContractCase) -> ContractResult:
        failures: list[ContractFailure] = []
        runs: list[ContractRun] = []
        checks = 0
        for sequence in case.sequences:
            sequence_checks = checks
            sequence_failures = len(failures)
            executed_operations = 0
            subject = case.subject_factory()
            oracle = case.oracle_factory()
            for step, operation in enumerate(sequence.operations):
                executed_operations += 1
                missing = self._missing_capability(case, subject, oracle, operation)
                if missing is not None:
                    failures.append(
                        ContractFailure(
                            sequence=sequence.name,
                            step=step,
                            operation=operation.kind.value,
                            characteristic=None,
                            reason=missing,
                        )
                    )
                    break
                try:
                    expected = oracle.apply(operation)
                except Exception as error:
                    failures.append(
                        ContractFailure(
                            sequence=sequence.name,
                            step=step,
                            operation=operation.kind.value,
                            characteristic=None,
                            reason=f"oracle_error:{type(error).__name__}:{error}",
                        )
                    )
                    break
                try:
                    actual = subject.apply(operation)
                except Exception as error:
                    failures.append(
                        ContractFailure(
                            sequence=sequence.name,
                            step=step,
                            operation=operation.kind.value,
                            characteristic=None,
                            reason=f"subject_error:{type(error).__name__}:{error}",
                        )
                    )
                    break
                for characteristic in sorted(
                    case.characteristics, key=lambda item: item.value
                ):
                    checks += 1
                    comparator = case.comparators.get(characteristic, compare_values)
                    expected_value = expected.characteristic(characteristic)
                    actual_value = actual.characteristic(characteristic)
                    reason = comparator(expected_value, actual_value, case.tolerance)
                    if reason is not None:
                        failures.append(
                            ContractFailure(
                                sequence=sequence.name,
                                step=step,
                                operation=operation.kind.value,
                                characteristic=characteristic.value,
                                reason=reason,
                                expected=_short_repr(expected_value),
                                actual=_short_repr(actual_value),
                            )
                        )
            runs.append(
                ContractRun(
                    sequence=sequence.name,
                    operations=len(sequence.operations),
                    executed_operations=executed_operations,
                    checks=checks - sequence_checks,
                    failures=len(failures) - sequence_failures,
                )
            )
        return ContractResult(
            case=case.name,
            profile=case.profile.value,
            checks=checks,
            failures=tuple(failures),
            runs=tuple(runs),
        )

    @staticmethod
    def _missing_capability(
        case: CacheContractCase,
        subject: CacheAdapter,
        oracle: CacheOracle,
        operation: CacheOperation,
    ) -> str | None:
        required = OPERATION_CAPABILITIES[operation.kind]
        owners = (
            ("case", case.capabilities),
            ("subject", subject.capabilities),
            ("oracle", oracle.capabilities),
        )
        missing = [
            name for name, capabilities in owners if required not in capabilities
        ]
        if not missing:
            return None
        return f"unsupported_operation:{required.value}:{','.join(missing)}"


def compare_values(expected: Any, actual: Any, tolerance: Tolerance) -> str | None:
    path = _first_difference(expected, actual, tolerance, "value")
    return None if path is None else f"mismatch:{path}"


def _first_difference(
    expected: Any, actual: Any, tolerance: Tolerance, path: str
) -> str | None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return None if expected is actual else path
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return (
            None
            if math.isclose(
                float(expected),
                float(actual),
                abs_tol=tolerance.absolute,
                rel_tol=tolerance.relative,
            )
            else path
        )
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            return f"{path}.keys"
        for key in sorted(expected, key=str):
            difference = _first_difference(
                expected[key], actual[key], tolerance, f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if _is_sequence(expected) and _is_sequence(actual):
        if len(expected) != len(actual):
            return f"{path}.length"
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            difference = _first_difference(
                expected_value,
                actual_value,
                tolerance,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    return None if expected == actual else path


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _short_repr(value: Any, limit: int = 240) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."
