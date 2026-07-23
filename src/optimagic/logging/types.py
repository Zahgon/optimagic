from dataclasses import dataclass
from enum import Enum
from typing import Literal

from optimagic.optimization.fun_value import SpecificFunctionValue
from optimagic.typing import (
    DictLikeAccess,
    Direction,
    DirectionLiteral,
    PyTree,
)


class StepStatus(str, Enum):

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"


StepStatusLiteral = Literal["scheduled", "running", "complete", "skipped"]


class StepType(str, Enum):

    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"


StepTypeLiteral = Literal["optimization", "exploration"]


class ExistenceStrategy(str, Enum):

    RAISE = "raise"
    EXTEND = "extend"
    REPLACE = "replace"


ExistenceStrategyLiteral = Literal["raise", "extend", "replace"]


@dataclass(frozen=True)
class IterationState(DictLikeAccess):

    params: PyTree
    timestamp: float
    scalar_fun: float | None
    valid: bool
    raw_fun: SpecificFunctionValue | None
    step: int | None
    exceptions: str | None

    def combine(self, other: "IterationState") -> "IterationState":
        """Combine two iteration states.

        Args:
            other (IterationState): The second iteration state.

        Returns:
            IterationState: The combined iteration state.

        """
        raw = [e for e in [self.exceptions, other.exceptions] if e is not None]
        exceptions: str | None = None
        if raw:
            exceptions = "\n\n".join(raw)

        new = IterationState(
            params=self.params,
            timestamp=min(self.timestamp, other.timestamp),
            scalar_fun=self.scalar_fun or other.scalar_fun,
            valid=self.valid and other.valid,
            raw_fun=self.raw_fun or other.raw_fun,
            step=self.step,
            exceptions=exceptions,
        )
        return new


@dataclass(frozen=True)
class IterationStateWithId(IterationState):

    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError("rowid must not be None")


@dataclass(frozen=True)
class StepResult(DictLikeAccess):

    name: str
    type: StepType | StepTypeLiteral
    status: StepStatus | StepStatusLiteral
    n_iterations: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            object.__setattr__(self, "type", StepType(self.type))
        if isinstance(self.status, str):
            object.__setattr__(self, "status", StepStatus(self.status))


@dataclass(frozen=True)
class StepResultWithId(StepResult):

    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError("rowid must not be None")
        super().__post_init__()


@dataclass(frozen=True)
class ProblemInitialization(DictLikeAccess):

    direction: Direction | DirectionLiteral
    params: PyTree


@dataclass(frozen=True)
class ProblemInitializationWithId(ProblemInitialization):

    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError("rowid must not be None")
