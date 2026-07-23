
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from optimagic.exceptions import InvalidConstraintError
from optimagic.optimization.algo_options import CONSTRAINTS_ABSOLUTE_TOLERANCE
from optimagic.typing import PyTree

if TYPE_CHECKING:
    from optimagic.parameters.constraints.resolution import ResolutionContext

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class Constraint(ABC):

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _resolve(self, context: ResolutionContext) -> ResolvedConstraint | None:
        """Resolve the constraint's selectors to flat parameter positions.

        Returns None if the selection is empty, in which case the constraint is
        dropped.

        """


@dataclass(frozen=True)
class ConstraintSource:

    constraint: Constraint
    position: int

    def describe(self) -> str:
        return f"constraint {self.position}: {self.constraint!r}"


class ResolvedConstraint(ABC):  # noqa: B024
    pass


def _as_position_array(positions: Any) -> IntArray:
    pass


def _as_float_array(values: Any) -> FloatArray:
    pass


def identity_selector(x: PyTree) -> PyTree:
    return x


@dataclass(frozen=True)
class FixedConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "fixed", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

    def _resolve(self, context: ResolutionContext) -> ResolvedFixedConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedFixedConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedFixedConstraint(ResolvedConstraint):

    index: IntArray
    sources: tuple[ConstraintSource, ...]
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True)
class IncreasingConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "increasing", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedIncreasingConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedIncreasingConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedIncreasingConstraint(ResolvedConstraint):

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True)
class DecreasingConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "decreasing", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedDecreasingConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedDecreasingConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedDecreasingConstraint(ResolvedConstraint):

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True)
class EqualityConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "equality", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

    def _resolve(self, context: ResolutionContext) -> ResolvedEqualityConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedEqualityConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedEqualityConstraint(ResolvedConstraint):

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True)
class ProbabilityConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "probability", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedProbabilityConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedProbabilityConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedProbabilityConstraint(ResolvedConstraint):

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True)
class PairwiseEqualityConstraint(Constraint):

    selectors: list[Callable[[PyTree], PyTree]]

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "pairwise_equality", "selectors": self.selectors}

    def __post_init__(self) -> None:
        if len(self.selectors) < 2:
            raise InvalidConstraintError("At least two selectors must be provided.")

        if not all(callable(s) for s in self.selectors):
            raise InvalidConstraintError("All selectors must be callable.")

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedPairwiseEqualityConstraint | None:
        indices = tuple(context.select(selector) for selector in self.selectors)

        lengths = [len(index) for index in indices]
        if len(set(lengths)) != 1:
            msg = (
                "All selections of a pairwise equality constraint need to have the "
                f"same length. You have lengths {lengths} in "
                f"{context.source.describe()}."
            )
            raise InvalidConstraintError(msg)

        if len(indices[0]) == 0:
            return None

        return ResolvedPairwiseEqualityConstraint(
            indices=indices, sources=(context.source,)
        )


@dataclass(frozen=True, eq=False)
class ResolvedPairwiseEqualityConstraint(ResolvedConstraint):

    indices: tuple[IntArray, ...]
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        frozen = tuple(_as_position_array(index) for index in self.indices)
        object.__setattr__(self, "indices", frozen)


@dataclass(frozen=True)
class FlatCovConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    regularization: float = 0.0

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "covariance",
            "selector": self.selector,
            "regularization": self.regularization,
        }

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if not isinstance(self.regularization, float | int) or self.regularization < 0:
            raise InvalidConstraintError(
                "'regularization' must be a non-negative float or int."
            )

    def _resolve(self, context: ResolutionContext) -> ResolvedFlatCovConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedFlatCovConstraint(
            index=index,
            regularization=self.regularization,
            sources=(context.source,),
        )


@dataclass(frozen=True, eq=False)
class ResolvedFlatCovConstraint(ResolvedConstraint):

    index: IntArray
    regularization: float
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True)
class FlatSDCorrConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    regularization: float = 0.0

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "sdcorr",
            "selector": self.selector,
            "regularization": self.regularization,
        }

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if not isinstance(self.regularization, float | int) or self.regularization < 0:
            raise InvalidConstraintError(
                "'regularization' must be a non-negative float or int."
            )

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedFlatSDCorrConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedFlatSDCorrConstraint(
            index=index,
            regularization=self.regularization,
            sources=(context.source,),
        )


@dataclass(frozen=True, eq=False)
class ResolvedFlatSDCorrConstraint(ResolvedConstraint):

    index: IntArray
    regularization: float
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True)
class LinearConstraint(Constraint):

    selector: Callable[[PyTree], ArrayLike | "pd.Series[float]" | float | int] = (
        identity_selector
    )
    _: KW_ONLY
    weights: ArrayLike | "pd.Series[float]" | float | int | None = None
    lower_bound: float | int | None = None
    upper_bound: float | int | None = None
    value: float | int | None = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "linear",
            "selector": self.selector,
            "weights": self.weights,
            **_select_non_none(
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                value=self.value,
            ),
        }

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )

        if not isinstance(self.weights, np.ndarray | list | pd.Series | float | int):
            raise InvalidConstraintError(
                "'weights' must be an array-like, a pandas Series, a float, or an int."
            )

        if self.lower_bound is not None and not isinstance(
            self.lower_bound, float | int
        ):
            raise InvalidConstraintError("'lower_bound' must be a float or an int.")

        if self.upper_bound is not None and not isinstance(
            self.upper_bound, float | int
        ):
            raise InvalidConstraintError("'upper_bound' must be a float or an int.")

        if self.value is not None and not isinstance(self.value, float | int):
            raise InvalidConstraintError("'value' must be a float or an int.")

    def _resolve(self, context: ResolutionContext) -> ResolvedLinearConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedLinearConstraint(
            index=index,
            weights=self._aligned_weights(index, context.source),
            lower_bound=-np.inf if self.lower_bound is None else self.lower_bound,
            upper_bound=np.inf if self.upper_bound is None else self.upper_bound,
            value=np.nan if self.value is None else self.value,
            sources=(context.source,),
        )

    def _aligned_weights(self, index: IntArray, source: ConstraintSource) -> FloatArray:
        """Broadcast and length-check the weights against the selected positions."""
        if isinstance(self.weights, (np.ndarray, list, tuple, pd.Series)):
            if len(self.weights) != len(index):
                msg = (
                    f"weights of length {len(self.weights)} could not be aligned "
                    f"with the {len(index)} selected parameters in "
                    f"{source.describe()}."
                )
                raise InvalidConstraintError(msg)
            out = np.asarray(self.weights, dtype=np.float64)
        elif isinstance(self.weights, (float, int)):
            out = np.full(len(index), float(self.weights))
        else:
            msg = (
                f"Invalid type for linear weights: {type(self.weights)}. The "
                f"problematic constraint is {source.describe()}."
            )
            raise InvalidConstraintError(msg)
        return out


@dataclass(frozen=True, eq=False)
class ResolvedLinearConstraint(ResolvedConstraint):

    index: IntArray
    weights: FloatArray
    sources: tuple[ConstraintSource, ...]
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    value: float = np.nan

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))
        object.__setattr__(self, "weights", _as_float_array(self.weights))


@dataclass(frozen=True)
class NonlinearConstraint(Constraint):

    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    func: Callable[[PyTree], ArrayLike | "pd.Series[float]" | float] | None = None
    derivative: Callable[[PyTree], PyTree] | None = None
    lower_bound: ArrayLike | "pd.Series[float]" | float | None = None
    upper_bound: ArrayLike | "pd.Series[float]" | float | None = None
    value: ArrayLike | "pd.Series[float]" | float | None = None
    tol: float = CONSTRAINTS_ABSOLUTE_TOLERANCE

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "nonlinear",
            "selector": self.selector,
            **_select_non_none(
                func=self.func,
                derivative=self.derivative,
                lower_bounds=self.lower_bound,
                upper_bounds=self.upper_bound,
                value=self.value,
                tol=self.tol,
            ),
        }

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )

        if self.tol is not None and (
            not isinstance(self.tol, float | int) or self.tol < 0
        ):
            raise InvalidConstraintError("'tol' must be non-negative.")

        if self.func is None or not callable(self.func):
            raise InvalidConstraintError("'func' must be callable.")

        if self.derivative is not None and not callable(self.derivative):
            raise InvalidConstraintError("'derivative' must be callable.")

    def _resolve(self, context: ResolutionContext) -> ResolvedConstraint | None:
        raise NotImplementedError(
            "Nonlinear constraints are directly passed to optimizers that support "
            "them and must not be resolved."
        )


def _all_none(*args: Any) -> bool:
    return all(v is None for v in args)


def _select_non_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}
