from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.deprecations import replace_and_warn_about_deprecated_multistart_options
from optimagic.exceptions import InvalidMultistartError
from optimagic.typing import BatchEvaluator, BatchEvaluatorLiteral, PyTree



@dataclass(frozen=True)
class MultistartOptions:

    n_samples: int | None = None
    stopping_maxopt: int | None = None
    sampling_distribution: Literal["uniform", "triangular"] = "uniform"
    sampling_method: Literal["sobol", "random", "halton", "latin_hypercube"] = "random"
    sample: Sequence[PyTree] | None = None
    mixing_weight_method: (
        Literal["tiktak", "linear"] | Callable[[int, int, float, float], float]
    ) = "tiktak"
    mixing_weight_bounds: tuple[float, float] = (0.1, 0.995)
    convergence_xtol_rel: float | None = None
    convergence_max_discoveries: int = 2
    n_cores: int = 1
    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator = "joblib"
    batch_size: int | None = None
    seed: int | np.random.Generator | None = None
    error_handling: Literal["raise", "continue"] | None = None
    share_optimization: float | None = None
    convergence_relative_params_tolerance: float | None = None
    optimization_error_handling: Literal["raise", "continue"] | None = None
    exploration_error_handling: Literal["raise", "continue"] | None = None

    def __post_init__(self) -> None:
        _validate_attribute_types_and_values(self)


class MultistartOptionsDict(TypedDict):
    n_samples: NotRequired[int | None]
    stopping_maxopt: NotRequired[int | None]
    sampling_distribution: NotRequired[Literal["uniform", "triangular"]]
    sampling_method: NotRequired[
        Literal["sobol", "random", "halton", "latin_hypercube"]
    ]
    sample: NotRequired[Sequence[PyTree] | None]
    mixing_weight_method: NotRequired[
        Literal["tiktak", "linear"] | Callable[[int, int, float, float], float]
    ]
    mixing_weight_bounds: NotRequired[tuple[float, float]]
    convergence_xtol_rel: NotRequired[float | None]
    convergence_max_discoveries: NotRequired[int]
    n_cores: NotRequired[int]
    batch_evaluator: NotRequired[BatchEvaluatorLiteral | BatchEvaluator]
    batch_size: NotRequired[int | None]
    seed: NotRequired[int | np.random.Generator | None]
    error_handling: NotRequired[Literal["raise", "continue"] | None]
    share_optimization: NotRequired[float | None]
    convergence_relative_params_tolerance: NotRequired[float | None]
    optimization_error_handling: NotRequired[Literal["raise", "continue"] | None]
    exploration_error_handling: NotRequired[Literal["raise", "continue"] | None]


def pre_process_multistart(
    multistart: bool | MultistartOptions | MultistartOptionsDict | None,
) -> MultistartOptions | None:
    """Convert all valid types of multistart to a optimagic.MultistartOptions.

    This just harmonizes multiple ways of specifying multistart options into a single
    format. It performs runime type checks, but it does not check whether multistart
    options are consistent with other option choices.

    Args:
        multistart: The user provided multistart options.
        n_params: The number of parameters in the optimization problem.

    Returns:
        The multistart options in the optimagic format.

    Raises:
        InvalidMultistartError: If the multistart options cannot be processed, e.g.
            because they do not have the correct type.

    """
    if isinstance(multistart, bool):
        multistart = MultistartOptions() if multistart else None
    elif isinstance(multistart, MultistartOptions) or multistart is None:
        pass
    else:
        try:
            multistart = MultistartOptions(**multistart)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if isinstance(e, InvalidMultistartError):
                raise e
            raise InvalidMultistartError(
                f"Invalid multistart options of type: {type(multistart)}. Multistart "
                "options must be of type optimagic.MultistartOptions, a dictionary "
                "with valid keys, None, or a boolean."
            ) from e

    if multistart is not None:
        multistart = replace_and_warn_about_deprecated_multistart_options(multistart)
        multistart = cast(MultistartOptions, multistart)

    return multistart


def _validate_attribute_types_and_values(options: MultistartOptions) -> None:
    pass




def _tiktak_weights(
    iteration: int, n_iterations: int, min_weight: float, max_weight: float
) -> float:
    return np.clip(np.sqrt(iteration / n_iterations), min_weight, max_weight)


def _linear_weights(
    iteration: int, n_iterations: int, min_weight: float, max_weight: float
) -> float:
    unscaled = iteration / n_iterations
    span = max_weight - min_weight
    return min_weight + unscaled * span


WEIGHT_FUNCTIONS = {
    "tiktak": _tiktak_weights,
    "linear": _linear_weights,
}


@dataclass(frozen=True)
class InternalMultistartOptions:

    n_samples: int
    weight_func: Callable[[int, int], float]
    convergence_xtol_rel: float
    convergence_max_discoveries: int
    sampling_distribution: Literal["uniform", "triangular"]
    sampling_method: Literal["sobol", "random", "halton", "latin_hypercube"]
    sample: NDArray[np.float64] | None
    seed: int | np.random.Generator | None
    n_cores: int
    batch_evaluator: BatchEvaluator
    batch_size: int
    error_handling: Literal["raise", "continue"]
    stopping_maxopt: int

    def __post_init__(self) -> None:
        must_be_at_least_1 = [
            "n_samples",
            "stopping_maxopt",
            "n_cores",
            "batch_size",
            "convergence_max_discoveries",
        ]

        for attr in must_be_at_least_1:
            if getattr(self, attr) < 1:
                raise InvalidMultistartError(f"{attr} must be at least 1.")

        if self.batch_size < self.n_cores:
            raise InvalidMultistartError("batch_size must be at least n_cores.")

        if self.convergence_xtol_rel < 0:
            raise InvalidMultistartError("convergence_xtol_rel must be at least 0.")


def get_internal_multistart_options_from_public(
    options: MultistartOptions,
    params: PyTree,
    params_to_internal: Callable[[PyTree], NDArray[np.float64]],
) -> InternalMultistartOptions:
    """Get internal multistart options from public multistart options.

    Args:
        options: The pre-processed multistart options.
        params: The parameters of the optimization problem.
        params_to_internal: A function that converts parameters to internal parameters.

    Returns:
        InternalMultistartOptions: The updated options with runtime defaults.

    """
    x = params_to_internal(params)

    if options.sample is not None:
        sample = np.array([params_to_internal(x) for x in list(options.sample)])
        n_samples = len(options.sample)
    else:
        sample = None
        n_samples = options.n_samples  # type: ignore

    batch_size = options.n_cores if options.batch_size is None else options.batch_size
    batch_evaluator = process_batch_evaluator(options.batch_evaluator)

    if callable(options.mixing_weight_method):
        weight_func = options.mixing_weight_method
    else:
        _weight_method = WEIGHT_FUNCTIONS[options.mixing_weight_method]

    weight_func = partial(
        _weight_method,
        min_weight=options.mixing_weight_bounds[0],
        max_weight=options.mixing_weight_bounds[1],
    )

    if n_samples is None:
        if options.stopping_maxopt is None:
            n_samples = 100 * len(x)
        else:
            n_samples = 10 * options.stopping_maxopt

    if options.share_optimization is None:
        share_optimization = 0.1
    else:
        share_optimization = options.share_optimization

    if options.stopping_maxopt is None:
        stopping_maxopt = max(1, int(share_optimization * n_samples))
    else:
        stopping_maxopt = options.stopping_maxopt

    if options.error_handling is not None:
        error_handling = options.error_handling
    else:
        error_handling = "continue"

    if options.convergence_xtol_rel is not None:
        convergence_xtol_rel = options.convergence_xtol_rel
    else:
        convergence_xtol_rel = 0.01

    return InternalMultistartOptions(
        convergence_max_discoveries=options.convergence_max_discoveries,
        n_cores=options.n_cores,
        sampling_distribution=options.sampling_distribution,
        sampling_method=options.sampling_method,
        seed=options.seed,
        sample=sample,
        n_samples=n_samples,
        weight_func=weight_func,
        error_handling=error_handling,
        convergence_xtol_rel=convergence_xtol_rel,
        stopping_maxopt=stopping_maxopt,
        batch_evaluator=batch_evaluator,
        batch_size=batch_size,
    )
