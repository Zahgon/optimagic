from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, TypedDict

from typing_extensions import NotRequired

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.config import DEFAULT_N_CORES
from optimagic.exceptions import InvalidNumdiffOptionsError
from optimagic.typing import BatchEvaluatorLiteral


@dataclass(frozen=True)
class NumdiffOptions:

    method: Literal[
        "central", "forward", "backward", "central_cross", "central_average"
    ] = "central"
    step_size: float | None = None
    scaling_factor: float = 1
    min_steps: float | None = None
    n_cores: int = DEFAULT_N_CORES
    batch_evaluator: BatchEvaluatorLiteral | Callable = "joblib"  # type: ignore

    def __post_init__(self) -> None:
        _validate_attribute_types_and_values(self)


class NumdiffOptionsDict(TypedDict):
    method: NotRequired[
        Literal["central", "forward", "backward", "central_cross", "central_average"]
    ]
    step_size: NotRequired[float | None]
    scaling_factor: NotRequired[float]
    min_steps: NotRequired[float | None]
    n_cores: NotRequired[int]
    batch_evaluator: NotRequired[BatchEvaluatorLiteral | Callable]  # type: ignore


def pre_process_numdiff_options(
    numdiff_options: NumdiffOptions | NumdiffOptionsDict | None,
) -> NumdiffOptions | None:
    """Convert all valid types of Numdiff options to optimagic.NumdiffOptions class.

    This just harmonizes multiple ways of specifying numdiff options into a single
    format. It performs runtime type checks, but it does not check whether numdiff
    options are consistent with other option choices.

    Args:
        numdiff_options: The user provided numdiff options.

    Returns:
        The numdiff options in the optimagic format.

    Raises:
        InvalidNumdiffOptionsError: If numdiff options cannot be processed, e.g. because
            they do not have the correct type.

    """
    if isinstance(numdiff_options, NumdiffOptions) or numdiff_options is None:
        pass
    else:
        try:
            numdiff_options = NumdiffOptions(**numdiff_options)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if isinstance(e, InvalidNumdiffOptionsError):
                raise e
            raise InvalidNumdiffOptionsError(
                f"Invalid numdiff options of type: {type(numdiff_options)}. Numdiff "
                "options must be of type optimagic.NumdiffOptions, a dictionary with a"
                "subset of the keys {'method', 'step_size', 'scaling_factor', "
                "'min_steps', 'n_cores', 'batch_evaluator'}, or None."
            ) from e

    return numdiff_options


def _validate_attribute_types_and_values(options: NumdiffOptions) -> None:
    pass


class NumdiffPurpose(str, Enum):
    OPTIMIZE = "optimize"
    ESTIMATE_JACOBIAN = "estimate_jacobian"
    ESTIMATE_HESSIAN = "estimate_hessian"


def get_default_numdiff_options(
    purpose: NumdiffPurpose,
) -> NumdiffOptions:
    """Get default numerical derivatives options for a given purpose.

    Args:
        purpose: For what purpose the numdiff options are used.

    Returns:
        The numdiff options with defaults filled in.

    """
    defaults: NumdiffOptionsDict = {}

    if purpose == NumdiffPurpose.OPTIMIZE:
        defaults["method"] = "forward"

    if purpose == NumdiffPurpose.ESTIMATE_JACOBIAN:
        defaults["method"] = "central"

    if purpose == NumdiffPurpose.ESTIMATE_HESSIAN:
        defaults["method"] = "central_cross"
        defaults["scaling_factor"] = 2

    return NumdiffOptions(**defaults)
