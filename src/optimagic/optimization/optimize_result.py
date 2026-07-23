import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from optimagic import deprecations
from optimagic.logging.logger import LogReader
from optimagic.optimization.history import History
from optimagic.shared.compat import pd_df_map
from optimagic.typing import PyTree
from optimagic.utilities import to_pickle


@dataclass
class OptimizeResult:

    params: Any
    fun: float
    start_fun: float
    start_params: Any
    algorithm: str
    direction: str
    n_free: int

    message: str | None = None
    success: bool | None = None
    n_fun_evals: int | None = None
    n_jac_evals: int | None = None
    n_hess_evals: int | None = None
    n_iterations: int | None = None
    status: int | None = None
    jac: PyTree | None = None
    hess: PyTree | None = None
    hess_inv: PyTree | None = None
    max_constraint_violation: float | None = None

    history: History | None = None

    convergence_report: Dict | None = None

    multistart_info: Optional["MultistartInfo"] = None
    algorithm_output: Dict[str, Any] | None = None
    logger: LogReader | None = None


    @property
    def criterion(self) -> float:
        msg = "The criterion attribute is deprecated. Use the fun attribute instead."
        warnings.warn(msg, FutureWarning)
        return self.fun

    @property
    def start_criterion(self) -> float:
        pass

    @property
    def n_criterion_evaluations(self) -> int | None:
        pass

    @property
    def n_derivative_evaluations(self) -> int | None:
        pass


    @property
    def x(self) -> PyTree:
        pass

    @property
    def x0(self) -> PyTree:
        pass

    @property
    def nfev(self) -> int | None:
        pass

    @property
    def nit(self) -> int | None:
        pass

    @property
    def njev(self) -> int | None:
        pass

    @property
    def nhev(self) -> int | None:
        pass

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self) -> str:
        first_line = (
            f"{self.direction.title()} with {self.n_free} free parameters terminated"
        )

        if self.success is not None:
            snippet = "successfully" if self.success else "unsuccessfully"
            first_line += f" {snippet}"

        counters = [
            ("criterion evaluations", self.n_fun_evals),
            ("derivative evaluations", self.n_jac_evals),
            ("iterations", self.n_iterations),
        ]

        counters = [(n, v) for n, v in counters if v is not None]

        if counters:
            name, val = counters[0]
            counter_msg = f"after {val} {name}"
            if len(counters) >= 2:
                for name, val in counters[1:-1]:
                    counter_msg += f", {val} {name}"

                name, val = counters[-1]
                counter_msg += f" and {val} {name}"
            first_line += f" {counter_msg}"

        first_line += "."

        if self.message:
            message = f"The {self.algorithm} algorithm reported: {self.message}"
        else:
            message = None

        if self.start_fun is not None and self.fun is not None:
            improvement = (
                f"The value of criterion improved from {self.start_fun} to {self.fun}."
            )
        else:
            improvement = None

        if self.convergence_report is not None:
            convergence = _format_convergence_report(
                self.convergence_report, self.algorithm
            )
        else:
            convergence = None

        sections = [first_line, improvement, message, convergence]
        sections = [sec for sec in sections if sec is not None]

        msg = "\n\n".join(sections)

        return msg

    def to_pickle(self, path):
        """Save the OptimizeResult object to pickle.

        Args:
            path (str, pathlib.Path): A str or pathlib.path ending in .pkl or .pickle.

        """
        to_pickle(self, path=path)


@dataclass(frozen=True)
class MultistartInfo:

    start_parameters: list[PyTree]
    local_optima: list[OptimizeResult]
    exploration_sample: list[PyTree]
    exploration_results: list[float]

    def __getitem__(self, key):
        deprecations.throw_dict_access_future_warning(key, obj_name=type(self).__name__)
        return getattr(self, key)

    @property
    def n_optimizations(self) -> int:
        pass


def _format_convergence_report(report, algorithm):
    pass


def _create_stars(sr):
    stars = pd.cut(
        sr,
        bins=[-np.inf, 1e-10, 1e-8, 1e-5, np.inf],
        labels=["***", "** ", "*  ", "   "],
    ).astype("str")

    return stars


def _format_float(number):
    pass
