
from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_FTOL_ABS,
    CONVERGENCE_SECOND_BEST_XTOL_ABS,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt

InitSimplexLiteral = Literal["pfeffer", "nash", "gao_han", "varadhan_borchers"]
InitSimplexCallable = Callable[[NDArray[np.float64]], NDArray[np.float64]]
from optimagic.typing import BatchEvaluator, BatchEvaluatorLiteral


@mark.minimizer(
    name="neldermead_parallel",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=False,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=True,
)
@dataclass(frozen=True)
class NelderMeadParallel(Algorithm):

    init_simplex_method: InitSimplexLiteral | InitSimplexCallable = "gao_han"
    n_cores: PositiveInt = 1
    adaptive: bool = True
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_XTOL_ABS
    batch_evaluator: BatchEvaluator | BatchEvaluatorLiteral = "joblib"

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        raw = neldermead_parallel(
            criterion=cast(
                Callable[[NDArray[np.float64]], float],
                problem.fun,
            ),
            x=x0,
            init_simplex_method=self.init_simplex_method,
            n_cores=self.n_cores,
            adaptive=self.adaptive,
            stopping_maxiter=self.stopping_maxiter,
            convergence_ftol_abs=self.convergence_ftol_abs,
            convergence_xtol_abs=self.convergence_xtol_abs,
            batch_evaluator=self.batch_evaluator,
        )

        res = InternalOptimizeResult(
            x=raw["solution_x"],
            fun=raw["solution_criterion"],
            n_iterations=raw["n_iterations"],
            success=raw["success"],
            message=raw["reached_convergence_criterion"],
        )

        return res


def neldermead_parallel(
    criterion,
    x,
    *,
    init_simplex_method="gao_han",
    n_cores=1,
    adaptive=True,
    stopping_maxiter=STOPPING_MAXITER,
    convergence_ftol_abs=CONVERGENCE_SECOND_BEST_FTOL_ABS,
    convergence_xtol_abs=CONVERGENCE_SECOND_BEST_XTOL_ABS,
    batch_evaluator="joblib",
):
    if x.ndim >= 1:
        x = x.ravel()  # check if the vector of initial values is one-dimensional

    j = len(x)  # size of the parameter vector

    if n_cores <= 1:
        p = 1  # if number of cores is nonpositive, set it to 1
    else:
        if n_cores >= j:  # number of parallelisation cannot be bigger than
            p = int(j - 1)
        else:
            p = int(n_cores)

    alpha, gamma, beta, tau = _init_algo_params(adaptive, j)


    if not callable(init_simplex_method):
        s = globals()["_" + init_simplex_method](x)
    else:
        s = init_simplex_method(x)

    batch_evaluator = process_batch_evaluator(batch_evaluator)

    f_s = np.array(batch_evaluator(func=criterion, arguments=s, n_cores=n_cores))[
        :, None
    ]

    def func_parallel(args):
        pass

    optimal = False  # optmisation condition, if True stop the algorithem
    iterations = 0  # number of criterion evaluations

    while not optimal:
        iterations += 1  # new iteration

        row = np.argsort(f_s.ravel())
        s = np.take(s, row, axis=0)
        f_s = np.take(f_s, row, axis=0)

        m = (s[:-p, :].sum(axis=0)) / (j - p + 1)

        s_j_r = m + alpha * (m - s[-p:, :])

        s[-p:, :], f_s[-p:, :], shrink_count = np.split(
            np.vstack(
                batch_evaluator(
                    func=func_parallel,
                    arguments=tuple(
                        (
                            criterion,
                            s[j + 1 - p + i, :],
                            s_j_r[i, :],
                            f_s[0, :],
                            f_s[j + 1 - p + i, :],
                            f_s[j - p + i, :],
                            m,
                        )
                        for i in range(p)
                    ),
                    n_cores=p,
                )
            ),
            [-2, -1],
            axis=1,
        )

        if shrink_count.sum() == p:
            s = (
                tau * s[0:1, :] + (1 - tau) * s
            )  # new simplex is a linear combination of the best point
            f_s = np.array(
                batch_evaluator(
                    func=criterion,
                    arguments=s,
                    n_cores=n_cores,
                )
            )[:, None]

        if (
            np.max(np.abs(f_s[0, :] - f_s[1:, :])) <= convergence_ftol_abs
            and np.max(np.abs(s[0, :] - s[1:,])) <= convergence_xtol_abs
        ):
            optimal = True
            converge = True
            reason_to_stop = "Termination codition satisfied"
        elif (
            iterations >= stopping_maxiter
        ):  # if maximum amount of iteration is exceeded
            optimal = True
            converge = False
            reason_to_stop = "Maximum number of interation exceeded"

    result = {
        "solution_x": s[np.nonzero(f_s == f_s.min())[0][0], :],
        "solution_criterion": f_s.min(),
        "n_iterations": iterations,
        "success": converge,
        "reached_convergence_criterion": reason_to_stop,
    }
    return result


def _init_algo_params(adaptive, j):
    if adaptive:
        return (
            1,
            1 + 2 / j,
            0.75 - 1 / (2 * j),
            1 - 1 / j,
        )
    else:
        return (
            1,
            2,
            0.5,
            0.5,
        )


def _init_simplex(x):
    s = np.vstack(
        [
            x,
        ]
        * (len(x) + 1)
    ).astype(np.float64)

    return s


def _pfeffer(x):
    s = _init_simplex(x)

    c_p = 1.05

    np.fill_diagonal(s[1:, :], x * c_p * (x != 0) + 0.00025 * (x == 0))

    return s


def _nash(x):
    s = _init_simplex(x)

    c_n = 0.1

    np.fill_diagonal(s[1:, :], (x != 0) * (np.max(x) * c_n + x) + c_n * (x == 0))
    return s


def _gao_han(x):
    s = _init_simplex(x)

    c_h = np.minimum(np.maximum(np.max(x), 1), 10)
    j = len(x)

    s = (
        s
        + np.vstack(
            [
                np.array([[(1 - (j + 1) ** 0.5) / j]]) * np.ones([1, j]),
                np.eye(j),
            ]
        )
        * c_h
    )

    return s


def _varadhan_borchers(x):
    s = _init_simplex(x)

    j = len(x)
    c_s = np.maximum(1, ((x**2).sum()) ** 0.5)
    beta1 = c_s / (j * 2**0.5) * ((j + 1) ** 0.5 + j - 1)
    beta2 = c_s / (j * 2**0.5) * ((j + 1) ** 0.5 - 1)

    s[1:, :] = s[1:, :] + np.full([j, j], beta2) + np.eye(j) * (beta1 - beta2)

    return s
