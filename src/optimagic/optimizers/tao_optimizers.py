
import functools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PETSC4PY_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_GTOL_SCALED,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt
from optimagic.utilities import calculate_trustregion_initial_radius


@mark.minimizer(
    name="tao_pounders",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=IS_PETSC4PY_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class TAOPounders(Algorithm):

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    convergence_gtol_scaled: NonNegativeFloat = CONVERGENCE_GTOL_SCALED
    trustregion_initial_radius: NonNegativeFloat | None = None
    stopping_maxiter: PositiveInt = STOPPING_MAXITER

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        raw = tao_pounders(
            criterion=problem.fun,
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            convergence_gtol_abs=self.convergence_gtol_abs,
            convergence_gtol_rel=self.convergence_gtol_rel,
            convergence_gtol_scaled=self.convergence_gtol_scaled,
            trustregion_initial_radius=self.trustregion_initial_radius,
            stopping_maxiter=self.stopping_maxiter,
        )

        res = InternalOptimizeResult(
            x=raw["solution_x"],
            fun=raw["solution_criterion"],
            success=raw["success"],
            message=raw["message"],
            n_fun_evals=raw["n_fun_evals"],
            n_jac_evals=0,
            n_hess_evals=0,
            n_iterations=raw["n_iterations"],
            info={
                "gradient_norm": raw["gradient_norm"],
                "criterion_norm": raw["criterion_norm"],
                "convergence_code": raw["convergence_code"],
                "convergence_reason": raw["reached_convergence_criterion"],
            },
        )

        return res


def tao_pounders(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    convergence_gtol_rel=CONVERGENCE_GTOL_REL,
    convergence_gtol_scaled=CONVERGENCE_GTOL_SCALED,
    trustregion_initial_radius=None,
    stopping_maxiter=STOPPING_MAXITER,
):
    r"""Minimize a function using the POUNDERs algorithm.

    For details see
    :ref: `tao_algorithm`.

    """
    if not IS_PETSC4PY_INSTALLED:
        raise NotInstalledError(
            "The 'tao_pounders' algorithm requires the petsc4py package to be "
            "installed. If you are using Linux or MacOS, install the package with "
            "'conda install -c conda-forge petsc4py'. The package is not available on "
            "Windows. Windows users can use optimagics 'pounders' algorithm instead."
        )
    from petsc4py import PETSc

    first_eval = criterion(x)
    n_errors = len(first_eval)
    _x = _initialise_petsc_array(x)
    residuals_out = _initialise_petsc_array(n_errors)

    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    tao.setType("pounders")

    tao.setFromOptions()

    def func_tao(tao, x, resid_out):  # noqa: ARG001
        pass

    tao.setResidual(func_tao, residuals_out)

    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(_x)
    elif trustregion_initial_radius <= 0:
        raise ValueError("The initial trust region radius must be > 0.")
    tao.setInitialTrustRegionRadius(trustregion_initial_radius)

    if lower_bounds is not None or upper_bounds is not None:
        if lower_bounds is None:
            lower_bounds = np.full(len(x), -np.inf)
        if upper_bounds is None:
            upper_bounds = np.full(len(x), np.inf)
        lower_bounds = _initialise_petsc_array(lower_bounds)
        upper_bounds = _initialise_petsc_array(upper_bounds)
        tao.setVariableBounds(lower_bounds, upper_bounds)

    tao.setInitial(_x)

    default_gatol = convergence_gtol_abs if convergence_gtol_abs else -1
    default_gttol = convergence_gtol_scaled if convergence_gtol_scaled else -1
    default_grtol = convergence_gtol_rel if convergence_gtol_rel else -1
    tao.setTolerances(
        gatol=default_gatol,
        grtol=default_grtol,
        gttol=default_gttol,
    )

    if stopping_maxiter is not None:
        tao.setConvergenceTest(functools.partial(_max_iters, stopping_maxiter))
    elif convergence_gtol_scaled is False and convergence_gtol_abs is False:
        tao.setConvergenceTest(functools.partial(_grtol_conv, convergence_gtol_rel))
    elif convergence_gtol_rel is False and convergence_gtol_scaled is False:
        tao.setConvergenceTest(functools.partial(_gatol_conv, convergence_gtol_abs))
    elif convergence_gtol_scaled is False:
        tao.setConvergenceTest(
            functools.partial(
                _grtol_gatol_conv,
                convergence_gtol_rel,
                convergence_gtol_abs,
            )
        )

    tao.solve()

    results = _process_pounders_results(residuals_out, tao)

    petsc_bounds = [b for b in (lower_bounds, upper_bounds) if b is not None]
    for obj in [tao, _x, residuals_out, *petsc_bounds]:
        obj.destroy()

    return results


def _initialise_petsc_array(len_or_array):
    """Initialize an empty array or fill in provided values.

    Args:
        len_or_array (int or numpy.ndarray): If the value is an integer, allocate an
            empty array with the given length. If the value is an array, allocate an
            array of equal length and fill in the values.

    """
    from petsc4py import PETSc

    length = len_or_array if isinstance(len_or_array, int) else len(len_or_array)

    array = PETSc.Vec().create(PETSc.COMM_WORLD)
    array.setSizes(length)
    array.setFromOptions()

    if isinstance(len_or_array, np.ndarray):
        array.array = len_or_array

    return array


def _max_iters(max_iterations, tao):
    pass


def _gatol_conv(absolute_gradient_tolerance, tao):
    pass


def _grtol_conv(relative_gradient_tolerance, tao):
    pass


def _grtol_gatol_conv(relative_gradient_tolerance, absolute_gradient_tolerance, tao):
    pass


def _translate_tao_convergence_reason(tao_resaon):
    mapping = {
        3: "absolute_gradient_tolerance below critical value",
        4: "relative_gradient_tolerance below critical value",
        5: "scaled_gradient_tolerance below critical value",
        6: "step size small",
        7: "objective below min value",
        8: "user defined",
        -2: "maxits reached",
        -4: "numerical problems",
        -5: "max funcevals reached",
        -6: "line search failure",
        -7: "trust region failure",
        -8: "user defined",
    }
    return mapping[tao_resaon]


def _process_pounders_results(residuals_out, tao):
    convergence_code = tao.getConvergedReason()
    convergence_reason = _translate_tao_convergence_reason(convergence_code)

    results = {
        "solution_x": tao.solution.array,
        "solution_criterion": tao.function,
        "solution_derivative": None,
        "solution_hessian": None,
        "n_fun_evals": tao.getIterationNumber(),
        "n_jac_evals": None,
        "n_iterations": None,
        "success": bool(convergence_code >= 0),
        "reached_convergence_criterion": (
            convergence_reason if convergence_code >= 0 else None
        ),
        "message": convergence_reason,
        "solution_criterion_values": residuals_out.array,
        "gradient_norm": tao.gnorm,
        "criterion_norm": tao.cnorm,
        "convergence_code": convergence_code,
    }

    return results
