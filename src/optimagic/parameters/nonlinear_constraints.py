import itertools
from dataclasses import asdict
from functools import partial

import numpy as np
import pandas as pd
from pybaum import tree_flatten, tree_just_flatten, tree_unflatten

from optimagic.differentiation.derivatives import first_derivative
from optimagic.exceptions import InvalidConstraintError, InvalidFunctionError
from optimagic.optimization.algo_options import CONSTRAINTS_ABSOLUTE_TOLERANCE
from optimagic.parameters.block_trees import block_tree_to_matrix
from optimagic.parameters.tree_registry import get_registry


def process_nonlinear_constraints(
    nonlinear_constraints,
    params,
    bounds,
    converter,
    numdiff_options,
    skip_checks,
):
    """Process and prepare nonlinear constraints for internal use.

    A user-provided nonlinear constraint consists of a function that is evaluated on a
    selection of parameters returning a scalar or vector that must either be equal to
    a fixed value (equality constraint) or smaller and larger than or equal to a lower
    and upper bound (inequality constraint).

    This function processes the nonlinear constraints in the following way:

    1. The constraint a <= g(x) <= b is transformed to h(x) >= 0, where h(x) is
       - h(x) = g(x), if a == 0 and b == inf
       - h(x) = g(x) - a, if a != 0 and b == inf
       - h(x) = (g(x) - a, -g(x) + b) >= 0, if a != 0 and b != inf.

    2. The equality constraint g(x) = v is transformed to h(x) >= 0, where
       h(x) = (g(x) - v, -g(x) + v).

    3. Vector constraints are transformed to a list of scalar constraints.
       g(x) = (g1(x), g2(x), ...) >= 0 is transformed to (g1(x) >= 0, g2(x) >= 0, ...).

    4. The constraint function (defined on a selection of user-facing parameters) is
       transformed to be evaluated on the internal parameters.


    Args:
        nonlinear_constraints (list[dict]): List of dictionaries, each representing a
            nonlinear constraint.
        params (pandas): A pytree containing the parameters with respect to which the
            criterion is optimized. Examples are a numpy array, a pandas Series,
            a DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        bounds (Bounds): Bounds object containing information on the bounds of the
            parameters. See :ref:`bounds` for details.
        converter (Converter): NamedTuple with methods to convert between internal and
            external parameters, derivatives and function outputs.
        numdiff_options (NumdiffOptions): Options for numerical derivatives. See
            :ref:`first_derivative` for details. Note that the default method is changed
            to "forward" for speed reasons.
        skip_checks (bool): Whether checks on the inputs are skipped. This makes the
            optimization faster, especially for very fast constraint functions. Default
            False.

    Returns:
        list[dict]: List of processed constraints.

    """
    constraint_evals = []
    for _constraint in nonlinear_constraints:
        _eval = _check_validity_and_return_evaluation(_constraint, params, skip_checks)
        constraint_evals.append(_eval)

    processed = []
    for _constraint, _eval in zip(
        nonlinear_constraints, constraint_evals, strict=False
    ):
        _processed_constraint = _process_nonlinear_constraint(
            _constraint,
            constraint_eval=_eval,
            params=params,
            bounds=bounds,
            converter=converter,
            numdiff_options=numdiff_options,
        )
        processed.append(_processed_constraint)

    return processed


def _process_nonlinear_constraint(
    c, constraint_eval, params, bounds, converter, numdiff_options
):
    """Process a single nonlinear constraint."""

    external_selector = _process_selector(c)  # functional selector

    constraint_func = c["func"]

    if constraint_eval is None:
        selected = external_selector(params)
        constraint_eval = constraint_func(selected)

    if bounds is not None:

        constraint_bounds = None
    else:
        constraint_bounds = None

    _n_constr = len(np.atleast_1d(constraint_eval))



    if "derivative" in c:
        if not callable(c["derivative"]):
            msg = "Jacobian of constraints needs to be callable."
            raise ValueError(msg)
        jacobian = c["derivative"]
    else:
        def jacobian(p):
            return first_derivative(
                constraint_func,
                p,
                bounds=constraint_bounds,
                error_handling="raise_strict",
                **asdict(numdiff_options),
            ).derivative

    selection_indices, n_params = _get_selection_indices(params, external_selector)

    def _internal_jacobian(x):
        pass

    _type = "eq" if "value" in c else "ineq"

    if _type == "eq":

        _value = np.atleast_1d(np.array(c["value"], dtype=float))

        def internal_constraint_func(x):
            pass

        jacobian_from_internal = _internal_jacobian
        n_constr = _n_constr

    else:

        def _internal_constraint_func(x):
            pass

        lower_bounds = c.get("lower_bounds", 0)
        upper_bounds = c.get("upper_bounds", np.inf)

        transformation = _get_transformation(lower_bounds, upper_bounds)

        internal_constraint_func = _compose_funcs(
            _internal_constraint_func, transformation["func"]
        )

        jacobian_from_internal = _compose_funcs(
            _internal_jacobian, transformation["derivative"]
        )

        n_constr = 2 * _n_constr if transformation["name"] == "stack" else _n_constr

    internal_constr = {
        "n_constr": n_constr,
        "type": _type,
        "fun": internal_constraint_func,  # internal name for 'func'
        "jac": jacobian_from_internal,  # internal name for 'derivative'
        "tol": c.get("tol", CONSTRAINTS_ABSOLUTE_TOLERANCE),
    }

    return internal_constr


def equality_as_inequality_constraints(nonlinear_constraints):
    """Return constraints where equality constraints are converted to inequality."""
    constraints = [_equality_to_inequality(c) for c in nonlinear_constraints]
    return constraints


def _equality_to_inequality(c):
    """Transform a single constraint.

    An equality constaint g(x) = 0 can be transformed to two inequality constraints
    using (g(x), -g(x)) >= 0. Hence, the number of constraints doubles, and the
    constraint functions itself as well as the derivative need to be updated.

    """
    if c["type"] == "eq":

        def transform(x, func):
            pass

        out = {
            "fun": partial(transform, func=c["fun"]),
            "jac": partial(transform, func=c["jac"]),
            "n_constr": 2 * c["n_constr"],
            "tol": c["tol"],
            "type": "ineq",
        }
    else:
        out = c
    return out


def vector_as_list_of_scalar_constraints(nonlinear_constraints):
    """Return constraints where vector constraints are converted to scalar constraints.

    This is necessary for internal optimizers that only support scalar constraints.

    """
    list_of_constraints_lists = [
        _vector_to_list_of_scalar(c) for c in nonlinear_constraints
    ]
    constraints = list(itertools.chain.from_iterable(list_of_constraints_lists))
    return constraints


def _vector_to_list_of_scalar(constraint):
    if constraint["n_constr"] > 1:
        out = []
        for k in range(constraint["n_constr"]):
            c = constraint.copy()
            fun, jac = _get_components(constraint["fun"], constraint["jac"], idx=k)
            c["fun"] = fun
            c["jac"] = jac
            c["n_constr"] = 1
            out.append(c)
    else:
        out = [constraint]
    return out


def _get_components(fun, jac, idx):
    """Return function and derivative for a single component of a vector function.

    Args:
        fun (callable): Function that returns a vector.
        jac (callable): Derivative of the function that returns a matrix.
        idx (int): Index of the component.

    Returns:
        callable: Component function at index idx.
        callable: Jacobian of the component function.

    """
    fun_component = lambda x: fun(x)[idx]
    jac_component = lambda x: jac(x)[idx]
    return fun_component, jac_component




def _process_selector(c):
    if "selector" in c:
        selector = c["selector"]
    elif "loc" in c:

        def selector(params):
            return params.loc[c["loc"]]

    elif "query" in c:

        def selector(params):
            return params.query(c["query"])

    else:
        selector = _identity
    return selector


def _compose_funcs(f, g):
    return lambda x: g(f(x))


def _identity(x):
    pass




def _extend_jacobian(jac_mat, selection_indices, n_params):
    """Extend Jacobian on selected parameters to full params.

    Jacobian of constraints is defined on a selection of the parameters, however, we
    need the Jacobian on the full params. Since the Jacobian is trivially zero at the
    non-selected params we can simply fill a zero matrix.

    """
    jac_extended = np.zeros((jac_mat.shape[0], n_params))
    jac_extended[:, selection_indices] = jac_mat
    return jac_extended


def _get_selection_indices(params, selector):
    """Get index of selected flat params and number of flat params."""
    registry = get_registry(extended=True)
    flat_params, params_treedef = tree_flatten(params, registry=registry)
    n_params = len(flat_params)
    indices = np.arange(n_params, dtype=int)
    params_indices = tree_unflatten(params_treedef, indices, registry=registry)
    selected = selector(params_indices)
    selection_indices = np.array(
        tree_just_flatten(selected, registry=registry), dtype=int
    )
    return selection_indices, n_params




def _get_transformation(lower_bounds, upper_bounds):
    """Get transformation given bounds.

    The internal inequality constraint is defined as h(x) >= 0. However, the user can
    specify: a <= g(x) <= b. To get the internal represenation we need to transform the
    constraint.

    """
    transformation_type = _get_transformation_type(lower_bounds, upper_bounds)

    if transformation_type == "identity":
        transformer = {"func": _identity, "derivative": _identity}
    elif transformation_type == "subtract_lb":
        transformer = {
            "func": lambda v: v - lower_bounds,
            "derivative": _identity,
        }
    elif transformation_type == "stack":
        transformer = {
            "func": lambda v: np.concatenate(
                (v - lower_bounds, upper_bounds - v), axis=0
            ),
            "derivative": lambda v: np.concatenate((v, -v), axis=0),
        }
    transformer["name"] = transformation_type
    return transformer


def _get_transformation_type(lower_bounds, upper_bounds):
    lb_is_zero = not np.count_nonzero(lower_bounds)
    ub_is_inf = np.all(np.isposinf(upper_bounds))

    if lb_is_zero and ub_is_inf:
        _transformation_type = "identity"
    elif ub_is_inf:
        _transformation_type = "subtract_lb"
    else:
        _transformation_type = "stack"
    return _transformation_type




def _check_validity_and_return_evaluation(c, params, skip_checks):
    """Check that nonlinear constraints are valid.

    Returns:
        constaint_eval: Evaluation of constraint at params, if skip_checks if False,
            else None.

    """

    if "func" not in c:
        raise InvalidConstraintError(
            "Constraint needs to have entry 'fun', representing the constraint "
            "function."
        )
    if not callable(c["func"]):
        raise InvalidConstraintError(
            "Entry 'fun' in nonlinear constraints has be callable."
        )

    if "derivative" in c and not callable(c["derivative"]):
        raise InvalidConstraintError(
            "Entry 'jac' in nonlinear constraints has be callable."
        )


    is_equality_constraint = "value" in c

    if is_equality_constraint:
        if "lower_bounds" in c or "upper_bounds" in c:
            raise InvalidConstraintError(
                "Only one of 'value' or ('lower_bounds', 'upper_bounds') can be "
                "passed to a nonlinear constraint."
            )

    if not is_equality_constraint:
        if "lower_bounds" not in c and "upper_bounds" not in c:
            raise InvalidConstraintError(
                "For inequality constraint at least one of ('lower_bounds', "
                "'upper_bounds') has to be passed to the nonlinear constraint."
            )

    if "lower_bounds" in c and "upper_bounds" in c:
        if not np.all(np.array(c["lower_bounds"]) <= np.array(c["upper_bounds"])):
            raise InvalidConstraintError(
                "If lower bounds need to less than or equal to upper bounds."
            )


    if "selector" in c:
        if not callable(c["selector"]):
            raise InvalidConstraintError(
                f"'selector' entry needs to be callable in constraint {c}."
            )
        else:
            try:
                c["selector"](params)
            except Exception as e:
                raise InvalidFunctionError(
                    "Error when calling 'selector' function on params in constraint "
                    f" {c}"
                ) from e

    elif "loc" in c:
        if not isinstance(params, (pd.Series, pd.DataFrame)):
            raise InvalidConstraintError(
                "params needs to be pd.Series or pd.DataFrame to use 'loc' selector in "
                f"in consrtaint {c}."
            )
        try:
            params.loc[c["loc"]]
        except (KeyError, IndexError) as e:
            raise InvalidConstraintError("'loc' string is invalid.") from e

    elif "query" in c:
        if not isinstance(params, pd.DataFrame):
            raise InvalidConstraintError(
                "params needs to be pd.DataFrame to use 'query' selector in "
                f"constraints {c}."
            )
        try:
            params.query(c["query"])
        except Exception as e:
            raise InvalidConstraintError(
                f"'query' string is invalid in constraint {c}."
            ) from e


    constraint_eval = None

    if not skip_checks:
        selector = _process_selector(c)

        try:
            constraint_eval = c["func"](selector(params))
        except Exception as e:
            raise InvalidFunctionError(
                f"Error when evaluating function of constraint {c}."
            ) from e

    return constraint_eval
