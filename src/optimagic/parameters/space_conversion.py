
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

import optimagic.parameters.kernel_transformations as kt
from optimagic.parameters.process_constraints import process_constraints


def get_space_converter(
    internal_params,
    internal_constraints,
):
    """Get functions to convert between in-/external space of params and derivatives.

    In the internal parameter space the optimization problem is unconstrained except
    for bounds.

    Args:
        internal_params (InternalParams): Dataclass with internal parameter values and
            bounds.
        internal_constraints (list): List of constraints with processed selector fields.

    Returns:
        SpaceConverter: The space converter.
        InternalParams: Dataclass with entries:
            - value (np.ndarray): Internal parameter values.
            - lower_bounds (np.ndarray | None): Lower bounds on the internal params.
            - upper_bounds (np.ndarray | None): Upper bounds on the internal params.
            - soft_lower_bounds (np.ndarray | None): Soft lower bounds on the internal
              params.
            - soft_upper_bounds (np.ndarray | None): Soft upper bounds on the internal
              params.
            - name (list): List of names of the external parameters.
            - free_mask (np.ndarray): Boolean mask representing which external parameter
              is free.

    """
    transformations, constr_info = process_constraints(
        constraints=internal_constraints,
        params_vec=internal_params.values,
        lower_bounds=internal_params.lower_bounds,
        upper_bounds=internal_params.upper_bounds,
        param_names=internal_params.names,
    )
    _params_to_internal = partial(
        reparametrize_to_internal,
        internal_free=constr_info["internal_free"],
        transformations=transformations,
    )

    _params_from_internal = partial(
        reparametrize_from_internal,
        fixed_values=constr_info["internal_fixed_values"],
        pre_replacements=constr_info["pre_replacements"],
        transformations=transformations,
        post_replacements=constr_info["post_replacements"],
    )

    _dim_internal = int(constr_info["internal_free"].sum())

    _pre_replace_jac = pre_replace_jacobian(
        pre_replacements=constr_info["pre_replacements"], dim_in=_dim_internal
    )

    _post_replace_jac = post_replace_jacobian(
        post_replacements=constr_info["post_replacements"]
    )

    _derivative_to_internal = partial(
        convert_external_derivative_to_internal,
        fixed_values=constr_info["internal_fixed_values"],
        pre_replacements=constr_info["pre_replacements"],
        transformations=transformations,
        pre_replace_jac=_pre_replace_jac,
        post_replace_jac=_post_replace_jac,
    )

    _has_transforming_constraints = bool(transformations)

    converter = SpaceConverter(
        params_to_internal=_params_to_internal,
        params_from_internal=_params_from_internal,
        derivative_to_internal=_derivative_to_internal,
        has_transforming_constraints=_has_transforming_constraints,
    )

    free_mask = constr_info["internal_free"]
    if (
        internal_params.soft_lower_bounds is not None
        and not _has_transforming_constraints
    ):
        _soft_lower = internal_params.soft_lower_bounds[free_mask]
    else:
        _soft_lower = None

    if (
        internal_params.soft_upper_bounds is not None
        and not _has_transforming_constraints
    ):
        _soft_upper = internal_params.soft_upper_bounds[free_mask]
    else:
        _soft_upper = None

    params = InternalParams(
        values=converter.params_to_internal(internal_params.values),
        lower_bounds=constr_info["lower_bounds"],
        upper_bounds=constr_info["upper_bounds"],
        names=internal_params.names,
        free_mask=free_mask,
        soft_lower_bounds=_soft_lower,
        soft_upper_bounds=_soft_upper,
    )
    return converter, params


@dataclass(frozen=True)
class SpaceConverter:
    params_to_internal: Callable
    params_from_internal: Callable
    derivative_to_internal: Callable
    has_transforming_constraints: bool


def reparametrize_to_internal(
    external,
    internal_free,
    transformations,
):
    pass


def reparametrize_from_internal(
    internal,
    fixed_values,
    pre_replacements,
    transformations,
    post_replacements,
):
    pass


def convert_external_derivative_to_internal(
    external_derivative,
    internal_values,
    fixed_values,
    pre_replacements,
    transformations,
    post_replacements=None,
    pre_replace_jac=None,
    post_replace_jac=None,
):
    pass


def _multiply_from_left(mat_list):
    """Multiply all matrices in the list, starting from the left.

    Note that this only affects the order in which the pairwise multiplications happen,
    not the actual result.

    """
    out = mat_list[0]
    for mat in mat_list[1:]:
        out = out @ mat
    return out


def _multiply_from_right(mat_list):
    """Multiply all matrices in the list, starting from the right.

    Note that this only affects the order in which the pairwise multiplications happen,
    not the actual result.

    """
    out = mat_list[-1]
    for mat in reversed(mat_list[:-1]):
        out = mat @ out
    return out


def pre_replace(internal_values, fixed_values, pre_replacements):
    pass


def pre_replace_jacobian(pre_replacements, dim_in):
    """Return Jacobian of pre-replacement step.

    Remark. The function ``pre_replace`` can have ``np.nan`` in its output. In
    this case we know from the underlying structure that the derivative of this
    output with respect to any of the inputs is zero. Here we use this additional
    knowledge; however, when the derivative is computed using a numerical
    differentiation technique this will not be the case. Thus the numerical
    derivative can differ from the derivative here in these cases.

    Args:
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.
        dim_in (int): Dimension of the internal parameters.

    Returns:
        jacobian (np.ndarray): The jacobian.

    Examples:
        >>> # Note: The example is the same as in the doctest of pre_replace
        >>> pre_replacements = np.array([1, -1, 0])
        >>> pre_replace_jacobian(pre_replacements, 2)
        array([[0., 1.],
               [0., 0.],
               [1., 0.]])

    """
    dim_out = len(pre_replacements)
    mask = pre_replacements >= 0
    position_in = pre_replacements[mask]
    position_out = np.arange(dim_out)[mask]

    jacobian = np.zeros((dim_out, dim_in))
    jacobian[position_out, position_in] = 1
    return jacobian


def transformation_jacobian(transformations, pre_replaced):
    pass


def post_replace(external_values, post_replacements):
    pass


def post_replace_jacobian(post_replacements):
    """Return Jacobian of post-replacement step.

    Args:
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.
        dim (int): The dimension of the external parameters.

    Returns:
        jacobian (np.ndarray): The Jacobian.

    Examples:
        >>> # Note: the example is the same as in the doctest of post_replace
        >>> post_replacements = np.array([-1, -1, 1])
        >>> post_replace_jacobian(post_replacements)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.]])

    """
    dim = len(post_replacements)
    mask = post_replacements >= 0
    positions_in = post_replacements[mask]
    positions_out = np.arange(dim)[mask]

    jacobian = np.eye(dim)
    jacobian[positions_out, :] *= 0
    jacobian[positions_out, positions_in] = 1
    return jacobian


@dataclass(frozen=True)
class InternalParams:
    values: np.ndarray
    lower_bounds: np.ndarray | None
    upper_bounds: np.ndarray | None
    soft_lower_bounds: np.ndarray | None = None
    soft_upper_bounds: np.ndarray | None = None
    names: list | None = None
    free_mask: np.ndarray | None = None
