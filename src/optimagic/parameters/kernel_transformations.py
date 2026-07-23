
import numpy as np

from optimagic.utilities import (
    chol_params_to_lower_triangular_matrix,
    cov_matrix_to_sdcorr_params,
    cov_params_to_matrix,
    dimension_to_number_of_triangular_elements,
    robust_cholesky,
    sdcorr_params_to_matrix,
)


def covariance_to_internal(external_values, constr):
    pass


def covariance_to_internal_jacobian(external_values, constr):
    pass


def covariance_from_internal(internal_values, constr):
    pass


def covariance_from_internal_jacobian(internal_values, constr):
    pass


def sdcorr_to_internal(external_values, constr):
    pass


def sdcorr_to_internal_jacobian(external_values, constr):
    pass


def sdcorr_from_internal(internal_values, constr):
    pass


def sdcorr_from_internal_jacobian(internal_values, constr):
    pass


def probability_to_internal(external_values, constr):
    pass


def probability_to_internal_jacobian(external_values, constr):
    pass


def probability_from_internal(internal_values, constr):
    pass


def probability_from_internal_jacobian(internal_values, constr):
    pass


def linear_to_internal(external_values, constr):
    pass


def linear_to_internal_jacobian(external_values, constr):
    pass


def linear_from_internal(internal_values, constr):
    pass


def linear_from_internal_jacobian(internal_values, constr):
    pass


def _elimination_matrix(dim):
    pass


def _duplication_matrix(dim):
    pass


def _transformation_matrix(dim):
    pass


def _commutation_matrix(dim):
    pass


def _unit_vector_or_zeros(index, size):
    pass
