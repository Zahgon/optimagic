import numpy as np


def _standard_logistic(size, rng):
    pass


def _standard_uniform(size, rng):
    pass


def _standard_normal(size, rng):
    pass


def _standard_gumbel(size, rng):
    pass


def _standard_laplace(size, rng):
    pass


NOISE_DISTRIBUTIONS = {
    "normal": _standard_normal,
    "gumbel": _standard_gumbel,
    "logistic": _standard_logistic,
    "uniform": _standard_uniform,
    "laplace": _standard_laplace,
}
