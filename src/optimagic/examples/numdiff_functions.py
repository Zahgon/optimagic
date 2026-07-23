
import numpy as np
from scipy.stats import norm

FLOAT_EPS = np.finfo(float).eps



def logit_loglike(params, y, x):
    return logit_loglikeobs(params, y, x).sum()


def logit_loglikeobs(params, y, x):
    q = 2 * y - 1
    return np.log(1 / (1 + np.exp(-(q * np.dot(x, params)))))


def logit_loglike_gradient(params, y, x):
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return np.dot(y - c, x)


def logit_loglikeobs_jacobian(params, y, x):
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return (y - c).reshape(-1, 1) * x


def logit_loglike_hessian(params, y, x):  # noqa: ARG001
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return -np.dot(c * (1 - c) * x.T, x)




def probit_loglike(params, y, x):
    pass


def probit_loglikeobs(params, y, x):
    pass


def probit_loglike_gradient(params, y, x):
    pass


def probit_loglikeobs_jacobian(params, y, x):
    pass


def probit_loglike_hessian(params, y, x):
    pass
