import sys
from traceback import format_exception


class OptimagicError(Exception):
    pass


class TableExistsError(OptimagicError):
    pass


class InvalidFunctionError(OptimagicError):
    pass


class UserFunctionRuntimeError(OptimagicError):
    pass


class MissingInputError(OptimagicError):
    pass


class AliasError(OptimagicError):
    pass


class InvalidKwargsError(OptimagicError):
    pass


class InvalidParamsError(OptimagicError):
    pass


class InvalidConstraintError(OptimagicError):
    pass


class InvalidBoundsError(OptimagicError):
    pass


class IncompleteBoundsError(OptimagicError):
    pass


class InvalidScalingError(OptimagicError):
    pass


class InvalidMultistartError(OptimagicError):
    pass


class InvalidNumdiffOptionsError(OptimagicError):
    pass


class NotInstalledError(OptimagicError):
    pass


class NotAvailableError(OptimagicError):
    pass


class InvalidAlgoOptionError(OptimagicError):
    pass


class InvalidAlgoInfoError(OptimagicError):
    pass


class InvalidPlottingBackendError(OptimagicError):
    pass


class StopOptimizationError(OptimagicError):
    def __init__(self, message, current_status):
        super().__init__(message)
        self.message = message
        self.current_status = current_status

    def __reduce__(self):
        """Taken from here: https://tinyurl.com/y6eeys2f."""
        return (StopOptimizationError, (self.message, self.current_status))


def get_traceback():
    tb = format_exception(*sys.exc_info())
    if isinstance(tb, list):
        tb = "".join(tb)
    return tb


INVALID_INFERENCE_MSG = (
    "Taking the inverse of the information matrix failed. Only ever use this "
    "covariance matrix or standard errors based on it for diagnostic purposes, not for "
    "drawing conclusions."
)


INVALID_SENSITIVITY_MSG = (
    "Taking inverse failed during the calculation of sensitvity measures. Interpret "
    "them with caution."
)
