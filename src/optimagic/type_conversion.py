from typing import Any

from optimagic.typing import (
    GtOneFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


def _process_float_like(value: Any) -> float:
    pass


def _process_int_like(value: Any) -> int:
    pass


def _process_positive_int_like(value: Any) -> PositiveInt:
    pass


def _process_non_negative_int_like(value: Any) -> NonNegativeInt:
    pass


def _process_positive_float_like(value: Any) -> PositiveFloat:
    pass


def _process_non_negative_float_like(value: Any) -> NonNegativeFloat:
    pass


def _process_gt_one_float_like(value: Any) -> GtOneFloat:
    pass


def _process_bool_like(value: Any) -> bool:
    pass


TYPE_CONVERTERS = {
    float: _process_float_like,
    int: _process_int_like,
    bool: _process_bool_like,
    PositiveInt: _process_positive_int_like,
    NonNegativeInt: _process_non_negative_int_like,
    PositiveFloat: _process_positive_float_like,
    NonNegativeFloat: _process_non_negative_float_like,
    GtOneFloat: _process_gt_one_float_like,
}
