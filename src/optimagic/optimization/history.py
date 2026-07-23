import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pybaum import leaf_names, tree_just_flatten

from optimagic.parameters.tree_registry import get_registry
from optimagic.timing import CostModel
from optimagic.typing import Direction, EvalTask, PyTree


@dataclass(frozen=True)
class HistoryEntry:
    params: PyTree
    fun: float | None
    start_time: float
    stop_time: float
    task: EvalTask


class History:
    def __init__(
        self,
        direction: Direction,
        params: list[PyTree] | None = None,
        fun: list[float | None] | None = None,
        start_time: list[float] | None = None,
        stop_time: list[float] | None = None,
        batches: list[int] | None = None,
        task: list[EvalTask] | None = None,
    ) -> None:
        """Initialize a history.

        The history must know the direction of the optimization problem in order to
        correctly return monotone sequences. The history can be initialized empty, for
        example for usage during an optimization process, or with data, for example to
        recover a history from a log.

        """
        _validate_args_are_all_none_or_lists_of_same_length(
            params, fun, start_time, stop_time, batches, task
        )

        self.direction = direction
        self._params = params if params is not None else []
        self._fun = fun if fun is not None else []
        self._start_time = start_time if start_time is not None else []
        self._stop_time = stop_time if stop_time is not None else []
        self._batches = batches if batches is not None else []
        self._task = task if task is not None else []


    def add_entry(self, entry: HistoryEntry, batch_id: int | None = None) -> None:
        if batch_id is None:
            batch_id = self._get_next_batch_id()
        self._params.append(entry.params)
        self._fun.append(entry.fun)
        self._start_time.append(entry.start_time)
        self._stop_time.append(entry.stop_time)
        self._batches.append(batch_id)
        self._task.append(entry.task)

    def add_batch(
        self, batch: list[HistoryEntry], batch_size: int | None = None
    ) -> None:
        if batch_size is None:
            batch_size = len(batch)

        start = self._get_next_batch_id()
        n_batches = int(np.ceil(len(batch) / batch_size))
        ids = np.repeat(np.arange(start, start + n_batches), batch_size)[: len(batch)]

        for entry, id in zip(batch, ids, strict=False):
            self.add_entry(entry, id)

    def _get_next_batch_id(self) -> int:
        if not self._batches:
            batch = 0
        else:
            batch = self._batches[-1] + 1
        return batch



    def fun_data(self, cost_model: CostModel, monotone: bool = False) -> pd.DataFrame:
        pass

    @property
    def fun(self) -> list[float | None]:
        return self._fun

    @property
    def monotone_fun(self) -> NDArray[np.float64]:
        pass


    @property
    def is_accepted(self) -> NDArray[np.bool_]:
        pass


    def params_data(
        self, dropna: bool = False, collapse_batches: bool = False
    ) -> pd.DataFrame:
        pass

    @property
    def params(self) -> list[PyTree]:
        pass

    @property
    def flat_params(self) -> list[list[float]]:
        pass

    @property
    def flat_param_names(self) -> list[str]:
        pass


    def _get_total_timings(
        self, cost_model: CostModel | Literal["wall_time"]
    ) -> NDArray[np.float64]:
        pass

    def _get_timings_per_task(
        self, task: EvalTask, cost_factor: float | None
    ) -> NDArray[np.float64]:
        pass

    @property
    def start_time(self) -> list[float]:
        pass

    @property
    def stop_time(self) -> list[float]:
        pass


    @property
    def batches(self) -> list[int]:
        pass

    def _is_serial(self) -> bool:
        pass


    @property
    def task(self) -> list[EvalTask]:
        pass


    @property
    def time(self) -> list[float]:
        pass

    @property
    def criterion(self) -> list[float | None]:
        msg = "The attribute `criterion` of History is deprecated. Use `fun` instead."
        warnings.warn(msg, FutureWarning)
        return self.fun

    @property
    def runtime(self) -> list[float]:
        pass

    def __getitem__(self, key: str) -> Any:
        msg = "dict-like access to History is deprecated. Use attribute access instead."
        warnings.warn(msg, FutureWarning)
        return getattr(self, key)




def _get_flat_params(params: list[PyTree]) -> list[list[float]]:
    fast_path = len(params) > 0 and _is_1d_array(params[0])
    if fast_path:
        flatten = lambda x: x.tolist()
    else:
        registry = get_registry(extended=True)
        flatten = partial(tree_just_flatten, registry=registry)

    return [flatten(p) for p in params]


def _get_flat_param_names(param: PyTree) -> list[str]:
    fast_path = _is_1d_array(param)
    if fast_path:
        return np.arange(param.size).astype(str).tolist()

    registry = get_registry(extended=True)
    return leaf_names(param, registry=registry)


def _is_1d_array(param: PyTree) -> bool:
    return isinstance(param, np.ndarray) and param.ndim == 1


def _calculate_monotone_sequence(
    sequence: list[float | None], direction: Direction
) -> NDArray[np.float64]:
    sequence_arr = np.array(sequence, dtype=np.float64)  # converts None to nan
    nan_mask = np.isnan(sequence_arr)

    if direction == Direction.MINIMIZE:
        sequence_arr[nan_mask] = np.inf
        out = np.minimum.accumulate(sequence_arr)
    elif direction == Direction.MAXIMIZE:
        sequence_arr[nan_mask] = -np.inf
        out = np.maximum.accumulate(sequence_arr)

    out[nan_mask] = np.nan
    return out




def _validate_args_are_all_none_or_lists_of_same_length(
    *args: list[Any] | None,
) -> None:
    all_none = all(arg is None for arg in args)
    all_list = all(isinstance(arg, list) for arg in args)

    if not all_none:
        if all_list:
            unique_list_lengths = set(map(len, args))  # type: ignore[arg-type]

            if len(unique_list_lengths) != 1:
                raise ValueError("All list arguments must have the same length.")

        else:
            raise ValueError("All arguments must be lists of the same length or None.")


def _task_to_categorical(task: list[EvalTask]) -> "pd.Series[str]":
    EvalTaskDtype = pd.CategoricalDtype(categories=[t.value for t in EvalTask])
    return pd.Series([t.value for t in task], dtype=EvalTaskDtype)


def _apply_reduction_to_batches(
    data: NDArray[np.float64],
    batch_ids: list[int],
    reduction_function: Callable[[Iterable[float]], float],
) -> NDArray[np.float64]:
    """Apply a reduction operator on batches of data.

    This function assumes that batch_ids are non-empty and sorted.

    Args:
        data: 1d array with data.
        batch_ids: A list with batch ids whose length is equal to the size of data.
            Values need to be sorted and can be repeated.
        reduction_function: A reduction function that takes an iterable of floats as
            input (e.g., a numpy.ndarray or list of floats) and returns a scalar. The
            function must be able to handle NaN's.

    Returns:
        The transformed data. Has one entry per unique batch id, equal to the result of
        applying the reduction function to the data of that batch.

    """
    batch_starts, batch_stops = _get_batch_starts_and_stops(batch_ids)

    batch_results: list[float] = []

    for start, stop in zip(batch_starts, batch_stops, strict=True):
        batch_data = data[start:stop]
        batch_id = batch_ids[start]

        try:
            if np.isnan(batch_data).all():
                reduced = np.nan
            else:
                reduced = reduction_function(batch_data)
        except Exception as e:
            msg = (
                f"Calling function {reduction_function.__name__} on batch {batch_id} "
                "of the History raised an Exception. Please verify that "
                f"{reduction_function.__name__} is well-defined, takes an iterable of "
                "floats as input and returns a scalar. The function must be able to "
                "handle NaN's."
            )
            raise ValueError(msg) from e

        if not np.isscalar(reduced):
            msg = (
                f"Function {reduction_function.__name__} did not return a scalar for "
                f"batch {batch_id}. Please verify that {reduction_function.__name__} "
                "returns a scalar when called on an iterable of floats. The function "
                "must be able to handle NaN's."
            )
            raise ValueError(msg)

        batch_results.append(float(reduced))  # type: ignore[arg-type,unused-ignore]

    return np.array(batch_results, dtype=np.float64)


def _get_batch_starts_and_stops(batch_ids: list[int]) -> tuple[list[int], list[int]]:
    """Get start and stop indices of batches.

    This function assumes that batch_ids are non-empty and sorted.

    """
    ids_arr = np.array(batch_ids, dtype=np.int64)
    indices = np.where(ids_arr[:-1] != ids_arr[1:])[0] + 1
    list_indices: list[int] = indices.tolist()
    starts = [0, *list_indices]
    stops = [*starts[1:], len(batch_ids)]
    return starts, stops
