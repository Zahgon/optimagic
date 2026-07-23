from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Type, TypeVar, cast

import numpy as np
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.engine import Engine

from optimagic.logging.base import (
    NonUpdatableKeyValueStore,
    UpdatableKeyValueStore,
)
from optimagic.logging.sqlalchemy import (
    IterationStore,
    ProblemStore,
    SQLAlchemyConfig,
    StepStore,
)
from optimagic.logging.types import (
    ExistenceStrategy,
    ExistenceStrategyLiteral,
    IterationState,
    IterationStateWithId,
    ProblemInitialization,
    ProblemInitializationWithId,
    StepResult,
    StepResultWithId,
    StepType,
)
from optimagic.typing import (
    Direction,
    DirectionLiteral,
    IterationHistory,
    MultiStartIterationHistory,
    PyTree,
)


class LogOptions:

    _subclass_registry: list[Type[LogOptions]] = []

    def __init_subclass__(
        cls: Type[LogOptions], abstract: bool = False, **kwargs: dict[Any, Any]
    ):
        if not abstract:
            LogOptions._subclass_registry.append(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def available_option_types(cls) -> list[Type[LogOptions]]:
        pass


_LogOptionsType = TypeVar("_LogOptionsType", bound=LogOptions)


class LogReader(Generic[_LogOptionsType], ABC):

    _step_store: UpdatableKeyValueStore[StepResult, StepResultWithId]
    _iteration_store: NonUpdatableKeyValueStore[IterationState, IterationStateWithId]
    _problem_store: UpdatableKeyValueStore[
        ProblemInitialization, ProblemInitializationWithId
    ]

    @property
    def problem_df(self) -> pd.DataFrame:
        pass

    @classmethod
    def from_options(cls, log_options: LogOptions) -> LogReader[_LogOptionsType]:
        log_reader_class = _LOG_OPTION_LOG_READER_REGISTRY.get(type(log_options), None)

        if log_reader_class is None:
            raise ValueError(
                f"No LogReader implementation found for type "
                f"{type(log_options)}. Available option types: "
                f"\n {list(_LOG_OPTION_LOG_READER_REGISTRY.keys())}"
            )

        return log_reader_class._create(log_options)

    @classmethod
    @abstractmethod
    def _create(cls, log_options: _LogOptionsType) -> LogReader[_LogOptionsType]:
        pass

    def read_iteration(self, iteration: int) -> IterationStateWithId:
        pass

    def read_history(self) -> IterationHistory:
        pass

    @staticmethod
    def _normalize_direction(
        direction: Direction | DirectionLiteral,
    ) -> Direction:
        if isinstance(direction, str):
            direction = Direction(direction)
        return direction

    def _build_history_dataframe(self) -> pd.DataFrame:
        steps = self._step_store.to_df()
        raw_res = self._iteration_store.select()

        history: dict[str, list[Any]] = {
            "params": [],
            "fun": [],
            "time": [],
            "step": [],
        }

        for data in raw_res:
            if data.scalar_fun is not None:
                history["params"].append(data.params)
                history["fun"].append(data.scalar_fun)
                history["time"].append(data.timestamp)
                history["step"].append(data.step)

        times = np.array(history["time"])
        times -= times[0]
        history["time"] = times.tolist()

        df = pd.DataFrame(history)
        df = df.merge(
            steps[[f"{self._step_store.primary_key}", "type"]],
            left_on="step",
            right_on=f"{self._step_store.primary_key}",
        )
        return df.drop(columns=f"{self._step_store.primary_key}")

    @staticmethod
    def _split_exploration_and_optimization(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        exploration = df.query(f"type == '{StepType.EXPLORATION.value}'").drop(
            columns=["step", "type"]
        )
        histories = df.query(f"type == '{StepType.OPTIMIZATION.value}'")
        histories = histories.drop(columns="type").set_index("step", append=True)

        return None if exploration.empty else exploration, histories

    @staticmethod
    def _sort_exploration(
        exploration: pd.DataFrame | None, optimization_type: Direction
    ) -> IterationHistory | None:
        if exploration is not None:
            is_minimization = optimization_type is Direction.MINIMIZE
            exploration = exploration.sort_values(by="fun", ascending=is_minimization)
            exploration_dict = cast(dict[str, Any], exploration.to_dict(orient="list"))
            return IterationHistory(**exploration_dict)
        return exploration

    @staticmethod
    def _extract_best_history(
        histories: pd.DataFrame, optimization_type: Direction
    ) -> tuple[IterationHistory, list[IterationHistory] | None]:
        groupby_step_criterion = histories["fun"].groupby(level="step")

        if optimization_type is Direction.MINIMIZE:
            best_idx = groupby_step_criterion.min().idxmin()
        else:
            best_idx = groupby_step_criterion.max().idxmax()

        remaining_indices = (
            histories.index.get_level_values("step").unique().difference([best_idx])
        )

        best_history: pd.DataFrame | pd.Series[Any] = histories.xs(
            best_idx, level="step"
        )

        def _to_dict(pandas_obj: pd.DataFrame | pd.Series) -> dict[str, Any]:
            if isinstance(pandas_obj, pd.DataFrame):
                result = pandas_obj.to_dict(orient="list")
            else:
                result = best_history.to_dict()
            return cast(dict[str, Any], result)

        best_history_dict = _to_dict(best_history)
        local_histories = [
            _to_dict(histories.xs(idx, level="step")) for idx in remaining_indices
        ]
        if len(local_histories) == 0:
            remaining_histories = None
        else:
            remaining_histories = [
                IterationHistory(**history_dict) for history_dict in local_histories
            ]

        return IterationHistory(**best_history_dict), remaining_histories

    def read_multistart_history(
        self, direction: Direction | DirectionLiteral
    ) -> MultiStartIterationHistory:
        """Read and the multistart optimization history.

        Args:
            direction: The optimization direction, either as an enum or string.

        Returns:
            A `MultiStartIterationHistory` object containing the best history,
                local histories, and exploration history.

        """
        optimization_type = self._normalize_direction(direction)
        history_df = self._build_history_dataframe()
        exploration, optimization_history = self._split_exploration_and_optimization(
            history_df
        )
        exploration_history = self._sort_exploration(exploration, optimization_type)
        best_history, remaining_histories = self._extract_best_history(
            optimization_history, optimization_type
        )

        return MultiStartIterationHistory(
            best_history,
            local_histories=remaining_histories,
            exploration=exploration_history,
        )

    def read_start_params(self) -> PyTree:
        """Read the start parameters form the problem store.

        Returns:
            A pytree object representing the start parameter.

        """
        return self._problem_store.select(1)[0].params


_LogReaderType = TypeVar("_LogReaderType", bound=LogReader[Any])


class LogStore(Generic[_LogOptionsType, _LogReaderType], ABC):

    def __init__(
        self,
        iteration_store: NonUpdatableKeyValueStore[
            IterationState, IterationStateWithId
        ],
        step_store: UpdatableKeyValueStore[StepResult, StepResultWithId],
        problem_store: UpdatableKeyValueStore[
            ProblemInitialization, ProblemInitializationWithId
        ],
    ):
        self.step_store = step_store
        self.iteration_store = iteration_store
        self.problem_store = problem_store

    @classmethod
    def from_options(
        cls, log_options: LogOptions
    ) -> LogStore[_LogOptionsType, _LogReaderType]:
        logger_class = _LOG_OPTION_LOGGER_REGISTRY.get(type(log_options), None)

        if logger_class is None:
            raise ValueError(
                f"No Logger implementation found for type "
                f"{type(log_options)}. Available option types: "
                f"\n {list(_LOG_OPTION_LOGGER_REGISTRY.keys())}"
            )

        return logger_class.create(log_options)

    @classmethod
    @abstractmethod
    def create(
        cls, log_options: _LogOptionsType
    ) -> LogStore[_LogOptionsType, _LogReaderType]:
        pass


class SQLiteLogOptions(SQLAlchemyConfig, LogOptions):

    def __init__(
        self,
        path: str | Path,
        fast_logging: bool = True,
        if_database_exists: ExistenceStrategy
        | ExistenceStrategyLiteral = ExistenceStrategy.RAISE,
    ):
        url = f"sqlite:///{path}"
        self._fast_logging = fast_logging
        self._path = path
        if isinstance(if_database_exists, str):
            if_database_exists = ExistenceStrategy(if_database_exists)
        self.if_database_exists = if_database_exists
        super().__init__(url)

    @property
    def path(self) -> str | Path:
        pass

    def create_engine(self) -> Engine:
        pass

    def _configure_engine(self, engine: Engine) -> None:
        pass


class SQLiteLogReader(LogReader[SQLiteLogOptions]):

    def __init__(self, path: str | Path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path=}")

        log_options = SQLiteLogOptions(
            path, fast_logging=True, if_database_exists=ExistenceStrategy.EXTEND
        )
        self._iteration_store = IterationStore(log_options)
        self._step_store = StepStore(log_options)
        self._problem_store = ProblemStore(log_options)

    @classmethod
    def _create(cls, log_options: SQLiteLogOptions) -> SQLiteLogReader:
        """Create an instance of SQLiteLogReader using the provided log options.

        Args:
            log_options (SQLiteLogOptions): Configuration options for the SQLite log.

        Returns:
            SQLiteLogReader: An instance of SQLiteLogReader initialized with the
            provided log options.

        """
        return cls(log_options.path)


class _SQLiteLogStore(LogStore[SQLiteLogOptions, SQLiteLogReader]):

    @staticmethod
    def _handle_existing_database(
        path: str | Path,
        if_database_exists: ExistenceStrategy | ExistenceStrategyLiteral,
    ) -> None:
        if isinstance(if_database_exists, str):
            if_database_exists = ExistenceStrategy(if_database_exists)
        database_exists = os.path.exists(path)
        if database_exists:
            if if_database_exists is ExistenceStrategy.RAISE:
                raise FileExistsError(
                    f"The database at {path} already exists. To reuse and extend "
                    f"the existing database, set if_database_exists to "
                    f"ExistenceStrategy.EXTEND."
                )
            elif if_database_exists is ExistenceStrategy.REPLACE:
                try:
                    os.remove(path)
                except PermissionError as e:
                    msg = (
                        f"Failed to remove file {path}. "
                        f"In particular, this can happen on Windows "
                        f"machines, when a different process is accessing the file, "
                        f"which results in a PermissionError. In this case, delete"
                        f"the file manually."
                    )
                    raise RuntimeError(msg) from e

    @classmethod
    def create(cls, log_options: SQLiteLogOptions) -> _SQLiteLogStore:
        cls._handle_existing_database(log_options.path, log_options.if_database_exists)

        iteration_store = IterationStore(log_options)
        step_store = StepStore(log_options)
        problem_store = ProblemStore(log_options)
        return cls(iteration_store, step_store, problem_store)


_LOG_OPTION_LOGGER_REGISTRY: dict[Type[LogOptions], Type[LogStore[Any, Any]]] = {
    SQLiteLogOptions: _SQLiteLogStore
}
_LOG_OPTION_LOG_READER_REGISTRY: dict[Type[LogOptions], Type[LogReader[Any]]] = {
    SQLiteLogOptions: SQLiteLogReader
}
