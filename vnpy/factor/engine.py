# factor_engine.py

import gc
import importlib
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging import DEBUG, ERROR, INFO, WARNING
from threading import Lock
from typing import Any, cast
from typing import Mapping, Optional, TypeVar, Union
import asyncio
from tqdm import tqdm

import dask
import dask.diagnostics
import numpy as np
import pandas as pd
import polars as pl  # Ensure polars is imported
import polars.selectors as cs
import psutil
from dask.delayed import Delayed

from vnpy.event import Event, EventEngine
from vnpy.trader.event import EVENT_DATAMANAGER_LOAD_BAR_REQUEST, EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST, \
    EVENT_FACTORMAKER_CALCULATE, EVENT_FACTORMAKER_BAR_READY, EVENT_FACTORMAKER_FACTOR_READY, \
    EVENT_FACTORMAKER_CALCULATE_ALL
from vnpy.factor.base import APP_NAME, FactorMode
from vnpy.factor.memory import FactorMemory, MemoryData
from vnpy.factor.template import FactorTemplate
from vnpy.trader.constant import Exchange, Interval
from vnpy.utils.datetimes import DatetimeUtils, TimeFreq

# FactorTemplate and FactorMemory are assumed to be defined above or importable
from vnpy.factor.utils.factor_utils import (
    init_factors,
    load_factor_setting,
    save_factor_setting,
)  # Ensure these utils are compatible
from vnpy.factor.utils.memory_utils import truncate_memory as truncate_bar_memory
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.event import (
    EVENT_BAR,
    EVENT_FACTOR,
    EVENT_LOG,
    EVENT_TICK,
    EVENT_FACTOR_FILLING,
    EVENT_FACTOR_BAR_UPDATE,
    EVENT_HISTORY_DATA_REQUEST,
    EVENT_DATAMANAGER_LOAD_BAR_RESPONSE, EVENT_DATAMANAGER_LOAD_FACTOR_RESPONSE
)
from vnpy.trader.object import BarData, LogData
from vnpy.trader.constant import Interval

from .setting import (
    FACTOR_MODULE_SETTINGS,
    get_factor_data_cache_path,
    get_factor_definitions_filename,
)

FACTOR_MODULE_NAME = FACTOR_MODULE_SETTINGS.get(
    "module_name", "vnpy.factor.factors"
)  # Use setting


# SYSTEM_MODE = SETTINGS.get('system.mode', 'LIVE') # Not used within FactorEngine


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\.\-@]", "_", name)
    return name


@dataclass
class CalculationMetrics:
    calculation_time: float
    memory_usage: float  # System memory usage delta
    cache_hits: int  # For external caches if used, Dask handles its own
    error_count: int


class FactorEngine(BaseEngine):
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        super().__init__(main_engine, event_engine, APP_NAME)

        # Use settings for paths and configuration
        self.setting_filename = get_factor_definitions_filename()  # Updated
        self.factor_data_dir = get_factor_data_cache_path()  # Updated

        # Load other settings
        self.factor_datetime_col = FACTOR_MODULE_SETTINGS.get(
            "datetime_col", "datetime"
        )
        self.loosen_ratio = FACTOR_MODULE_SETTINGS.get("loosen_ratio", 1.2)
        self.max_memory_length_bar = int(FACTOR_MODULE_SETTINGS.get(
            "max_memory_length_bar", 60
        ))
        self.max_memory_length_factor = int(FACTOR_MODULE_SETTINGS.get(
            "max_memory_length_factor", 60
        ))
        self.error_threshold = FACTOR_MODULE_SETTINGS.get("error_threshold", 3)

        try:
            self.module_factors = importlib.import_module(FACTOR_MODULE_NAME)
        except ImportError as e:
            self.write_log(
                f"Could not import factor module '{FACTOR_MODULE_NAME}': {e}. "
                "Factor loading might fail if factors are not self-contained.",
                level=ERROR,
            )
            self.module_factors = (
                None  # Allow engine to start, but factor loading may fail
            )

        # self.database = get_database()
        self.vt_symbols: list[str] = getattr(
            main_engine, "vt_symbols", []
        )  # Get from main_engine or default to empty
        if not self.vt_symbols and hasattr(main_engine, "get_all_contracts"):
            try:
                self.vt_symbols = [
                    contract.vt_symbol for contract in main_engine.get_all_contracts()
                ]
            except Exception as e:
                self.write_log(
                    f"Failed to get vt_symbols from main_engine.get_all_contracts(): {e}",
                    level=WARNING,
                )
        self.minimum_freq: TimeFreq = main_engine.minimum_freq

        self.stacked_factors: dict[str, FactorTemplate] = {}
        self.flattened_factors: dict[str, FactorTemplate] = {}

        self.memory_bar: dict[
            str, pl.DataFrame | MemoryData] = {}  # For OHLCV data. TODO: use MemoryData for a better management

        # NEW: Manages FactorMemory instances
        # self.factor_data_dir is already a Path object from get_factor_data_cache_path()
        self.factor_memory_instances: dict[str, FactorMemory] = {}
        self.latest_calculated_factors_cache: dict[str, pl.DataFrame] = {}

        self.dt: datetime | None = None
        self.bars: dict[str, BarData] = {}  # Current batch of bars
        self.tasks: dict[str, Delayed] = {}  # Dask tasks
        self.receiving_status: dict[str, bool] = (
            {sym: False for sym in self.vt_symbols} if self.vt_symbols else {}
        )

        self.metrics: dict[str, CalculationMetrics] = {}
        self.calculation_lock = Lock()  # Ensures only one execute_calculation runs at a time
        # self.thread_pool = ThreadPoolExecutor(max_workers=4) # Dask manages its own threading/processing
        self.consecutive_errors = 0

    def init_engine(self, fake: bool = False) -> None:
        self.write_log("Initializing FactorEngine...", level=INFO)
        self.factor_data_dir.mkdir(parents=True, exist_ok=True)
        self.register_event()

        # 1. Load factor configurations and initialize FactorTemplate instances
        # 2. Flatten the dependency tree in init_all_factors
        self.init_all_factors(flat_factors=True)  # Populates self.stacked_factors, determines max_lookbacks

        # 3. Initialize memory structures (memory_bar and FactorMemory instances)
        self.init_memory(fake=fake)

        # 4. Build Dask computational graph
        if self.flattened_factors:
            self.tasks = self.build_computational_graph()
            self.write_log(
                f"Built Dask computational graph with {len(self.tasks)} tasks.",
                level=INFO,
            )
        else:
            self.write_log(
                "No factors loaded, computational graph not built.", level=WARNING
            )

        # 5. Call on_start for all factors
        for factor in self.flattened_factors.values():
            if (
                    factor.inited and not factor.trading
            ):  # Start if inited but not yet trading
                self.call_factor_func(factor, factor.on_start)

        self.write_log("FactorEngine initialized successfully.", level=INFO)

    def register_event(self) -> None:
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_BAR, self.process_bar_event)
        self.event_engine.register(EVENT_DATAMANAGER_LOAD_BAR_RESPONSE, self.process_load_bar_response_event)
        self.event_engine.register(EVENT_DATAMANAGER_LOAD_FACTOR_RESPONSE, self.process_load_factor_response_event)
        self.event_engine.register(EVENT_FACTORMAKER_CALCULATE_ALL, self.process_factor_filling_event)
        # self.event_engine.register(EVENT_FACTORMAKER_FACTOR_READY, self.process_factor_filling_event)
        # self.event_engine.register(EVENT_ALL_READY,self.)

    @staticmethod
    def _adjust_period_(factor_name: str, attr: str, value: int | float):
        """

        Parameters
        ----------
        factor_name : str
            name of the factor
        attr : str
            the name of period like attribute in factor parameters
        value : int | float
            period value

        Returns
        -------

        """
        if factor_name == "trix" and attr == "timeperiod":
            return 3 * value
        return value

    def init_all_factors(self, flat_factors: bool = True) -> None:
        """Loads factor settings, initializes FactorTemplate instances, and determines max lookback periods."""
        self.write_log(
            "Loading factor settings and initializing factor instances...", level=INFO
        )  # Changed DEBUG to INFO
        try:
            factor_settings_list: list = load_factor_setting(
                self.setting_filename
            )  # Expects a list of settings dicts
            if not factor_settings_list:
                self.write_log(
                    "No factor settings found or file is empty.", level=WARNING
                )
                self.stacked_factors = {}
                return
        except FileNotFoundError:
            self.write_log(
                f"Factor settings file '{self.setting_filename}' not found. No factors loaded.",
                level=WARNING,
            )
            self.stacked_factors = {}
            return
        except Exception as e:
            self.write_log(
                f"Error loading factor settings from '{self.setting_filename}': {e}",
                level=ERROR,
            )
            self.stacked_factors = {}
            return

        # init_factors should take the list of settings and the factors module
        inited_factor_instances: list[FactorTemplate] = init_factors(
            self.module_factors,  # Module for finding primary factor classes
            factor_settings_list,
            dependencies_module_lookup_for_instances=self.module_factors,  # Module for their dependencies too
        )
        self.stacked_factors = {f.factor_key: f for f in inited_factor_instances}
        self.write_log(
            f"Loaded {len(self.stacked_factors)} stacked factors.", level=INFO
        )

        if flat_factors:
            self.flattened_factors = self.complete_factor_tree(self.stacked_factors)
            self.write_log(f"Flattened {len(self.flattened_factors)} factors", level=INFO)

        # Determine max lookback for bar data from factor parameters
        # And potentially a global default for FactorMemory max_rows if not specified per factor
        lookback_attrs = [
            "lookback_period",
            "window",
            "period",
            "fast_period",
            "slow_period",
            "signal_period",
            "fastperiod",
            "slowperiod",
            "signalperiod",
            "len",
            "length",
            "n",
            "k",
            "factor_memory_max_rows",
            "timeperiod"
        ]  # Added more common param names

        all_bar_lookbacks = [self.max_memory_length_bar]  # Start with default
        all_factor_mem_max_rows = [self.max_memory_length_factor]

        for factor in self.flattened_factors.values():
            factor_params = factor.get_params()
            freq_multiplier = DatetimeUtils.interval2unix(factor.freq, ret_unit=self.minimum_freq)
            for attr in lookback_attrs:
                period_len = factor_params.get(attr)
                if isinstance(period_len, (int, float)) and period_len > 0:
                    period_len = self._adjust_period_(factor.factor_name, attr=attr, value=period_len)
                    all_bar_lookbacks.append(period_len * freq_multiplier)
                    all_factor_mem_max_rows.append(period_len * freq_multiplier)

        self.max_bar_lookback = max(all_bar_lookbacks)
        self.max_factor_lookback = max(all_factor_mem_max_rows)
        self.max_memory_length_bar = int(max(all_bar_lookbacks) * self.loosen_ratio)
        self.max_memory_length_factor = int(max(all_factor_mem_max_rows) * self.loosen_ratio)
        self.write_log(
            f"Max memory length for bar data: {self.max_memory_length_bar}, for factor data: {self.max_memory_length_factor}",
            level=INFO
        )

    def init_memory(self, fake: bool = False, end: datetime = None) -> None:
        """Initializes memory_bar (in-memory OHLCV) and FactorMemory instances for each factor.

        Parameters
        ----------
        end: datetime
            The end datetime for initializing memory. If None, uses current time.
            length is defined by max_memory_length_bar, max_memory_length_factor and minimum_freq.

        Examples
        --------
        >>> end=datetime(2025,1,1,1,0,0)
        >>> minimum_freq = TimeFreq.m
        >>> max_memory_length_bar = 60
        >>> max_memory_length_factor = 60
        [datetime(2025,1,1,0,1,0), datetime(2025,1,1,1,0,0)] will be loaded
        """
        self.write_log(
            "Initializing memory structures...", level=INFO
        )  # Changed DEBUG to INFO

        # If vt_symbols are not yet defined, collect them from all factors and request data
        if not self.vt_symbols:
            all_factor_symbols = set()
            for factor in self.flattened_factors.values():
                if hasattr(factor, "vt_symbols") and factor.vt_symbols:
                    all_factor_symbols.update(factor.vt_symbols)

            if all_factor_symbols:
                self.vt_symbols = sorted(list(all_factor_symbols))
                self.write_log(f"Collected {len(self.vt_symbols)} symbols from factors. Requesting history data.",
                               level=INFO)

                # Dispatch event to request historical bar data
                # The data provider should listen to this and send back EVENT_FACTOR_BAR_UPDATE
                request_event = Event(EVENT_HISTORY_DATA_REQUEST, {"symbols": self.vt_symbols})
                self.event_engine.put(request_event)
            else:
                self.write_log("No vt_symbols defined in factors and no symbols loaded from main_engine.",
                               level=WARNING)

        # 1. Initialize memory_bar (OHLCV)
        # Use a consistent schema for bar data DataFrames
        bar_data_schema = {"datetime": pl.Datetime(time_unit="us")}  # Datetime column
        if self.vt_symbols:
            for symbol in self.vt_symbols:
                bar_data_schema[symbol] = pl.Float64  # Price/volume data per symbol
        else:  # No symbols, create a placeholder column to avoid empty schema issues if needed later
            bar_data_schema["placeholder_value"] = pl.Float64

        for b_col in ["open", "high", "low", "close", "volume"]:
            self.memory_bar[b_col] = pl.DataFrame(
                data={}, schema=bar_data_schema.copy()
            )

        # 2. Initialize FactorMemory instances for each flattened factor
        self.factor_memory_instances.clear()
        for factor_key, factor_instance in self.flattened_factors.items():
            try:
                # Ensure factor_instance.vt_symbols is populated before calling get_output_schema,
                # as the schema generation might depend on the symbols.
                # This is a safeguard in case the factor's __init__ didn't set it from params.
                if (
                        not hasattr(factor_instance, "vt_symbols")
                        or not factor_instance.vt_symbols
                ):
                    factor_instance.vt_symbols = self.vt_symbols

                output_schema = factor_instance.get_output_schema()
                if self.factor_datetime_col not in output_schema:
                    raise ValueError(
                        f"Factor '{factor_key}' output schema must contain the datetime column '{self.factor_datetime_col}'."
                    )

                file_path = self.factor_data_dir.joinpath(
                    f"{safe_filename(factor_key)}.arrow"
                )

                # --- THIS IS THE KEY CHANGE ---
                self.factor_memory_instances[factor_key] = FactorMemory(
                    file_path=file_path,
                    max_rows=self.max_memory_length_factor,
                    schema=output_schema,
                    datetime_col=self.factor_datetime_col,
                    mode=factor_instance.factor_mode  # Pass the factor's mode to its memory manager
                )
            except Exception as e:
                self.write_log(
                    f"Failed to initialize FactorMemory for {factor_key}: {e}. This factor may not calculate correctly.",
                    level=ERROR,
                )

        # 3. Populate with fake data if requested
        if fake and self.vt_symbols:
            self.write_log("Populating with fake data...", level=DEBUG)
            fake_dates = pl.datetime_range(
                start=datetime.now()
                      - timedelta(days=max(self.max_memory_length_bar, self.max_memory_length_factor) // (24 * 60) + 1
                                  ),  # Enough days for minute data
                end=datetime.now(),
                interval="1m",
                time_unit="us",
                eager=True,
            ).alias("datetime")

            num_fake_rows = len(fake_dates)
            if num_fake_rows == 0:
                self.write_log(
                    "Could not generate fake dates for fake data population.",
                    level=WARNING,
                )
                return

            # Fake data for memory_bar
            for b_col in ["open", "high", "low", "close", "volume"]:
                bar_fake_df_data = {"datetime": fake_dates}
                for symbol in self.vt_symbols:
                    bar_fake_df_data[symbol] = np.random.rand(num_fake_rows) * 100 + (
                        50 if b_col != "volume" else 1000
                    )

                # Ensure schema matches what self.memory_bar[b_col] expects
                current_bar_schema = self.memory_bar[b_col].schema
                fake_bar_df = pl.DataFrame(bar_fake_df_data).select(
                    [
                        pl.col(c).cast(current_bar_schema[c])
                        for c in current_bar_schema.keys()
                    ]
                )
                self.memory_bar[b_col] = pl.concat(
                    [
                        self.memory_bar[b_col],
                        fake_bar_df.tail(self.max_memory_length_bar),
                    ],  # Keep within bar memory limits
                    how="vertical_relaxed",
                )

            # Fake data for FactorMemory instances
            for factor_key, fm_instance in self.factor_memory_instances.items():
                factor_output_schema = fm_instance.schema
                factor_fake_df_data = {self.factor_datetime_col: fake_dates}
                for col_name, col_type in factor_output_schema.items():
                    if col_name == self.factor_datetime_col:
                        continue
                    if col_type.base_type() in [pl.Float32, pl.Float64]:
                        factor_fake_df_data[col_name] = (
                                np.random.rand(num_fake_rows) * 10
                        )
                    elif col_type.base_type() in [
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                    ]:
                        factor_fake_df_data[col_name] = np.random.randint(
                            0, 5, size=num_fake_rows
                        )
                    elif col_type.base_type() == pl.Boolean:
                        factor_fake_df_data[col_name] = np.random.choice(
                            [True, False], size=num_fake_rows
                        )
                    else:  # Fallback for Utf8 or other types
                        factor_fake_df_data[col_name] = [
                            f"val_{i % 5}" for i in range(num_fake_rows)
                        ]

                fake_factor_df = pl.DataFrame(
                    factor_fake_df_data, schema=factor_output_schema
                )
                fm_instance.update_data(
                    fake_factor_df
                )  # FactorMemory handles truncation internally
                # Update cache with the latest fake row
                if not fake_factor_df.is_empty():
                    self.latest_calculated_factors_cache[factor_key] = (
                        fm_instance.get_latest_rows(1)
                    )
            self.write_log(
                f"fake memory initialization complete for {len(self.factor_memory_instances)} factors.",
                level=DEBUG,
            )
        elif not fake and self.vt_symbols:
            # TODO: retrieve data from database with event
            pass

    def complete_factor_tree(
            self, factors: dict[str, FactorTemplate]
    ) -> dict[str, FactorTemplate]:
        """Recursively flattens the dependency tree for all factors."""
        resolved_factors: dict[str, FactorTemplate] = {}

        def traverse_dependencies(
                factor: FactorTemplate, resolved: dict[str, FactorTemplate]
        ) -> None:
            if factor.factor_key in resolved:
                return

            # Resolve dependencies first
            for dependency_instance in factor.dependencies_factor:
                if isinstance(
                        dependency_instance, FactorTemplate
                ):  # Ensure it's an instance
                    traverse_dependencies(dependency_instance, resolved)
                else:
                    self.write_log(
                        f"Warning: Dependency {dependency_instance} for {factor.factor_key} is not a FactorTemplate instance. Skipping.",
                        level=WARNING,
                    )

            # Add the current factor after its dependencies are resolved
            resolved[factor.factor_key] = factor

        for _, factor_instance in factors.items():
            traverse_dependencies(factor_instance, resolved_factors)

        return resolved_factors

    def build_computational_graph(self) -> dict[str, Delayed]:
        """Builds a Dask computational graph based on flattened factors."""
        # self.write_log("Building Dask computational graph...", level=DEBUG) # Removed, init_engine logs similar message
        tasks: dict[str, Delayed] = {}

        # Dependency graph for topological sort: factor_key -> list of dependency_factor_keys
        dependency_graph: dict[str, list[str]] = {}
        for factor_key, factor_instance in self.flattened_factors.items():
            dependency_graph[factor_key] = [
                dep.factor_key
                for dep in factor_instance.dependencies_factor
                if isinstance(dep, FactorTemplate)
            ]

        try:
            # Sort factors topologically to define creation order for Dask tasks
            # Order: dependencies first, then the factors that depend on them.
            sorted_factor_keys = self._topological_sort(dependency_graph)
        except ValueError as e:  # Circular dependency
            self.write_log(f"Error building computational graph: {e}", level=ERROR)
            return {}  # Return empty tasks if graph is invalid

        # Build tasks in dependency order
        for factor_key in sorted_factor_keys:
            if factor_key in self.flattened_factors:  # Ensure factor still exists
                self.create_task(factor_key, self.flattened_factors, tasks)
            else:
                self.write_log(
                    f"Warning: Factor key {factor_key} from sort not in flattened_factors. Skipping task.",
                    level=WARNING,
                )

        return tasks

    def _topological_sort(self, graph: dict[str, list]) -> list:
        """Performs topological sort. Args: graph (node -> list of dependencies)."""
        visited_permanently = (
            set()
        )  # Nodes whose processing (including all descendants) is complete
        visited_temporarily = (
            set()
        )  # Nodes currently in the recursion stack (for cycle detection)
        order = []  # The topologically sorted list of nodes

        def visit(node: str):
            if node in visited_permanently:
                return
            if node in visited_temporarily:  # Cycle detected
                path = " -> ".join(list(visited_temporarily) + [node])
                raise ValueError(
                    f"Circular dependency detected in factor graph: {path}"
                )

            visited_temporarily.add(node)

            # Visit dependencies of the current node
            if (
                    node in graph
            ):  # Check if node is in graph (might be a leaf with no recorded deps)
                for dep_node in graph[node]:
                    visit(dep_node)

            visited_temporarily.remove(node)  # Remove from recursion stack
            visited_permanently.add(node)  # Mark as fully processed
            order.append(node)  # Add to the sorted list

        # Collect all unique nodes from the graph keys and dependency lists
        nodes_as_keys = set(graph.keys())
        nodes_as_dependencies = set()
        for dep_list in graph.values():
            nodes_as_dependencies.update(dep_list)
        all_nodes = nodes_as_keys | nodes_as_dependencies

        for (
                node_key
        ) in all_nodes:  # Iterate over all known nodes to ensure all are visited
            if node_key not in visited_permanently:
                visit(node_key)

        # The `order` list now has dependencies before the nodes that depend on them.
        # For Dask task creation, this is the correct order.
        return order

    def _get_current_memory_bar_dask_input(self) -> dict[str, pl.DataFrame]:
        """Prepares OHLCV bar data for Dask task input. Returns copies."""
        # Dask works best with immutable inputs or copies to avoid side effects
        return self.memory_bar.copy()  # Return a copy of the current memory_bar

    def _get_factor_memory_instance_for_dask(self, factor_key: str) -> FactorMemory:
        """Returns the FactorMemory instance for a factor.
        Dask should be able to handle passing this instance if its state (like _lock)
        is managed correctly or not used across workers.
        """
        if factor_key not in self.factor_memory_instances:
            # This should not happen if init_memory was successful for all flattened_factors
            raise RuntimeError(
                f"FactorMemory instance for {factor_key} not found. Engine not properly initialized."
            )
        return self.factor_memory_instances[factor_key]

    def create_task(
            self,
            factor_key: str,
            factors_dict: dict[str, FactorTemplate],  # All flattened factors
            tasks_dict: dict[str, Delayed],  # Accumulator for created tasks
    ) -> Delayed:
        """Recursively creates a Dask task for a given factor and its dependencies."""
        if factor_key in tasks_dict:  # Task already created
            return tasks_dict[factor_key]

        factor_instance = factors_dict[factor_key]

        # Prepare Dask Delayed objects for dependencies
        dependency_input_tasks: dict[str, Delayed] = {}
        for dep_factor_template in factor_instance.dependencies_factor:
            dep_key = dep_factor_template.factor_key
            # Recursively create/get task for each dependency
            dependency_input_tasks[dep_key] = self.create_task(
                dep_key, factors_dict, tasks_dict
            )

        # Get the FactorMemory instance for this factor
        # This instance will be passed to the factor.calculate method
        factor_memory_obj_delayed = dask.delayed(
            self._get_factor_memory_instance_for_dask
        )(factor_key)

        if not factor_instance.dependencies_factor:  # Leaf factor (depends on bar data)
            # Input data is the current OHLCV bar memory
            bar_data_delayed_input = dask.delayed(
                self._get_current_memory_bar_dask_input
            )()

            task = dask.delayed(factor_instance.calculate)(
                input_data=bar_data_delayed_input, memory=factor_memory_obj_delayed
            )
        else:  # Factor with dependencies on other factors
            task = dask.delayed(
                factor_instance.calculate
            )(
                input_data=dependency_input_tasks,  # Dict of Delayed objects (results of other factors)
                memory=factor_memory_obj_delayed,
            )

        tasks_dict[factor_key] = task
        return task

    def on_bars(self, dt: datetime, bars: dict[str, BarData]) -> None:
        """Processes a complete slice of bars for a given datetime."""
        if not bars:
            self.write_log(
                f"on_bars called with empty bars for dt: {dt}. Skipping.", level=DEBUG
            )
            return

        # 1. Update memory_bar (OHLCV data)
        # Prepare a single row DataFrame for each OHLCV type
        new_ohlcv_rows: dict[str, dict[str, Any]] = {
            b_col: {self.factor_datetime_col: dt}
            for b_col in ["open", "high", "low", "close", "volume"]
        }

        for vt_symbol, bar_data_obj in bars.items():
            if (
                    vt_symbol not in self.vt_symbols
            ):  # Dynamically add new symbols if encountered
                self.vt_symbols.append(vt_symbol)
                # Update schemas for memory_bar if a new symbol appears
                for b_col in self.memory_bar.keys():
                    if vt_symbol not in self.memory_bar[b_col].columns:
                        self.memory_bar[b_col] = self.memory_bar[b_col].with_columns(
                            pl.lit(None, dtype=pl.Float64).alias(vt_symbol)
                        )

            new_ohlcv_rows["open"][vt_symbol] = bar_data_obj.open_price
            new_ohlcv_rows["high"][vt_symbol] = bar_data_obj.high_price
            new_ohlcv_rows["low"][vt_symbol] = bar_data_obj.low_price
            new_ohlcv_rows["close"][vt_symbol] = bar_data_obj.close_price
            new_ohlcv_rows["volume"][vt_symbol] = bar_data_obj.volume

        for b_col, data_dict_for_row in new_ohlcv_rows.items():
            try:
                # Ensure the schema for the new row matches the existing DataFrame schema
                expected_schema = self.memory_bar[b_col].schema
                # Create a 1-row DataFrame.
                # new_ohlcv_rows[b_col] (aliased as data_dict_for_row) contains the datetime
                # and available symbol data for the current bar.
                # Providing the full expected_schema to pl.DataFrame constructor ensures
                # that any columns in expected_schema but missing from data_dict_for_row
                # (e.g., symbols that didn't have a bar in this specific dt) are created as nulls.
                new_row_df = pl.DataFrame([data_dict_for_row], schema=expected_schema)

                self.memory_bar[b_col] = pl.concat(
                    [self.memory_bar[b_col], new_row_df],
                    how="vertical_relaxed",  # vertical_relaxed is more robust to minor schema diffs if they occur
                )
            except Exception as e:
                self.write_log(
                    f"Error updating memory_bar for '{b_col}': {e}. Affected datetime: {dt}",
                    level=ERROR,
                )  # Removed verbose Row data
                # Potentially stop processing if bar memory update fails critically
                return

        # 2. Execute factor calculations using Dask
        if self.tasks:
            self.execute_calculation(
                dt=dt
            )  # Results written to FactorMemory, cache updated
        else:
            self.write_log(
                "No Dask tasks defined, skipping factor calculation.", level=DEBUG
            )

        # 3. Broadcast the FactorMemory instances directly
        if self.factor_memory_instances:
            # Broadcast FactorMemory instances directly
            event_data = {k: v.get_latest_rows(1) for k, v in self.factor_memory_instances.items()}
            self.event_engine.put(Event(EVENT_FACTOR, event_data))

        # 4. Maintain memory length for bar data (OHLCV)
        # FactorMemory instances handle their own truncation.
        self._truncate_memory_bar()

        # self.write_log(f"Finished processing bars for {dt}.", level=DEBUG) # Removed, can be too noisy

    def execute_calculation(self, dt: datetime, high_performance=False) -> None:
        """Executes the Dask computational graph and updates FactorMemory instances."""
        if not self.tasks:
            self.write_log("No tasks to execute.", level=DEBUG)
            return

        self.write_log(f"Executing Dask computation for datetime {dt}...", level=DEBUG, show=~high_performance)

        # Ensure thread-safety for the entire calculation execution block if needed,
        # though Dask's local scheduler might handle this internally.
        # The self.calculation_lock here prevents multiple calls to execute_calculation from overlapping.
        with self.calculation_lock:
            start_time = time.time()
            if not high_performance:
                initial_resources = self._monitor_resources()

            # Configure Dask for local single threaded execution (default if no cluster)
            # dask.config.set(scheduler='single-threaded') # this would fix the interpreter shut down issue
            # dask.config.set(num_workers=psutil.cpu_count(logical=False)) # Optional: set num_workers

            try:
                if high_performance:
                    # Compute all tasks. Results will be a list of DataFrames.
                    with dask.config.set(scheduler="single-threaded"):
                        computed_results = dask.compute(
                            *self.tasks.values(), optimize_graph=True
                        )
                else:
                    with dask.diagnostics.ProgressBar(minimum=0.1):
                        # Compute all tasks. Results will be a list of DataFrames.
                        with dask.config.set(scheduler="single-threaded"):
                            computed_results = dask.compute(
                                *self.tasks.values(), optimize_graph=True
                            )
                end_time = time.time()
                self.write_log(
                    f"Dask computation finished in {end_time - start_time:.3f}s.",
                    level=DEBUG, show=~high_performance,
                )

                # Clear cache before populating with new results
                self.latest_calculated_factors_cache.clear()

                calculation_errors_count = 0
                # Lists to store error details for summary logging
                computation_issues: list[str] = []
                memory_update_issues: list[str] = []
                missing_memory_instances: list[str] = []

                error_factors = []
                # Process results: update FactorMemory instances and the latest_factors_cache
                for factor_key, result_df in zip(
                        self.tasks.keys(), computed_results, strict=False
                ):
                    if result_df[:, "btcusdt.BINANCE"].is_nan().all():
                        error_factors.append(factor_key)
                    if result_df is None:
                        computation_issues.append(f"{factor_key}: returned None")
                        calculation_errors_count += 1
                        continue
                    if not isinstance(result_df, pl.DataFrame):
                        computation_issues.append(
                            f"{factor_key}: returned non-DataFrame type {type(result_df)}"
                        )
                        calculation_errors_count += 1
                        continue

                    fm_instance = self.factor_memory_instances.get(factor_key)
                    if fm_instance:
                        try:
                            fm_instance.update_data(result_df)
                            latest_row = fm_instance.get_latest_rows(1)
                            if not latest_row.is_empty():
                                self.latest_calculated_factors_cache[factor_key] = latest_row
                        except Exception as e:
                            # Log full traceback here as it's an unexpected error during memory update
                            memory_update_issues.append(
                                f"{factor_key}: {e}\n{traceback.format_exc(limit=2)}"
                            )  # Limit traceback depth
                            calculation_errors_count += 1
                    else:
                        missing_memory_instances.append(f"{factor_key}")
                        calculation_errors_count += 1

                if error_factors:
                    self.write_log(f"factor calculation error occurs in these factors: {str(error_factors)}",
                                   level=ERROR)

                if computation_issues:
                    self.write_log(
                        f"Computation issues for {len(computation_issues)} factors: {'; '.join(computation_issues)}",
                        level=WARNING,
                    )
                if memory_update_issues:
                    self.write_log(
                        f"FactorMemory update issues for {len(memory_update_issues)} factors. See details below:",
                        level=ERROR,
                    )
                    for issue in memory_update_issues:
                        self.write_log(
                            issue, level=ERROR
                        )  # Log each with traceback separately for clarity
                if missing_memory_instances:
                    self.write_log(
                        f"Missing FactorMemory instances for factors: {', '.join(missing_memory_instances)}",
                        level=ERROR,
                    )

                if not high_performance:
                    final_resources = self._monitor_resources()
                    # Update metrics (overall for the batch)
                    # Per-factor metrics would require more granular timing within Dask tasks.
                    self.metrics["overall_batch"] = CalculationMetrics(
                        calculation_time=end_time - start_time,
                        # initial_resources is expected to always contain "memory_percent".
                        # The .get() with fallback is defensive; memory_usage would be 0 if key were missing.
                        memory_usage=final_resources["memory_percent"]
                                     - initial_resources.get(
                            "memory_percent", final_resources["memory_percent"]
                        ),
                        cache_hits=0,  # Dask handles its own caching/optimization
                        error_count=calculation_errors_count,
                    )

                if calculation_errors_count > 0:
                    self.consecutive_errors += 1
                else:
                    self.consecutive_errors = 0  # Reset on successful batch

                if not high_performance:
                    self.write_log(
                        f"Factor calculations processed. Batch errors: {calculation_errors_count}",
                        level=DEBUG, show=~high_performance,
                    )
                    self.write_log(
                        f"Resource usage - Memory: {final_resources['memory_percent']:.1f}%, CPU: {final_resources['cpu_percent']:.1f}%",
                        level=DEBUG,
                    )

            except Exception as e_dask:
                self.consecutive_errors += 1
                self.write_log(
                    f"Critical error during Dask computation or result processing: {e_dask}\n{traceback.format_exc()}",
                    level=ERROR,
                )
                # Circuit breaker logic
                if self.consecutive_errors >= self.error_threshold:
                    self.write_log(
                        f"Consecutive error threshold ({self.error_threshold}) reached. Stopping all factors.",
                        level=ERROR,
                    )
                    self.stop_all_factors()
                    # Potentially re-raise to halt the engine if critical
                    # raise RuntimeError(f"FactorEngine critical failure after {self.consecutive_errors} errors.") from e_dask
            finally:
                # self._cleanup_memory_resources() # GC and other cleanup
                # if self.calculation_lock.locked():
                #     self.calculation_lock.release()
                pass
        if self.calculation_lock.locked():
            self.calculation_lock.release()

    def _truncate_memory_bar(self) -> None:
        """Truncates the in-memory OHLCV bar data."""
        if not self.memory_bar:
            return
        # Assuming truncate_bar_memory is a utility that works on Dict[str, pl.DataFrame]
        truncate_bar_memory(self.memory_bar, self.max_memory_length_bar)

    def _monitor_resources(self) -> dict[str, float]:
        process = psutil.Process()
        disk_io = psutil.disk_io_counters()
        mem_info = process.memory_info()
        return {
            "memory_rss_mb": mem_info.rss / (1024 * 1024),
            "memory_vms_mb": mem_info.vms / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(interval=0.05),  # Non-blocking
            "disk_io_read_mb": disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
            "disk_io_write_mb": disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
        }

    def _cleanup_memory_resources(self) -> None:
        """Performs garbage collection."""
        gc.collect()
        self.write_log(
            "Garbage collection performed.", level=DEBUG
        )  # Re-enabled this DEBUG log

    def _parse_gap_dict(self, gap_dict: dict) -> tuple[datetime | None, datetime | None, TimeFreq | None, list | None]:
        # This function was written by Gemini
        """
        Parses a dictionary of gaps to determine the overall time range and minimum interval.
        """
        begin = None
        end = None
        min_freq: TimeFreq | None = None
        vt_symbol_list = []
        symbol_list = []
        exchange_list = []

        for key, gaps in gap_dict.items():
            if not gaps:
                continue

            current_min_freq = min(DatetimeUtils.interval2freq(gap.interval) for gap in gaps)
            if min_freq is None:
                min_freq = current_min_freq
            else:
                min_freq = min(min_freq, current_min_freq)

            current_begin = min(gap.start for gap in gaps)
            if begin is None:
                begin = current_begin
            else:
                begin = min(begin, current_begin)

            current_end = max(gap.end for gap in gaps)
            if end is None:
                end = current_end
            else:
                end = max(end, current_end)

            vt_symbol = key.split('_')[-1]
            vt_symbol_list.append(vt_symbol)

        return begin, end, min_freq, list(set(vt_symbol_list))

    def send_load_bar_request(self, type_, param) -> None:
        """

        Parameters
        ----------
        type_ :
        param : dict
            - if type_ is 'gap', param should be a dictionary with keys:
                - 'begin': datetime, start of the gap
                - 'end': datetime, end of the gap
                - 'min_freq': TimeFreq, minimum interval for the data
                - 'vt_symbol_list': list of vt_symbols to load data for
            - if type_ is 'period', param should be a dictionary with keys:
                - 'begin': datetime, start of the period. inclusive
                - 'end': datetime, end of the period. inclusive
                - 'interval': TimeFreq, interval for the data
                - 'vt_symbol_list': list of vt_symbols to load data for

        Returns
        -------

        """
        # The following logic was adjusted by Gemini for performance enhancement.
        if type_ == 'gap':
            gap_dict = cast(Mapping[str, Any], param)
            if not gap_dict or not isinstance(gap_dict, dict):
                self.write_log("No gaps provided for loading historical data.", level=WARNING)
                return

            begin, end, min_freq, vt_symbol_list = self._parse_gap_dict(gap_dict)

            if not begin or not end or not min_freq:
                self.write_log(
                    "No valid gaps found in gap_dict. Cannot determine start and end for loading data.",
                    level=ERROR,
                )
                return

            # Calculate required lookback and prepare ranges. missing+window+2 of bar data needed
            freq_multiplier = DatetimeUtils.freq2unix(min_freq, ret_unit=self.minimum_freq)
            denominator = self.minimum_freq / TimeFreq.m
            max_lookback = int(self.max_memory_length_factor * freq_multiplier / denominator) # window
            calculate_start = begin - timedelta(minutes=max_lookback)
            lookback_start = calculate_start - timedelta(minutes=max_lookback)

            # Request historical data with lookback period
            hist_req = Event(
                type=EVENT_DATAMANAGER_LOAD_BAR_REQUEST,
                data=[{
                    "symbol": vt_symbol.split('.')[0],  # Extract symbol from vt_symbol
                    "exchange": Exchange(vt_symbol.split('.')[1]) if '.' in vt_symbol else Exchange.UNKNOWN,
                    "interval": Interval(DatetimeUtils.freq2str(min_freq)),
                    "start": lookback_start,
                    "end": end,
                    "calculate_start": calculate_start,
                    "ret": "polars"
                } for vt_symbol in vt_symbol_list],
            )
            self.write_log(f"Requesting historical data for symbols: {vt_symbol_list}", level=INFO)
            self.event_engine.put(hist_req)
        elif type_ == 'period':
            # when begin==end then we can load expected lookback period before begin
            period = cast(Mapping[str, Any], param)
            if not period or not isinstance(period, dict):
                self.write_log("No time period provided for loading historical data.", level=WARNING)
                return

            begin = period.get("begin")
            end = period.get("end")
            interval = period.get("interval")
            vt_symbol_list = period.get("vt_symbol_list", [])

            if not begin or not end or not interval or not vt_symbol_list:
                self.write_log(
                    "Invalid time period parameters. Cannot determine start, end, interval, or symbols.",
                    level=ERROR,
                )
                return

            # Calculate required lookback
            freq_multiplier = DatetimeUtils.interval2unix(interval, ret_unit=self.minimum_freq)
            denominator = self.minimum_freq / TimeFreq.m
            max_lookback = int(self.max_memory_length_factor * freq_multiplier / denominator)
            lookback_start = begin - timedelta(minutes=max_lookback)

            # Request historical data with lookback period
            hist_req = Event(
                type=EVENT_DATAMANAGER_LOAD_BAR_REQUEST,
                data=[{
                    "symbol": vt_symbol.split('.')[0],  # Extract symbol from vt_symbol
                    "exchange": Exchange(vt_symbol.split('.')[1]) if '.' in vt_symbol else Exchange.UNKNOWN,
                    "interval": interval,
                    "start": lookback_start,
                    "end": end,
                    "ret": "polars"
                } for vt_symbol in vt_symbol_list],
            )
            self.write_log(f"Requesting historical data for symbols: {vt_symbol_list}", level=INFO)
            self.event_engine.put(hist_req)
        else:
            raise NotImplementedError()

    def send_load_factor_request(self, type_, param) -> None:
        # This function was written by Gemini
        # The following logic was adjusted by Gemini for performance enhancement.
        if type_ == 'gap':
            gap_dict = cast(Mapping[str, Any], param)
            if not gap_dict or not isinstance(gap_dict, dict):
                self.write_log("No gaps provided for loading historical factor data.", level=WARNING)
                return

            begin, end, min_freq, vt_symbol_list = self._parse_gap_dict(gap_dict)
            factor_keys = list(self.flattened_factors.keys())

            if not begin or not end or not min_freq:
                self.write_log(
                    "No valid gaps found in gap_dict. Cannot determine start and end for loading factor data.",
                    level=ERROR,
                )
                return

            # Calculate required lookback
            freq_multiplier = DatetimeUtils.freq2unix(min_freq, ret_unit=self.minimum_freq)
            denominator = self.minimum_freq / TimeFreq.m
            max_lookback = int(self.max_memory_length_factor * freq_multiplier / denominator)
            lookback_start = begin - timedelta(minutes=max_lookback)

            # Request historical factor data with lookback period
            hist_event = Event(
                type=EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST,
                data=[{
                    "symbol": vt_symbol.split('.')[0],  # Extract symbol from vt_symbol
                    "exchange": Exchange(vt_symbol.split('.')[1]) if '.' in vt_symbol else Exchange.UNKNOWN,
                    "interval": Interval(DatetimeUtils.freq2str(min_freq)),
                    "factor_list": factor_keys,
                    "start": lookback_start,
                    "end": end,
                    "ret": "polars"
                } for vt_symbol in vt_symbol_list]
            )
            self.write_log(f"Requesting historical factor data for factors: {factor_keys}", level=INFO)
            self.event_engine.put(hist_event)
        else:
            raise NotImplementedError()

    def process_tick_event(self, event: Event) -> None:
        # FactorMaker typically works on bars, not ticks.
        # Implement if specific tick-based logic is needed.
        pass

    def process_bar_event(self, event: Event) -> None:
        """Processes an incoming bar, batches them by datetime, then calls on_bars."""
        bar: BarData = event.data

        # If vt_symbols list was initially empty, populate it as we see symbols
        if not self.vt_symbols and hasattr(self.main_engine, "get_all_contracts"):
            self.vt_symbols = [
                c.vt_symbol for c in self.main_engine.get_all_contracts()
            ]
            self.receiving_status = {
                sym: False for sym in self.vt_symbols
            }  # Re-init status

        if bar.vt_symbol not in self.receiving_status:  # New symbol encountered
            self.write_log(
                f"New symbol {bar.vt_symbol} encountered from BAR event. Adding to tracking.",
                level=INFO,
            )
            self.vt_symbols.append(bar.vt_symbol)
            self.receiving_status[bar.vt_symbol] = False
            # Might need to adjust schemas of self.memory_bar here if not handled in on_bars
            for b_col_key in self.memory_bar.keys():
                if bar.vt_symbol not in self.memory_bar[b_col_key].columns:
                    self.memory_bar[b_col_key] = self.memory_bar[
                        b_col_key
                    ].with_columns(pl.lit(None, dtype=pl.Float64).alias(bar.vt_symbol))

        # Check if the bar is continuous with the last bar in memory_bar
        last_dt = self.memory_bar['open'][-1, 'datetime'] if self.memory_bar['open'].shape[0] > 0 else datetime(1970, 1, 1)
        if bar.datetime-last_dt>timedelta(minutes=DatetimeUtils.freq2unix(self.minimum_freq,ret_unit='m')) or bar.datetime <= last_dt:
            self.write_log(
                f"Bar datetime {bar.datetime} is not continuous with last bar datetime {last_dt}. "
                ,level=ERROR
            )
            raise ValueError(f"Bar datetime {bar.datetime} is not continuous with last bar datetime {last_dt}. ")

        # If bar is from a new datetime, process the previous batch
        if self.dt is None:  # First bar of a new batch
            self.dt = bar.datetime

        if bar.datetime > self.dt:
            # only when strictly all bar datas are received, `on_bars` will be called
            # if any(
            #         self.receiving_status.values()
            # ):  # If any bars were received for self.dt
            #     self.write_log(
            #         f"Bar time roll: new {bar.datetime}, processing previous {self.dt}.",
            #         level=DEBUG,
            #     )  # Made more concise
            #     # Some symbols might not have sent a bar for self.dt.
            #     # on_bars needs to handle potentially incomplete self.bars for self.dt
            #     # by using placeholders or only processing available data.
            #     # For simplicity, we assume on_bars will use what's in self.bars.
            #     # If strict all-symbol-bar-receipt is needed, logic here would be more complex.
            #     self.on_bars(
            #         self.dt, self.bars.copy()
            #     )  # Process completed (or partially completed) previous batch

            # Reset for the new batch
            self.dt = bar.datetime
        elif bar.datetime < self.dt:
            # If bar datetime is earlier than current dt, log and skip
            self.write_log(
                f"Received bar with datetime {bar.datetime} earlier than current batch datetime {self.dt}. Skipping.",
                level=ERROR,
            )
            raise ValueError(f"Received bar with datetime {bar.datetime} earlier than current batch datetime {self.dt}. ")

        # bar.datetime must equals to self.dt
        self.bars[bar.vt_symbol] = bar
        self.receiving_status[bar.vt_symbol] = True

        # Check if all *expected* symbols for the current self.dt have arrived
        # This assumes self.vt_symbols is the list of all symbols we expect bars from.
        if all(
                self.receiving_status[sym]
                for sym in self.vt_symbols
                if sym in self.receiving_status
        ):
            # self.write_log(f"All expected bars received for {self.dt}. Processing batch.", level=DEBUG) # Removed, can be noisy
            self.on_bars(self.dt, self.bars.copy())

            # Reset for the next interval after processing
            # self.dt = None
            self.bars.clear()
            self.receiving_status = {sym: False for sym in self.vt_symbols}

    def process_factor_filling_event(self, event: Event) -> None:
        """
        Processes a factor filling event for batch updates with historical data.

        The event.data must contain:
        - start_dt (datetime): Start time for filling
        - end_dt (datetime): End time for filling

        Optional fields:
        - vt_symbols (list[str]): List of symbols to process
        - interval (int): Minutes between calculations (default: 1)

        This method:
        1. Loads historical bar data including required lookback period
        2. Executes factor calculations for each interval
        3. Broadcasts factor events with results

        Raises:
            RuntimeError: If critical errors occur during processing
        """
        # Type and data validation
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            self.write_log("Invalid factor filling event data format", level=ERROR)
            return

        # Reset error state at start
        self.consecutive_errors = 0

        start_dt: datetime | None = event.data.get("start_dt")
        end_dt: datetime | None = event.data.get("end_dt")
        interval: Interval | None = event.data.get("interval", Interval.MINUTE)

        current_dt = start_dt
        total_intervals = int(
            (end_dt - start_dt).total_seconds() / DatetimeUtils.interval2unix(interval, ret_unit='s'))
        processed_count = 0

        self.write_log(
            f"Starting gap filling from {start_dt} to {end_dt} ({total_intervals} intervals)",
            level=INFO)

        # while current_dt <= end_dt:
        for i in tqdm(range(total_intervals + 1), desc="Calculating missing factors", unit="interval"):
        # for i in range(total_intervals + 1):
            # if current_dt > end_dt:
            #     self.write_log(
            #         f"Current datetime {current_dt} exceeds end datetime {end_dt}. Stopping gap filling.",
            #         level=INFO
            #     )
            #     break
            try:
                # 2. Execute factor calculations using Dask
                if self.tasks:
                    self.execute_calculation(
                        dt=current_dt,
                        high_performance=True
                    )  # Results written to FactorMemory, cache updated
                else:
                    self.write_log(
                        "No Dask tasks defined, skipping factor calculation.", level=DEBUG
                    )

                # 3. Broadcast the FactorMemory instances directly
                if self.factor_memory_instances:
                    # Broadcast FactorMemory instances directly
                    event_data = self.latest_calculated_factors_cache
                    # event_data = {k: v.get_latest_rows(1) for k, v in self.factor_memory_instances.items()}
                    # self.event_engine.put(Event(EVENT_FACTOR, event_data))
                    self.event_engine.put(Event(EVENT_FACTOR_FILLING, event_data))
                    self.write_log(f"event_data's first value {next(iter(event_data.values()))}",level=INFO)
                    # print(f"dt{current_dt}")
                else:
                    self.write_log("no calc res", level=WARNING)

            except Exception as e:
                self.write_log(f"Error processing interval {current_dt}: {str(e)}", level=ERROR)
                if self.consecutive_errors >= self.error_threshold:
                    raise RuntimeError(
                        f"Consecutive error threshold reached during gap filling at {current_dt}")

            current_dt += timedelta(minutes=1)  # Move to next interval

        self.write_log(
            f"Gap filling completed. Processed {processed_count} intervals with "
            f"{self.consecutive_errors} errors",
            level=INFO
        )

    def process_load_bar_response_event(self, event: Event) -> None:
        """
        Processes the event to query bar data from the database and initialize it into memory.
        """
        bars = event.data
        if not len(bars):
            self.write_log("No bar data received for factor bar update.", level=WARNING)
            return
        # data conversion
        # Convert list of BarData objects to a dictionary of lists for polars DataFrame
        if isinstance(bars, list) and isinstance(bars[0], BarData):
            data_dict = {
                "datetime": [bar.datetime for bar in bars],
                "open": [bar.open_price for bar in bars],
                "high": [bar.high_price for bar in bars],
                "low": [bar.low_price for bar in bars],
                "close": [bar.close_price for bar in bars],
                "volume": [bar.volume for bar in bars],
                "vt_symbol": [bar.vt_symbol for bar in bars]
            }
            df = pl.DataFrame(data_dict)
        elif isinstance(bars, pd.DataFrame):
            df = pl.from_pandas(bars)
        elif isinstance(bars, pl.DataFrame):
            df = bars
        else:
            self.write_log(
                "Invalid bar data format. Expected list of BarData, pandas DataFrame, or polars DataFrame.",
                level=ERROR
            )
            return

        # Transform DataFrame to wide format with symbols as columns
        memory_start=None
        memory_end=None
        for col in ["open", "high", "low", "close", "volume"]:
            # Group by datetime and create columns for each symbol
            temp_df = df.select(["datetime", "ticker", col])
            temp_df = temp_df.pivot(index='datetime', columns='ticker', values=col)
            temp_df = temp_df.select([self.factor_datetime_col] + list(
                sorted(set(temp_df.columns) - {self.factor_datetime_col, "factor_key"})))
            self.memory_bar[col] = temp_df
            memory_start=temp_df['datetime'].min() if memory_start is None else min(memory_start,temp_df['datetime'].min())
            memory_end=temp_df['datetime'].max() if memory_end is None else max(memory_end,temp_df['datetime'].max())

        self.write_log(f"Initialized bar memory {memory_start}~{memory_end} with {len(df)} bars from database.", level=INFO)

        # start calculation after bar memory is initialized
        self.event_engine.put(Event(EVENT_FACTORMAKER_BAR_READY, data={
            "start_dt": memory_start,
            "end_dt": memory_end,
            "interval": Interval(DatetimeUtils.freq2str(self.minimum_freq))
        }))

    def process_load_factor_response_event(self, event: Event) -> None:
        """
        Processes the event to query factor data from the database and initialize it into memory.
        """
        factors = event.data
        if factors is None or len(factors) == 0:
            self.write_log("No factor data received for factor initialization.", level=WARNING)
            return

        # Convert DataFrame to a dictionary of lists for polars DataFrame
        if isinstance(factors, pd.DataFrame):
            df = pl.from_pandas(factors)
        elif isinstance(factors, pl.DataFrame):
            df = factors
        else:
            self.write_log(
                "Invalid factor data format. Expected pandas DataFrame or polars DataFrame.",
                level=ERROR
            )
            return

        df_unpivoted = df.unpivot(index=['datetime', 'ticker'], on=cs.numeric(), variable_name='factor_key',
                                  value_name='vt_symbol')
        df_pivoted = df_unpivoted.pivot(index=['datetime', 'factor_key'], columns=['ticker'], values='vt_symbol')

        # Process each factor and initialize FactorMemory instances
        temp_df = None
        for group in df_pivoted.group_by("factor_key"):
            factor_key: str = group[0][0]
            if factor_key not in self.flattened_factors:
                self.write_log(
                    f"Factor {factor_key} not found in flattened factors.",
                    level=ERROR
                )
                continue

            memory_instance = self.factor_memory_instances.get(factor_key)

            if not memory_instance:
                fp = self.factor_data_dir.joinpath(f"{safe_filename(factor_key)}.arrow")
                schema = {"datetime": pl.Datetime(time_unit="us")}
                # Allows schema with only datetime if vt_symbols is empty.
                # FactorMemory might require at least one value column depending on its own logic,
                # but this factor will produce an empty value set for such a schema.
                for symbol in self.vt_symbols:
                    schema[symbol] = pl.Float64

                if self.factor_datetime_col not in schema:
                    raise ValueError(
                        f"Factor '{factor_key}' output schema must contain the datetime column '{self.factor_datetime_col}'."
                    )
                memory_instance = FactorMemory(
                    fp, self.max_memory_length_factor, schema, self.factor_datetime_col, FactorMode.BACKTEST
                )

            # Update the FactorMemory instance with the new data
            temp_df = group[1]
            temp_df = temp_df.select([self.factor_datetime_col] + list(
                sorted(set(temp_df.columns) - {self.factor_datetime_col, "factor_key"})))  # Remove factor_key column
            memory_instance.update_data(temp_df)
            self.factor_memory_instances[factor_key] = memory_instance

        self.write_log(f"Initialized factor memory with {len(df)} records from database.", level=INFO)

        # start calculation after factor memory is initialized
        if len(temp_df) <= self.max_factor_lookback + 1:
            self.write_log("Not enough data to start factor calculations. Minimum lookback not met.", level=WARNING)
            return
        start_dt = temp_df[int(self.max_factor_lookback), "datetime"]
        end_dt = temp_df[-1, "datetime"]
        self.event_engine.put(Event(EVENT_FACTORMAKER_FACTOR_READY, data={
            "start_dt": start_dt,
            "end_dt": end_dt,
            "interval": Interval(DatetimeUtils.freq2str(self.minimum_freq))
        }))

    def stop_all_factors(self) -> None:
        self.write_log("Stopping all factors...", level=INFO)
        for (
                factor
        ) in self.flattened_factors.values():  # Stop all, including dependencies
            if factor.trading:
                self.call_factor_func(factor, factor.on_stop)
        self.write_log(f"{len(self.flattened_factors)} factors stopped.", level=INFO)

    def close(self) -> None:
        self.write_log("Closing FactorEngine...", level=INFO)
        self.stop_all_factors()

        # Get a list of settings dictionaries from each "stacked" factor instance
        settings_to_save = [
            f.to_setting() for f in self.stacked_factors.values()
        ]  # CHANGED

        try:
            # save_factor_setting utility should be able to handle a list of dicts
            save_factor_setting(settings_to_save, self.setting_filename)
            self.write_log(
                f"Factor settings saved to {self.setting_filename}", level=INFO
            )
        except Exception as e:
            self.write_log(f"Error saving factor settings: {e}", level=ERROR)

        self.factor_memory_instances.clear()
        self.write_log("FactorEngine closed.", level=INFO)

    def call_factor_func(
            self, factor: FactorTemplate, func: Callable, params: object = None
    ) -> None:
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            # factor.trading = False # on_stop might already do this
            # factor.inited = False # Don't reset inited on every error in a func call
            msg: str = f"Exception in '{func.__name__}' for factor {factor.factor_key}:\n{traceback.format_exc()}"
            self.write_log(msg, factor=factor, level=ERROR)  # Pass factor object

    def write_log(
            self, msg: str, factor: FactorTemplate | None = None, level: int = INFO, show=True
    ) -> None:
        if not show or level < INFO:
            return
        log_msg = f"{factor.factor_key}: {msg}" if factor else f"{msg}"
        log: LogData = LogData(msg=log_msg, gateway_name=APP_NAME, level=level)
        event = Event(type=EVENT_LOG, data=log)
        self.event_engine.put(event)

    # Other utility functions like get_factor_parameters, stop_factor, etc. can be adapted
    # from the original FactorEngine if still needed.
    # For example:
    def get_factor_parameters(self, factor_key: str) -> dict:
        factor = self.flattened_factors.get(factor_key)
        if not factor:
            self.write_log(
                f"Factor {factor_key} not found in flattened factors.", level=ERROR
            )
            return {}
        return factor.get_params()

    def get_factor_instance(self, factor_key: str) -> FactorTemplate | None:
        return self.flattened_factors.get(factor_key)

    def get_factor_memory_instance(self, factor_key: str) -> FactorMemory | None:
        return self.factor_memory_instances.get(factor_key)

    async def get_memory_bar(self):
        """
        Asynchronously retrieves the in-memory bar data.

        Returns:
            dict: A dictionary containing the in-memory bar data.
        """
        return self.memory_bar
