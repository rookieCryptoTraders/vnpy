from dataclasses import dataclass
import gc
import time
from logging import INFO, DEBUG, WARNING, ERROR
from threading import Lock
from pathlib import Path

# Third-party imports
import dask.delayed
import polars as pl
import dask
from dask.delayed import Delayed
import dask.diagnostics

# VnTrader imports
from vnpy.factor.memory import FactorMemory
from vnpy.factor.template import FactorTemplate
from vnpy.factor.base import APP_NAME
from vnpy.factor.setting import get_backtest_data_cache_path

# Default datetime column name
DEFAULT_DATETIME_COL = "datetime"

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(APP_NAME + "_FactorCalculator")


def safe_filename(name: str) -> str:
    import re

    name = re.sub(r"[^\w\.\-@]", "_", name)
    return name


@dataclass
class CalculationMetrics:
    calculation_time: float
    memory_usage: float
    cache_hits: int
    error_count: int


class FactorCalculator:
    """
    Calculates historical values for a single, pre-initialized and pre-flattened
    factor graph, given prepared market data.
    """

    engine_name = APP_NAME + "FactorCalculator"

    def __init__(self, output_data_dir_for_cache: str | None = None) -> None:
        self.output_data_dir: Path = (
            Path(output_data_dir_for_cache)
            if output_data_dir_for_cache
            else get_backtest_data_cache_path()
        )

        # State set per computation run
        self.vt_symbols: list[str] = []
        self.memory_bar: dict[str, pl.DataFrame] = {}
        self.num_data_rows: int = 0

        self.flattened_factors: dict[
            str, FactorTemplate
        ] = {}  # Now an input to compute_factor_values
        self.sorted_factor_keys: list[str] = []
        self.factor_memory_instances: dict[str, FactorMemory] = {}
        self.target_factor_instance: FactorTemplate | None = None
        self.dask_tasks: dict[str, Delayed] = {}
        self.calculation_lock = Lock()
        self.factor_datetime_col: str = DEFAULT_DATETIME_COL

        self._write_log(
            f"FactorCalculator initialized. Factor cache dir: {self.output_data_dir}",
            level=INFO,
        )
        self._prepare_output_directory()

    def _prepare_output_directory(self) -> None:
        try:
            self.output_data_dir.mkdir(parents=True, exist_ok=True)
            self._write_log(
                f"Factor cache directory ensured at: {self.output_data_dir}", level=INFO
            )
        except OSError as e:
            self._write_log(
                f"Error creating factor cache directory {self.output_data_dir}: {e}. Critical error.",
                level=ERROR,
            )
            raise

    def compute_factor_values(
        self,
        target_factor_instance_input: FactorTemplate,
        flattened_factors_input: dict[str, FactorTemplate],  # Pre-flattened graph
        memory_bar_input: dict[str, pl.DataFrame],
        num_data_rows_input: int,
        vt_symbols_for_run: list[str],
    ) -> pl.DataFrame | None:
        """
        Computes values for the given target factor using the provided pre-flattened
        factor graph and market data.

        Args:
            target_factor_instance_input: The fully initialized FactorTemplate instance (target).
            flattened_factors_input: A dictionary of all FactorTemplate instances in the
                                     flattened dependency graph (key: factor_key, value: instance).
            memory_bar_input: Pre-loaded OHLCV data.
            num_data_rows_input: Number of rows in the memory_bar_input.
            vt_symbols_for_run: List of symbols relevant for this computation.

        Returns:
            Polars DataFrame of the calculated target factor values, or None on failure.
        """
        if not isinstance(target_factor_instance_input, FactorTemplate):
            self._write_log(
                "Invalid target_factor_instance_input: Must be a FactorTemplate instance.",
                level=ERROR,
            )
            return None
        if not isinstance(flattened_factors_input, dict) or not all(
            isinstance(f, FactorTemplate) for f in flattened_factors_input.values()
        ):
            self._write_log(
                "Invalid flattened_factors_input: Must be a dict of FactorTemplate instances.",
                level=ERROR,
            )
            return None
        if target_factor_instance_input.factor_key not in flattened_factors_input:
            self._write_log(
                f"Target factor '{target_factor_instance_input.factor_key}' not found in flattened_factors_input.",
                level=ERROR,
            )
            return None

        # Set up internal state for this computation run
        self.target_factor_instance = target_factor_instance_input
        self.flattened_factors = (
            flattened_factors_input  # Use the provided flattened graph
        )
        self.memory_bar = memory_bar_input
        self.num_data_rows = num_data_rows_input
        self.vt_symbols = sorted(list(set(vt_symbols_for_run)))

        if not self.memory_bar or self.num_data_rows == 0:
            self._write_log(
                "Memory bar is empty or num_data_rows is 0. Cannot compute.",
                level=ERROR,
            )
            return None
        if not self.vt_symbols:
            self._write_log("vt_symbols_for_run is empty. Cannot compute.", level=ERROR)
            return None
        # Align vt_symbols in all factors within the flattened_factors_input if they differ.
        # This ensures consistency for the current computation run.
        for factor_key, factor_instance in self.flattened_factors.items():
            if factor_instance.vt_symbols != self.vt_symbols:
                self._write_log(
                    f"Aligning vt_symbols in factor '{factor_key}' to {self.vt_symbols}",
                    level=DEBUG,
                )
                factor_instance.vt_symbols = self.vt_symbols
                # If FactorTemplate._init_dependency_instances needs to be called on symbol change,
                # the BacktestEngine should ensure this happens before passing the flattened graph.
                # Here, we assume the instances are ready or their logic adapts.

        self._write_log(
            f"Starting calculation for: {self.target_factor_instance.factor_key} with symbols {self.vt_symbols}",
            level=INFO,
        )

        if not self._determine_sorted_keys_from_flattened_graph():
            return None
        if not self._initialize_factor_memory():
            return None  # Uses self.sorted_factor_keys and self.flattened_factors
        if not self._build_dask_computational_graph():
            return None
        if not self._execute_batch_dask_graph():
            return None

        target_key = self.target_factor_instance.factor_key
        if target_key in self.factor_memory_instances:
            factor_data_df = self.factor_memory_instances[target_key].get_data()
            if factor_data_df is None:
                self._write_log(
                    f"Data for target factor '{target_key}' is None after retrieval.",
                    level=ERROR,
                )
                return None
            self._write_log(
                f"Successfully retrieved data for '{target_key}'. Shape: {factor_data_df.shape}",
                level=INFO,
            )
            return factor_data_df
        else:
            self._write_log(
                f"FactorMemory for '{target_key}' not found after execution.",
                level=ERROR,
            )
            return None

    # --- Internal Helper Methods ---
    def _determine_sorted_keys_from_flattened_graph(self) -> bool:
        """
        Determines the calculation order (topological sort) from self.flattened_factors.
        """
        if not self.flattened_factors:
            self._write_log(
                "No flattened factors provided. Cannot determine calculation order.",
                level=ERROR,
            )
            return False

        dependency_graph: dict[str, list[str]] = {
            fk: [
                dep.factor_key
                for dep in fi.dependencies_factor
                if isinstance(dep, FactorTemplate)
            ]
            for fk, fi in self.flattened_factors.items()
        }
        try:
            self.sorted_factor_keys = self._topological_sort(dependency_graph)
            self._write_log(
                f"Determined calculation order for {len(self.sorted_factor_keys)} factors.",
                level=DEBUG,
            )
            return True
        except ValueError as e:  # Circular dependency
            self._write_log(
                f"Circular dependency detected in factor graph: {e}", level=ERROR
            )
            return False

    def _initialize_factor_memory(self) -> bool:
        self._write_log(
            f"Initializing factor memory for {len(self.sorted_factor_keys)} factors...",
            level=INFO,
        )
        if self.num_data_rows <= 0:
            self._write_log(
                "num_data_rows is <=0. FactorMemory max_rows=1.", level=WARNING
            )
        max_rows = self.num_data_rows if self.num_data_rows > 0 else 1
        self.factor_memory_instances.clear()
        for key in self.sorted_factor_keys:
            instance = self.flattened_factors.get(
                key
            )  # Get from the pre-flattened input
            if not instance:
                self._write_log(
                    f"Factor instance '{key}' not found in flattened_factors. Skipping memory init.",
                    level=ERROR,
                )
                continue
            try:
                # vt_symbols should have been aligned in compute_factor_values
                if instance.vt_symbols != self.vt_symbols:
                    self._write_log(
                        f"Warning: vt_symbols mismatch for '{key}' during FactorMemory init. "
                        f"Expected {self.vt_symbols}, got {instance.vt_symbols}. "
                        "This might lead to schema issues if schema depends on symbols.",
                        level=WARNING,
                    )

                schema = instance.get_output_schema()
                if self.factor_datetime_col not in schema:
                    raise ValueError(
                        f"Factor '{key}' schema missing datetime col '{self.factor_datetime_col}'."
                    )
                fp = self.output_data_dir.joinpath(f"{safe_filename(key)}.arrow")
                self.factor_memory_instances[key] = FactorMemory(
                    fp, max_rows, schema, self.factor_datetime_col
                )
            except Exception as e:
                self._write_log(
                    f"Failed to init FactorMemory for {key}: {e}", level=ERROR
                )
                return False
        self._write_log(
            f"Initialized {len(self.factor_memory_instances)} FactorMemory instances.",
            level=INFO,
        )
        return True

    def _build_dask_computational_graph(self) -> bool:
        self._write_log(
            f"Building Dask graph for {len(self.sorted_factor_keys)} factors...",
            level=INFO,
        )
        if not self.memory_bar:
            self._write_log("memory_bar empty. Cannot build Dask graph.", level=ERROR)
            return False

        self.dask_tasks.clear()
        for key in self.sorted_factor_keys:
            try:
                self._create_dask_task(
                    key, self.flattened_factors, self.dask_tasks
                )
            except Exception as e:
                self._write_log(
                    f"Error creating Dask task for '{key}': {e}", level=ERROR
                )
                return False
        if not self.dask_tasks and self.flattened_factors:
            self._write_log("Dask graph empty but factors exist.", level=ERROR)
            return False
        self._write_log(
            f"Built Dask graph with {len(self.dask_tasks)} tasks.", level=INFO
        )
        return True

    def _execute_batch_dask_graph(self) -> bool:
        if not self.dask_tasks:
            self._write_log("No Dask tasks to execute.", level=INFO)
            return True
        self._write_log(
            f"Executing Dask graph for {len(self.dask_tasks)} factors...", level=INFO
        )
        with self.calculation_lock:
            start_time = time.time()
            try:
                with dask.diagnostics.ProgressBar():
                    results = dask.compute(
                        *self.dask_tasks.values(),
                        optimize_graph=True,
                        scheduler="threads"
                    )
                self._write_log(
                    f"Dask computation finished in {time.time() - start_time:.3f}s.",
                    level=INFO,
                )
                errors = 0
                for key, df_res in zip(self.dask_tasks.keys(), results, strict=False):
                    if df_res is None or not isinstance(df_res, pl.DataFrame):
                        self._write_log(
                            f"Factor '{key}' computation error or wrong type: {type(df_res)}",
                            level=WARNING,
                        )
                        errors += 1
                        continue
                    fm = self.factor_memory_instances.get(key)
                    if fm:
                        try:
                            fm.update_data(df_res)
                        except Exception as e:
                            self._write_log(
                                f"Factor '{key}' memory update error: {e}", level=ERROR
                            )
                            errors += 1
                    else:
                        self._write_log(
                            f"Factor '{key}' no FactorMemory found.", level=ERROR
                        )
                        errors += 1
                if errors > 0:
                    self._write_log(
                        f"Factor calculations processed with {errors} errors.",
                        level=WARNING,
                    )
                    return False
                return True
            except Exception as e:
                self._write_log(f"Critical Dask error: {e}", level=ERROR)
                return False
            finally:
                self._cleanup_memory_resources()

    def _topological_sort(self, graph: dict[str, list[str]]) -> list[str]:
        visited_perm, visited_temp, order = set(), set(), []
        all_nodes = sorted(list(set(graph.keys()).union(*graph.values())))

        def _visit(node: str):
            if node in visited_perm:
                return
            if node in visited_temp:
                raise ValueError(f"Circular dependency: {node}")
            visited_temp.add(node)
            for dep_node in sorted(graph.get(node, [])):
                _visit(dep_node)
            visited_temp.remove(node)
            visited_perm.add(node)
            order.append(node)

        for n in all_nodes:
            if n not in visited_perm:
                _visit(n)
        return order

    def _get_factor_memory_instance_for_dask(self, factor_key: str) -> FactorMemory:
        if factor_key not in self.factor_memory_instances:
            raise RuntimeError(f"FactorMemory for {factor_key} not found.")
        return self.factor_memory_instances[factor_key]

    def _get_bar_memory_for_dask(self) -> dict[str, pl.DataFrame]:
        return self.memory_bar.copy()

    def _create_dask_task(
        self,
        factor_key: str,
        factors: dict[str, FactorTemplate],
        tasks: dict[str, Delayed]
    ) -> Delayed:
        if factor_key in tasks:
            return tasks[factor_key]
        instance = factors.get(factor_key)  # Get from the provided flattened_factors
        if not instance:
            raise ValueError(
                f"Factor instance for key '{factor_key}' not found in the provided flattened_factors."
            )

        dep_tasks = {
            dep.factor_key: self._create_dask_task(
                dep.factor_key, factors, tasks
            )
            for dep in instance.dependencies_factor
        }
        """mem_delayed = dask.delayed(self._get_factor_memory_instance_for_dask)(
            factor_key
        )"""
        task_in = dask.delayed(self._get_bar_memory_for_dask)() if not instance.dependencies_factor else dep_tasks
        tasks[factor_key] = dask.delayed(instance.calculate)(
            input_data=task_in, memory=None
        )
        return tasks[factor_key]

    def _cleanup_memory_resources(self) -> None:
        gc.collect()
        self._write_log("GC performed.", level=DEBUG)

    def _clear_memory_instances(self) -> None:
        self._cleanup_memory_resources()
        for fm in self.factor_memory_instances.values():
            fm.clear()
        self.factor_memory_instances.clear()
        self._write_log("FactorMemory instances cleared.", level=DEBUG)

    def _write_log(self, msg: str, level: int = INFO) -> None:
        log_msg = f"[{self.engine_name}] {msg}"
        level_map = {
            DEBUG: logger.debug,
            INFO: logger.info,
            WARNING: logger.warning,
            ERROR: logger.error,
        }
        log_func = level_map.get(level, logger.info)
        log_func(log_msg, gateway_name=self.engine_name)

    def close(self) -> None:
        self._write_log("FactorCalculator closed.", level=INFO)
        self._clear_memory_instances()
