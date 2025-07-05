import importlib
from datetime import datetime, timedelta
from logging import INFO, DEBUG, WARNING, ERROR
from typing import Any
from pathlib import Path

# Third-party imports
import pandas as pd
import polars as pl
import numpy as np  # For placeholder data loading

# VnTrader imports
from vnpy.factor.backtesting.factor_analyzer import FactorAnalyser, get_annualization_factor
from vnpy.factor.backtesting.factor_calculator import FactorCalculator
from vnpy.factor.backtesting.factor_initializer import FactorInitializer
from vnpy.factor.template import FactorTemplate
from vnpy.trader.constant import Interval
from vnpy.trader.setting import SETTINGS
from vnpy.factor.base import (
    APP_NAME,
    FactorMode,
)  # FactorMode might be needed for init_factors
from vnpy.factor.utils.factor_utils import (
    init_factors,
    load_factor_setting,
)  # For initializing factors

# Assuming FactorCalculator and FactorAnalyser are in sibling files
# (e.g., factor_calculator.py, factor_analyser.py) in the same directory.
# If your project structure is different (e.g., a package), adjust imports accordingly.


# Default datetime column name
DEFAULT_DATETIME_COL = "datetime"

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(APP_NAME + "_BacktestEngine")


class BacktestEngine:
    """
    Orchestrates factor calculation and analysis for single factor backtesting.
    It initializes the factor graph, loads data, uses FactorCalculator to compute
    factor values, and FactorAnalyser to analyse and report results.
    """

    engine_name = APP_NAME + "BacktestOrchestrator"

    def __init__(
        self,
        factor_module_name: str | None = None,
        output_data_dir_for_calculator_cache: str | None = None,
        output_data_dir_for_analyser_reports: str | None = None,
    ):
        self.factor_module_name: str = factor_module_name or SETTINGS.get(
            "factor.module_name", "vnpy.factor.factors"
        )
        self.output_data_dir_for_calculator_cache = output_data_dir_for_calculator_cache
        self.output_data_dir_for_analyser_reports = output_data_dir_for_analyser_reports

        # Data loaded by the engine
        self.memory_bar: dict[str, pl.DataFrame] = {}
        self.num_data_rows: int = 0
        self.factor_datetime_col: str = DEFAULT_DATETIME_COL

        self._write_log(f"{self.engine_name} initialized.", level=INFO)

    def _load_bar_data_engine(
        self, start: datetime, end: datetime, interval: Interval, vt_symbols: list[str]
    ) -> bool:
        """
        Loads historical OHLCV bar data for the orchestrator.
        Populates self.memory_bar and self.num_data_rows.
        THIS IS A PLACEHOLDER. Replace with actual data loading logic.
        """
        self._write_log(
            f"Loading bar data for {len(vt_symbols)} symbols from {start} to {end} ({interval.value}).",
            level=INFO,
        )
        self.memory_bar.clear()

        if not vt_symbols:
            self._write_log("No vt_symbols provided for data loading.", level=WARNING)
            self.num_data_rows = 0
            return False

        delta = end - start
        if interval == Interval.MINUTE:
            expected_rows = int(delta.total_seconds() / 60)
        elif interval == Interval.HOUR:
            expected_rows = int(delta.total_seconds() / 3600)
        elif interval == Interval.DAILY:
            expected_rows = delta.days
        else:
            self._write_log(
                f"Unsupported interval '{interval}' for placeholder data generation.",
                level=ERROR,
            )
            return False

        if expected_rows <= 0:
            self._write_log(
                f"Date range {start} to {end} with interval {interval.value} results in <=0 expected rows. No data loaded.",
                level=WARNING,
            )
            self.num_data_rows = 0
            return True

        datetime_values = []
        current_dt = start
        while current_dt <= end and len(datetime_values) < expected_rows:
            datetime_values.append(current_dt)
            if interval == Interval.MINUTE:
                current_dt += timedelta(minutes=1)
            elif interval == Interval.HOUR:
                current_dt += timedelta(hours=1)
            elif interval == Interval.DAILY:
                current_dt += timedelta(days=1)
            else:
                break

        self.num_data_rows = len(datetime_values)

        if not datetime_values:
            self.num_data_rows = 0
            self._write_log(
                "No datetime values generated. Check start/end dates and interval.",
                level=WARNING,
            )
            return True

        dt_series = pl.Series(
            self.factor_datetime_col, datetime_values, dtype=pl.Datetime(time_unit="us")
        )

        for bar_field in ["open", "high", "low", "close", "volume"]:
            data_for_field = {self.factor_datetime_col: dt_series}
            for symbol in vt_symbols:
                if bar_field == "volume":
                    data_for_field[symbol] = np.random.randint(
                        100, 10000, size=self.num_data_rows
                    )
                else:
                    base_price = np.random.rand() * 100 + 50
                    open_prices = (
                        base_price + np.random.randn(self.num_data_rows).cumsum() * 0.1
                    )
                    close_prices = (
                        open_prices + np.random.randn(self.num_data_rows) * 0.05
                    )

                    _low = np.minimum(open_prices, close_prices)
                    _high = np.maximum(open_prices, close_prices)

                    low_prices = _low - np.abs(
                        np.random.randn(self.num_data_rows) * 0.02 * _low
                    )
                    high_prices = _high + np.abs(
                        np.random.randn(self.num_data_rows) * 0.02 * _high
                    )

                    low_prices = np.minimum(low_prices, open_prices)
                    low_prices = np.minimum(low_prices, close_prices)
                    high_prices = np.maximum(high_prices, open_prices)
                    high_prices = np.maximum(high_prices, close_prices)

                    if bar_field == "open":
                        data_for_field[symbol] = open_prices
                    elif bar_field == "high":
                        data_for_field[symbol] = high_prices
                    elif bar_field == "low":
                        data_for_field[symbol] = low_prices
                    elif bar_field == "close":
                        data_for_field[symbol] = close_prices
            try:
                self.memory_bar[bar_field] = pl.DataFrame(data_for_field)
            except Exception as e:
                self._write_log(
                    f"Error creating DataFrame for field '{bar_field}': {e}",
                    level=ERROR,
                )
                return False

        if "close" not in self.memory_bar or self.memory_bar["close"].is_empty():
            self._write_log(
                "'close' data is missing or empty after simulated loading.",
                level=ERROR,
            )
            self.num_data_rows = 0
            return False

        self.num_data_rows = self.memory_bar["close"].height
        if self.num_data_rows == 0:
            self._write_log(
                "Loaded 'close' data is empty. No data rows available.",
                level=WARNING,
            )
            return True

        self._write_log(
            f"Successfully simulated loading of {self.num_data_rows} rows for {len(vt_symbols)} symbols.",
            level=DEBUG,
        )
        return True

    def _init_and_flatten_factor(
        self,
        factor_definition: FactorTemplate | dict | str,
        vt_symbols_for_factor: list[str],
        factor_json_conf_path: str | None = None,
    ) -> tuple[FactorTemplate | None, dict[str, FactorTemplate] | None]:
        """
        Initializes the target factor and flattens its dependency tree using FactorInitializer.
        """
        initializer = FactorInitializer(
            factor_module_name=self.factor_module_name,
            factor_json_conf_path=factor_json_conf_path,
        )
        return initializer.init_and_flatten(
            factor_definition=factor_definition,
            vt_symbols_for_factor=vt_symbols_for_factor,
        )

    def _create_calculator(self) -> FactorCalculator:
        """Creates and returns a FactorCalculator instance."""
        # FactorCalculator is now simpler and doesn't take vt_symbols or factor_module_name in init
        return FactorCalculator(
            output_data_dir_for_cache=self.output_data_dir_for_calculator_cache
        )

    def _run_factor_computation(
        self,
        calculator: FactorCalculator,
        target_factor_instance: FactorTemplate,
        flattened_factors: dict[str, FactorTemplate],
        vt_symbols_for_run: list[str],
        data_to_use: dict[str, pl.DataFrame],
    ) -> pl.DataFrame | None:
        """Uses the calculator to compute factor values on the given data."""
        self._write_log("Starting factor computation...", level=INFO)

        if "close" not in data_to_use or data_to_use["close"].is_empty():
            self._write_log("No data provided for computation.", level=WARNING)
            return None

        num_rows = data_to_use["close"].height

        factor_df = calculator.compute_factor_values(
            target_factor_instance_input=target_factor_instance,
            flattened_factors_input=flattened_factors,
            memory_bar_input=data_to_use,
            num_data_rows_input=num_rows,
            vt_symbols_for_run=vt_symbols_for_run,
        )
        return factor_df

    def _run_factor_analysis(
        self,
        factor_df: pl.DataFrame,
        market_close_prices_df: pl.DataFrame,  # Pass only the close prices needed
        target_factor_instance: FactorTemplate,
        analysis_start_dt: datetime,
        analysis_end_dt: datetime,
        num_quantiles: int,
        long_short_percentile: float,
        report_filename_prefix: str,
    ) -> Path | None:
        """Uses the analyser to process results and generate a report."""
        self._write_log("Starting factor analysis...", level=INFO)
        analyser = FactorAnalyser(
            output_data_dir_for_reports=self.output_data_dir_for_analyser_reports
        )

        analyser.annualization_factor = get_annualization_factor(market_close_prices_df[DEFAULT_DATETIME_COL])

        self._write_log(
            f"annualization_factor set to {analyser.annualization_factor} based on market close prices.",
            level=DEBUG,
        )

        if market_close_prices_df.is_empty():
            self._write_log(
                "Market close prices missing for analysis. Aborting analysis.",
                level=ERROR,
            )
            analyser.close()
            return None

        report_path = analyser.run_analysis_and_report(
            factor_data_df=factor_df,
            market_close_prices_df=market_close_prices_df,
            factor_instance=target_factor_instance,
            analysis_start_dt=analysis_start_dt,
            analysis_end_dt=analysis_end_dt,
            num_quantiles=num_quantiles,
            long_short_percentile=long_short_percentile,
            report_filename_prefix=report_filename_prefix,
        )

        analyser.close()
        return report_path

    def run_single_factor_backtest(
        self,
        factor_definition: FactorTemplate | dict | str,
        start_datetime: datetime,
        end_datetime: datetime,
        vt_symbols_for_factor: list[str],
        factor_json_conf_path: str | None = None,
        data_interval: Interval = Interval.MINUTE,
        # Analysis parameters
        num_quantiles: int = 5,
        long_short_percentile: float = 0.5,
        report_filename_prefix: str = "factor_analysis_report",
    ) -> Path | None:
        """
        Runs a complete single factor backtest by coordinating FactorCalculator and FactorAnalyser.
        """
        # Step 1: Load Data
        if not self._load_bar_data_engine(
            start_datetime, end_datetime, data_interval, vt_symbols_for_factor
        ):
            self._write_log("Data loading failed. Aborting backtest.", level=ERROR)
            return None
        self._write_log("Data loading complete.", level=INFO)

        # Step 2: Initialize Factor
        target_factor_instance, flattened_factors = self._init_and_flatten_factor(
            factor_definition=factor_definition,
            vt_symbols_for_factor=vt_symbols_for_factor,
            factor_json_conf_path=factor_json_conf_path,
        )

        if not target_factor_instance or not flattened_factors:
            self._write_log(
                "Factor initialization failed. Aborting backtest.", level=ERROR
            )
            return None
        self._write_log(
            f"Running single factor backtest for {target_factor_instance.factor_key}",
            level=INFO,
        )

        # Step 3: Calculate Factor Values
        calculator = self._create_calculator()

        factor_df = self._run_factor_computation(
            calculator=calculator,
            target_factor_instance=target_factor_instance,
            flattened_factors=flattened_factors,
            vt_symbols_for_run=vt_symbols_for_factor,
            data_to_use=self.memory_bar,
        )
        calculator.close()  # Close calculator after computation is done

        if factor_df is None:
            self._write_log(
                "Factor calculation failed. Aborting analysis.", level=ERROR
            )
            return None
        self._write_log("Factor calculation complete.", level=INFO)

        # Prepare market data (close prices) for the analyser, aligned with factor_df
        market_close_prices_df: pl.DataFrame | None = None
        if "close" in self.memory_bar and not self.memory_bar["close"].is_empty():
            if not factor_df.is_empty():
                # Align close_prices with the factor_df's datetime index
                market_close_prices_df = self.memory_bar["close"].join(
                    factor_df.select(pl.col(self.factor_datetime_col)),
                    on=self.factor_datetime_col,
                    how="inner",
                )
            else:  # factor_df is empty, use all loaded close prices for the period
                market_close_prices_df = self.memory_bar["close"].clone()

        if market_close_prices_df is None or market_close_prices_df.is_empty():
            self._write_log(
                "Aligned market close prices are not available or empty. Aborting analysis.",
                level=ERROR,
            )
            return None

        # Step 4: Analyse Factor Results
        self._write_log("Analyzing factor performance.", level=INFO)
        actual_analysis_start_dt = start_datetime
        actual_analysis_end_dt = end_datetime
        if not factor_df.is_empty():
            try:
                min_dt_val = factor_df.select(pl.col(DEFAULT_DATETIME_COL).min()).item()
                max_dt_val = factor_df.select(pl.col(DEFAULT_DATETIME_COL).max()).item()
                if isinstance(min_dt_val, datetime|pd.Timestamp):
                    actual_analysis_start_dt = (
                        pd.to_datetime(min_dt_val).to_pydatetime()
                        if isinstance(min_dt_val, pd.Timestamp)
                        else min_dt_val
                    )
                if isinstance(max_dt_val, datetime|pd.Timestamp):
                    actual_analysis_end_dt = (
                        pd.to_datetime(max_dt_val).to_pydatetime()
                        if isinstance(max_dt_val, pd.Timestamp)
                        else max_dt_val
                    )
            except Exception as e_dt:
                self._write_log(
                    f"Could not derive precise start/end from factor_df: {e_dt}. Using original period.",
                    WARNING,
                )

        report_path = self._run_factor_analysis(
            factor_df=factor_df,
            market_close_prices_df=market_close_prices_df,
            target_factor_instance=target_factor_instance,
            analysis_start_dt=actual_analysis_start_dt,
            analysis_end_dt=actual_analysis_end_dt,
            num_quantiles=num_quantiles,
            long_short_percentile=long_short_percentile,
            report_filename_prefix=report_filename_prefix,
        )

        if report_path:
            self._write_log(
                f"Backtest and analysis complete. Report: {report_path}", level=INFO
            )
        else:
            self._write_log("Analysis and reporting failed.", level=WARNING)

        return report_path

    def run_train_test_backtest(
        self,
        factor_definition: FactorTemplate | dict | str,
        start_datetime: datetime,
        end_datetime: datetime,
        vt_symbols_for_factor: list[str],
        factor_json_conf_path: str | None = None,
        data_interval: Interval = Interval.MINUTE,
        test_size_ratio: float = 0.3,
        num_quantiles: int = 5,
        long_short_percentile: float = 0.5,
        report_filename_prefix: str = "factor_analysis_report",
    ) -> tuple[Path | None, Path | None]:
        """
        Runs a backtest with separate training and testing sets.

        It loads the full data, splits it, calculates the factor on each subset,
        and generates separate analysis reports for both training and testing periods.

        Args:
            factor_definition: The factor to be backtested.
            start_datetime: The start date for the entire data period.
            end_datetime: The end date for the entire data period.
            vt_symbols_for_factor: List of symbols for the backtest.
            factor_json_conf_path: Path to factor configuration JSON.
            data_interval: The interval of the historical data.
            test_size_ratio: The proportion of data to use for the test set.
            num_quantiles: Number of quantiles for factor analysis.
            long_short_percentile: Percentile for long/short portfolio construction.
            report_filename_prefix: Prefix for the report filenames.

        Returns:
            A tuple containing the paths to the training report and the testing report.
        """
        # 1. Load all necessary data
        if not self._load_bar_data_engine(
            start_datetime, end_datetime, data_interval, vt_symbols_for_factor
        ):
            self._write_log("Data loading failed, aborting backtest.", level=ERROR)
            return None, None

        # 2. Split data into training and testing sets
        try:
            train_data, test_data = self._split_data(test_size_ratio)
            train_rows = train_data["close"].height
            test_rows = test_data["close"].height
            self._write_log(
                f"Data split into {train_rows} train rows and {test_rows} test rows.",
                level=INFO,
            )
        except ValueError as e:
            self._write_log(f"Error splitting data: {e}", level=ERROR)
            return None, None

        # 3. Initialize the factor
        target_factor, flattened_factors = self._init_and_flatten_factor(
            factor_definition=factor_definition,
            vt_symbols_for_factor=vt_symbols_for_factor,
            factor_json_conf_path=factor_json_conf_path,
        )
        if not target_factor or not flattened_factors:
            self._write_log("Factor initialization failed.", level=ERROR)
            return None, None

        # 4. Process Training Data
        self._write_log("Processing training data...", level=INFO)
        train_report = self._process_data_subset(
            target_factor,
            flattened_factors,
            train_data,
            vt_symbols_for_factor,
            num_quantiles,
            long_short_percentile,
            f"{report_filename_prefix}_train",
        )

        # 5. Process Testing Data
        self._write_log("Processing testing data...", level=INFO)
        test_report = self._process_data_subset(
            target_factor,
            flattened_factors,
            test_data,
            vt_symbols_for_factor,
            num_quantiles,
            long_short_percentile,
            f"{report_filename_prefix}_test",
        )

        return train_report, test_report

    def _process_data_subset(
        self,
        target_factor: FactorTemplate,
        flattened_factors: dict[str, FactorTemplate],
        data_subset: dict[str, pl.DataFrame],
        vt_symbols: list[str],
        num_quantiles: int,
        long_short_percentile: float,
        report_prefix: str,
    ) -> Path | None:
        """
        A helper function to run factor computation and analysis on a subset of data.
        """
        if data_subset["close"].is_empty():
            self._write_log(f"Data subset for '{report_prefix}' is empty.", WARNING)
            return None

        calculator = self._create_calculator()
        factor_df = self._run_factor_computation(
            calculator, target_factor, flattened_factors, vt_symbols, data_subset
        )
        calculator.close()

        if factor_df is None or factor_df.is_empty():
            self._write_log(f"Factor calculation failed for '{report_prefix}'.", WARNING)
            return None

        start_dt = data_subset["close"][self.factor_datetime_col].min()
        end_dt = data_subset["close"][self.factor_datetime_col].max()

        report_path = self._run_factor_analysis(
            factor_df=factor_df,
            market_close_prices_df=data_subset["close"],
            target_factor_instance=target_factor,
            analysis_start_dt=start_dt,
            analysis_end_dt=end_dt,
            num_quantiles=num_quantiles,
            long_short_percentile=long_short_percentile,
            report_filename_prefix=report_prefix,
        )
        return report_path

    def _split_data(
        self, test_size_ratio: float
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        """Splits the loaded data into training and testing sets."""
        if not self.memory_bar or "close" not in self.memory_bar or self.memory_bar["close"].is_empty():
            raise ValueError("Memory bar is not loaded or 'close' data is missing.")

        total_rows = self.num_data_rows
        if total_rows == 0:
            raise ValueError("No data available to split.")

        train_size = int(total_rows * (1 - test_size_ratio))
        train_data, test_data = {}, {}

        for key, df in self.memory_bar.items():
            if isinstance(df, pl.DataFrame) and not df.is_empty():
                train_data[key] = df.slice(0, train_size)
                test_data[key] = df.slice(train_size)
            else:
                train_data[key] = df
                test_data[key] = df

        return train_data, test_data

    def _write_log(self, msg: str, level: int = INFO) -> None:
        level_map = {
            DEBUG: logger.debug,
            INFO: logger.info,
            WARNING: logger.warning,
            ERROR: logger.error,
        }
        log_func = level_map.get(level, logger.info)
        log_func(msg, gateway_name=self.engine_name)
