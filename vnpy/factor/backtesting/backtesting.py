import importlib
from datetime import datetime, timedelta
from logging import INFO, DEBUG, WARNING, ERROR 
from typing import Any, Dict, Optional, List, Union, Tuple
from pathlib import Path

# Third-party imports
import pandas as pd
import polars as pl
import numpy as np # For placeholder data loading

# VnTrader imports
from vnpy.factor.backtesting.factor_analyzer import FactorAnalyser
from vnpy.factor.backtesting.factor_calculator import FactorCalculator
from vnpy.factor.template import FactorTemplate
from vnpy.trader.constant import Interval
from vnpy.trader.setting import SETTINGS
from vnpy.factor.base import APP_NAME, FactorMode # FactorMode might be needed for init_factors
from vnpy.factor.utils.factor_utils import init_factors, load_factor_setting # For initializing factors

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
        factor_module_name: Optional[str] = None,
        output_data_dir_for_calculator_cache: Optional[str] = None,
        output_data_dir_for_analyser_reports: Optional[str] = None,
    ):
        self.factor_module_name: str = factor_module_name or SETTINGS.get('factor.module_name', 'vnpy_portfoliostrategy.factors')
        self.output_data_dir_for_calculator_cache = output_data_dir_for_calculator_cache
        self.output_data_dir_for_analyser_reports = output_data_dir_for_analyser_reports
        
        self.module_factors: Optional[Any] = None
        try:
            self.module_factors = importlib.import_module(self.factor_module_name)
            self._write_log(f"Successfully imported factor module: '{self.factor_module_name}'", level=INFO)
        except ImportError as e:
            self._write_log(f"Could not import factor module '{self.factor_module_name}': {e}. Factor initialization will fail.", level=ERROR)
            raise

        # Data loaded by the engine
        self.memory_bar: Dict[str, pl.DataFrame] = {}
        self.num_data_rows: int = 0
        self.factor_datetime_col: str = DEFAULT_DATETIME_COL

        self._write_log(f"{self.engine_name} initialized.", level=INFO)

    def _load_bar_data_engine(self, 
        start: datetime, 
        end: datetime, 
        interval: Interval, 
        vt_symbols: List[str]
    ) -> bool:
        """
        Loads historical OHLCV bar data for the orchestrator.
        Populates self.memory_bar and self.num_data_rows.
        THIS IS A PLACEHOLDER. Replace with actual data loading logic.
        """
        self._write_log(f"Orchestrator Placeholder: load_bar_data called for {start} to {end}, interval {interval.value}, symbols: {vt_symbols}", level=INFO)
        self.memory_bar.clear() 

        if not vt_symbols:
            self._write_log("No vt_symbols provided for data loading.", level=WARNING)
            self.num_data_rows = 0
            return False 

        delta = end - start
        if interval == Interval.MINUTE: expected_rows = int(delta.total_seconds() / 60)
        elif interval == Interval.HOUR: expected_rows = int(delta.total_seconds() / 3600)
        elif interval == Interval.DAILY: expected_rows = delta.days
        else: 
            self._write_log(f"Unsupported interval '{interval}' for placeholder data generation.", level=ERROR)
            return False

        if expected_rows <= 0:
            self._write_log(f"Date range {start} to {end} with interval {interval.value} results in <=0 expected rows. No data loaded.", level=WARNING)
            self.num_data_rows = 0
            return True 

        datetime_values = []
        current_dt = start
        while current_dt <= end and len(datetime_values) < expected_rows :
            datetime_values.append(current_dt)
            if interval == Interval.MINUTE: current_dt += timedelta(minutes=1)
            elif interval == Interval.HOUR: current_dt += timedelta(hours=1)
            elif interval == Interval.DAILY: current_dt += timedelta(days=1)
            else: break 
        
        self.num_data_rows = len(datetime_values) 

        if not datetime_values: 
             self.num_data_rows = 0
             self._write_log("Placeholder: No datetime values generated. Check start/end dates and interval.", level=WARNING)
             return True 

        dt_series = pl.Series(self.factor_datetime_col, datetime_values, dtype=pl.Datetime(time_unit="us"))

        for bar_field in ["open", "high", "low", "close", "volume"]:
            data_for_field = {self.factor_datetime_col: dt_series}
            for symbol in vt_symbols: 
                if bar_field == "volume":
                    data_for_field[symbol] = np.random.randint(100, 10000, size=self.num_data_rows)
                else:
                    base_price = np.random.rand() * 100 + 50 
                    open_prices = base_price + np.random.randn(self.num_data_rows).cumsum() * 0.1
                    close_prices = open_prices + np.random.randn(self.num_data_rows) * 0.05 
                    
                    _low = np.minimum(open_prices, close_prices)
                    _high = np.maximum(open_prices, close_prices)
                    
                    low_prices = _low - np.abs(np.random.randn(self.num_data_rows) * 0.02 * _low) 
                    high_prices = _high + np.abs(np.random.randn(self.num_data_rows) * 0.02 * _high) 
                    
                    low_prices = np.minimum(low_prices, open_prices)
                    low_prices = np.minimum(low_prices, close_prices)
                    high_prices = np.maximum(high_prices, open_prices)
                    high_prices = np.maximum(high_prices, close_prices)

                    if bar_field == "open": data_for_field[symbol] = open_prices
                    elif bar_field == "high": data_for_field[symbol] = high_prices
                    elif bar_field == "low": data_for_field[symbol] = low_prices
                    elif bar_field == "close": data_for_field[symbol] = close_prices
            try:
                self.memory_bar[bar_field] = pl.DataFrame(data_for_field)
            except Exception as e:
                 self._write_log(f"Placeholder: Error creating DataFrame for field '{bar_field}': {e}", level=ERROR)
                 return False

        if "close" not in self.memory_bar or self.memory_bar["close"].is_empty():
            self._write_log("Placeholder: 'close' data is missing or empty after simulated loading.", level=ERROR)
            self.num_data_rows = 0 
            return False

        self.num_data_rows = self.memory_bar["close"].height 
        if self.num_data_rows == 0:
             self._write_log("Placeholder: Loaded 'close' data is empty. No data rows available.", level=WARNING)
             return True 

        self._write_log(f"Orchestrator Placeholder: Successfully simulated loading of {self.num_data_rows} rows for {len(vt_symbols)} symbols.", level=INFO)
        return True

    def _init_and_flatten_factor(
        self,
        factor_definition: Union[FactorTemplate, Dict, str],
        vt_symbols_for_factor: List[str],
        factor_json_conf_path: Optional[str] = None
    ) -> Tuple[Optional[FactorTemplate], Optional[Dict[str, FactorTemplate]]]:
        """
        Initializes the target factor and flattens its dependency tree.
        Returns the target factor instance and the dictionary of all flattened factors.
        """
        self._write_log(f"Initializing and flattening factor based on definition. Symbols: {vt_symbols_for_factor}", level=INFO)
        if not self.module_factors:
            self._write_log("Factor module not loaded. Cannot initialize factor.", level=ERROR)
            return None, None

        target_factor_instance: Optional[FactorTemplate] = None
        
        # Logic to get a single FactorTemplate instance based on factor_definition
        if isinstance(factor_definition, FactorTemplate):
            target_factor_instance = factor_definition
            # Ensure vt_symbols are aligned if provided
            if target_factor_instance.params.get_parameter('vt_symbols') != vt_symbols_for_factor:
                target_factor_instance.params.set_parameters({'vt_symbols': vt_symbols_for_factor})
                target_factor_instance._init_dependency_instances() # Re-initialize dependencies with new symbols
        elif isinstance(factor_definition, dict):
            setting_copy = factor_definition.copy()
            setting_copy["factor_mode"] = FactorMode.BACKTEST.name
            if "params" not in setting_copy: setting_copy["params"] = {}
            setting_copy["params"]["vt_symbols"] = vt_symbols_for_factor
            try:
                inited_list = init_factors(self.module_factors, [setting_copy], self.module_factors)
                if inited_list: target_factor_instance = inited_list[0]
            except Exception as e:
                self._write_log(f"Error initializing factor from dict: {e}", level=ERROR); return None, None
        elif isinstance(factor_definition, str): # factor_key
            if not factor_json_conf_path:
                self._write_log("factor_json_conf_path needed for factor_key definition.", level=ERROR); return None, None
            try:
                all_settings = load_factor_setting(str(factor_json_conf_path))
                found_setting = None
                for setting_item_outer in all_settings:
                    setting_item = setting_item_outer
                    if len(setting_item_outer) == 1 and isinstance(list(setting_item_outer.values())[0], dict) and "class_name" in list(setting_item_outer.values())[0]:
                        setting_item = list(setting_item_outer.values())[0]

                    temp_setting = setting_item.copy()
                    temp_setting["factor_mode"] = FactorMode.BACKTEST.name
                    if "params" not in temp_setting: temp_setting["params"] = {}
                    temp_setting["params"]["vt_symbols"] = vt_symbols_for_factor
                    
                    TempFactorClass = getattr(self.module_factors, temp_setting["class_name"])
                    temp_instance = TempFactorClass(setting=temp_setting, dependencies_module_lookup=self.module_factors)
                    if temp_instance.factor_key == factor_definition:
                        found_setting = setting_item.copy(); break
                if found_setting:
                    final_setting = found_setting
                    final_setting["factor_mode"] = FactorMode.BACKTEST.name
                    if "params" not in final_setting: final_setting["params"] = {}
                    final_setting["params"]["vt_symbols"] = vt_symbols_for_factor
                    inited_list = init_factors(self.module_factors, [final_setting], self.module_factors)
                    if inited_list: target_factor_instance = inited_list[0]
                else:
                    self._write_log(f"Factor key '{factor_definition}' not found in '{factor_json_conf_path}'.", level=ERROR)
                    return None, None
            except Exception as e:
                self._write_log(f"Error initializing factor from key '{factor_definition}': {e}", level=ERROR); return None, None
        else:
            self._write_log(f"Invalid factor_definition type: {type(factor_definition)}", level=ERROR); return None, None

        if not target_factor_instance:
            self._write_log("Failed to create target_factor_instance.", level=ERROR); return None, None
        
        self._write_log(f"Target factor instance created: {target_factor_instance.factor_key}", level=INFO)

        # Flatten the dependency tree
        stacked_factors = {target_factor_instance.factor_key: target_factor_instance}
        flattened_factors = self._complete_factor_tree(stacked_factors)

        if not flattened_factors:
            self._write_log("Failed to flatten factor tree.", level=ERROR); return None, None
        
        self._write_log(f"Factor tree flattened. Total factors in graph: {len(flattened_factors)}", level=DEBUG)
        return target_factor_instance, flattened_factors

    def _complete_factor_tree(self, root_factors: Dict[str, FactorTemplate]) -> Dict[str, FactorTemplate]:
        """
        Helper method to recursively traverse dependencies and build a flat dictionary
        of all unique FactorTemplate instances.
        """
        resolved_factors: Dict[str, FactorTemplate] = {}
        def traverse(factor: FactorTemplate):
            if factor.factor_key in resolved_factors:
                return
            for dep_instance in factor.dependencies_factor: # These are already FactorTemplate instances
                if not isinstance(dep_instance, FactorTemplate):
                     # This case should ideally not be hit if init_factors correctly resolves dependencies.
                    self._write_log(f"Warning: Dependency for {factor.factor_key} is not a FactorTemplate instance: {type(dep_instance)}. This might indicate an issue in how FactorTemplate initializes its dependencies_factor list.", level=WARNING)
                    continue
                traverse(dep_instance)
            resolved_factors[factor.factor_key] = factor

        for _, factor_instance in root_factors.items():
            traverse(factor_instance)
        return resolved_factors


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
        flattened_factors: Dict[str, FactorTemplate],
        vt_symbols_for_run: List[str] # Pass the current run's symbols
    ) -> Optional[pl.DataFrame]:
        """Uses the calculator to compute factor values using its loaded data."""
        self._write_log("Starting factor value computation phase...", level=INFO)
        
        factor_df = calculator.compute_factor_values(
            target_factor_instance_input=target_factor_instance,
            flattened_factors_input=flattened_factors,
            memory_bar_input=self.memory_bar, # Pass orchestrator's loaded data
            num_data_rows_input=self.num_data_rows,
            vt_symbols_for_run=vt_symbols_for_run 
        )
        # calculator.close() # Calculator can be closed after all computations if it were long-lived
        return factor_df


    def _run_factor_analysis(
        self,
        factor_df: pl.DataFrame,
        market_close_prices_df: pl.DataFrame, # Pass only the close prices needed
        target_factor_instance: FactorTemplate,
        analysis_start_dt: datetime, 
        analysis_end_dt: datetime,   
        num_quantiles: int,
        returns_look_ahead_period: int,
        long_percentile_threshold: float,
        short_percentile_threshold: float,
        report_filename_prefix: str
    ) -> Optional[Path]:
        """Uses the analyser to process results and generate a report."""
        self._write_log("Starting factor analysis phase...", level=INFO)
        analyser = FactorAnalyser(
            output_data_dir_for_reports=self.output_data_dir_for_analyser_reports
        )
        
        if market_close_prices_df.is_empty():
            self._write_log("Market close prices missing for analysis. Aborting analysis.", level=ERROR)
            analyser.close()
            return None

        report_path = analyser.run_analysis_and_report(
            factor_data_df=factor_df,
            market_close_prices_df=market_close_prices_df,
            factor_instance=target_factor_instance,
            analysis_start_dt=analysis_start_dt,
            analysis_end_dt=analysis_end_dt,
            num_quantiles=num_quantiles,
            returns_look_ahead_period=returns_look_ahead_period,
            long_percentile_threshold=long_percentile_threshold,
            short_percentile_threshold=short_percentile_threshold,
            report_filename_prefix=report_filename_prefix
        )
        analyser.close()
        return report_path

    def run_single_factor_backtest(
        self,
        factor_definition: Union[FactorTemplate, Dict, str],
        start_datetime: datetime,
        end_datetime: datetime,
        vt_symbols_for_factor: List[str],
        factor_json_conf_path: Optional[str] = None,
        data_interval: Interval = Interval.MINUTE,
        # Analysis parameters
        num_quantiles: int = 5,
        returns_look_ahead_period: int = 1,
        long_percentile_threshold: float = 0.7,
        short_percentile_threshold: float = 0.3,
        report_filename_prefix: str = "factor_analysis_report"
    ) -> Optional[Path]:
        """
        Runs a complete single factor backtest by coordinating FactorCalculator and FactorAnalyser.
        """
        self._write_log(f"Orchestrating single factor backtest for symbols: {vt_symbols_for_factor}", level=INFO)

        # Step 1: Load Data (Orchestrator handles this)
        if not self._load_bar_data_engine(start_datetime, end_datetime, data_interval, vt_symbols_for_factor):
            self._write_log("Failed to load bar data in orchestrator. Aborting.", ERROR)
            return None
        if self.num_data_rows == 0:
            self._write_log("No bar data rows loaded by orchestrator. Aborting.", WARNING)
            return None

        # Step 2: Initialize Factor and Flatten Dependency Tree
        target_factor_instance, flattened_factors = self._init_and_flatten_factor(
            factor_definition=factor_definition,
            vt_symbols_for_factor=vt_symbols_for_factor,
            factor_json_conf_path=factor_json_conf_path
        )
        if not target_factor_instance or not flattened_factors:
            self._write_log("Failed to initialize or flatten factor. Aborting.", level=ERROR)
            return None

        # Step 3: Calculate Factor Values
        calculator = self._create_calculator() 
        
        factor_df = self._run_factor_computation(
            calculator=calculator,
            target_factor_instance=target_factor_instance,
            flattened_factors=flattened_factors,
            vt_symbols_for_run=vt_symbols_for_factor # Use the symbols for this specific run
        )
        calculator.close() # Close calculator after computation is done

        if factor_df is None:
            self._write_log("Factor calculation failed. Aborting analysis.", level=ERROR)
            return None
        
        # Prepare market data (close prices) for the analyser, aligned with factor_df
        market_close_prices_df: Optional[pl.DataFrame] = None
        if "close" in self.memory_bar and not self.memory_bar["close"].is_empty():
            if not factor_df.is_empty():
                 # Align close_prices with the factor_df's datetime index
                market_close_prices_df = self.memory_bar["close"].join(
                    factor_df.select(pl.col(self.factor_datetime_col)),
                    on=self.factor_datetime_col,
                    how="inner" 
                )
            else: # factor_df is empty, use all loaded close prices for the period
                market_close_prices_df = self.memory_bar["close"].clone()
        
        if market_close_prices_df is None or market_close_prices_df.is_empty():
            self._write_log("Aligned market close prices are not available or empty. Aborting analysis.", level=ERROR)
            return None


        # Step 4: Analyse Factor Results
        actual_analysis_start_dt = start_datetime
        actual_analysis_end_dt = end_datetime
        if not factor_df.is_empty():
            try:
                min_dt_val = factor_df.select(pl.col(DEFAULT_DATETIME_COL).min()).item()
                max_dt_val = factor_df.select(pl.col(DEFAULT_DATETIME_COL).max()).item()
                if isinstance(min_dt_val, (datetime, pd.Timestamp)):
                    actual_analysis_start_dt = pd.to_datetime(min_dt_val).to_pydatetime() if isinstance(min_dt_val, pd.Timestamp) else min_dt_val
                if isinstance(max_dt_val, (datetime, pd.Timestamp)):
                    actual_analysis_end_dt = pd.to_datetime(max_dt_val).to_pydatetime() if isinstance(max_dt_val, pd.Timestamp) else max_dt_val
            except Exception as e_dt:
                self._write_log(f"Could not derive precise start/end from factor_df: {e_dt}. Using original period.", WARNING)

        report_path = self._run_factor_analysis(
            factor_df=factor_df,
            market_close_prices_df=market_close_prices_df,
            target_factor_instance=target_factor_instance,
            analysis_start_dt=actual_analysis_start_dt,
            analysis_end_dt=actual_analysis_end_dt,
            num_quantiles=num_quantiles,
            returns_look_ahead_period=returns_look_ahead_period,
            long_percentile_threshold=long_percentile_threshold,
            short_percentile_threshold=short_percentile_threshold,
            report_filename_prefix=report_filename_prefix
        )

        if report_path:
            self._write_log(f"Backtest and analysis complete. Report: {report_path}", level=INFO)
        else:
            self._write_log("Analysis and reporting failed.", level=WARNING)

        return report_path

    def _write_log(self, msg: str, level: int = INFO) -> None:
        log_msg = f"[{self.engine_name}] {msg}"
        level_map = {DEBUG: logger.debug, INFO: logger.info, WARNING: logger.warning, ERROR: logger.error}
        log_func = level_map.get(level, logger.info)
        log_func(log_msg)

