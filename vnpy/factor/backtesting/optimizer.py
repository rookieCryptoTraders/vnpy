# optimizer.py

import copy
import traceback # For detailed error logging
from datetime import datetime
from logging import INFO, DEBUG, WARNING, ERROR
from turtle import pd
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
from pandas import Interval
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import quantstats as qs

from vnpy.factor.backtesting.factor_calculator import FactorCalculator
from vnpy.factor.utils.factor_utils import apply_params_to_definition_dict

DEFAULT_DATETIME_COL = "datetime"
APP_NAME = "OPTIMIZER_FALLBACK"

from vnpy.factor.backtesting.backtesting import BacktestEngine
from vnpy.factor.backtesting.factor_analyzer import FactorAnalyser
from vnpy.factor.template import FactorTemplate # For Sharpe ratio calculation in estimator

from loguru import logger


class FactorBacktestEstimator(BaseEstimator):
    """
    Custom scikit-learn estimator for factor backtesting.
    It allows GridSearchCV to optimize factor parameters by running
    backtests and scoring them (e.g., using Sharpe ratio).
    """
    def __init__(
        self,
        backtest_engine_instance: BacktestEngine,
        base_factor_definition: Union[FactorTemplate, Dict], # Template to be modified
        full_training_memory_bar: Dict[str, pl.DataFrame],   # Full training dataset
        full_training_num_rows: int,
        vt_symbols: List[str],
        data_interval: Interval,
        factor_json_conf_path: Optional[str] = None, # For string-based factor_definition
        # Parameters for scoring logic (L/S portfolio analysis)
        returns_look_ahead_period: int = 1,
        long_percentile_threshold: float = 0.7,
        short_percentile_threshold: float = 0.3,
        # Internal state for params set by GridSearchCV
        **kwargs_params_to_set # Will be populated by set_params
    ):
        self.backtest_engine_instance = backtest_engine_instance
        self.base_factor_definition = base_factor_definition # Store the original template
        self.full_training_memory_bar = full_training_memory_bar
        self.full_training_num_rows = full_training_num_rows
        self.vt_symbols = vt_symbols
        self.data_interval = data_interval # Needed if FactorCalculator requires it, but data is pre-loaded for estimator
        self.factor_json_conf_path = factor_json_conf_path

        # Analysis params for scoring
        self.returns_look_ahead_period = returns_look_ahead_period
        self.long_percentile_threshold = long_percentile_threshold
        self.short_percentile_threshold = short_percentile_threshold
        
        # Store parameters set by GridSearchCV
        self.params_to_set: Dict[str, Any] = kwargs_params_to_set
        
        # This will hold the factor instance with the best params found by this estimator instance
        # (useful if GridSearchCV asks for best_estimator_)
        self._best_factor_instance_for_this_estimator: Optional[FactorTemplate] = None


    def fit(self, X, y=None):
        # Not much to do in fit for this type of estimator used with GridSearchCV,
        # as the main work (parameter setting and scoring) happens in set_params and score.
        # X and y are typically features and target, which we don't use directly here.
        # We use the full_training_memory_bar for CV splits.
        return self

    def score(self, X, y=None) -> float:
        """
        Calculates the score (Sharpe ratio of L/S portfolio) for the current parameter set.
        X here represents the indices of a CV split of the training data.
        """
        # 1. Create a working copy of the base factor definition for this scoring run
        current_trial_factor_definition: Union[FactorTemplate, Dict]
        if isinstance(self.base_factor_definition, FactorTemplate):
            # If FactorTemplate instances are truly cloneable without side effects
            current_trial_factor_definition = copy.deepcopy(self.base_factor_definition)
            # Or, more robustly, re-create from its settings if deepcopy is problematic
            # current_trial_factor_definition = self.base_factor_definition.to_setting()
        elif isinstance(self.base_factor_definition, dict):
            current_trial_factor_definition = copy.deepcopy(self.base_factor_definition)
        else:
            logger.error("Estimator: Invalid base_factor_definition type for score.")
            return -float('inf') # Very bad score

        # 2. Apply current parameters (set by GridSearchCV via set_params)
        if self.params_to_set:
            if isinstance(current_trial_factor_definition, FactorTemplate):
                current_trial_factor_definition.set_nested_params_for_optimizer(self.params_to_set)
            elif isinstance(current_trial_factor_definition, dict):
                current_trial_factor_definition = apply_params_to_definition_dict(
                    current_trial_factor_definition, self.params_to_set
                )
        
        # 3. Initialize and flatten the factor with current trial's parameters
        #    This uses the BacktestEngine's internal logic for consistency.
        target_factor_instance, flattened_factors = \
            self.backtest_engine_instance._init_and_flatten_factor(
                factor_definition=current_trial_factor_definition,
                vt_symbols_for_factor=self.vt_symbols, # Use symbols relevant to the training data
                factor_json_conf_path=self.factor_json_conf_path
            )

        if not target_factor_instance or not flattened_factors:
            logger.warning(f"Estimator: Failed to initialize/flatten factor for params: {self.params_to_set}. Returning very low score.")
            return -float('inf')

        # 4. Get the data for the current CV split
        # X contains indices for the current CV split. Slice full_training_memory_bar.
        # TimeSeriesSplit provides (train_indices_for_split, test_indices_for_split_within_train_data)
        # For scoring, we use the "test" part of this CV split from the training data.
        cv_split_indices = X # X is already the test indices for this CV fold
        
        cv_split_memory_bar: Dict[str, pl.DataFrame] = {}
        min_len_for_cv_split = 0
        if not self.full_training_memory_bar.get("close", pl.DataFrame()).is_empty():
            min_len_for_cv_split = self.full_training_memory_bar["close"].select(pl.col(DEFAULT_DATETIME_COL).slice(cv_split_indices[0], len(cv_split_indices))).height

        if min_len_for_cv_split == 0:
             logger.warning("Estimator: CV split resulted in empty data. Returning very low score.")
             return -float('inf')
             
        for key, df in self.full_training_memory_bar.items():
            if isinstance(df, pl.DataFrame) and not df.is_empty():
                # Slice the DataFrame according to the CV split indices
                # Note: X (cv_split_indices) are absolute indices into full_training_memory_bar
                # Polars slice is (offset, length). We need to find the start index and length of this split.
                if len(cv_split_indices) > 0:
                    start_idx = cv_split_indices[0]
                    length = len(cv_split_indices)
                    cv_split_memory_bar[key] = df.slice(start_idx, length)
                else: # Should not happen if min_len > 0
                    cv_split_memory_bar[key] = df.clear()
            else:
                cv_split_memory_bar[key] = df # Pass non-DataFrame items as is

        cv_split_num_rows = min_len_for_cv_split


        # 5. Calculate factor values on this CV split
        #    FactorCalculator is created fresh or reconfigured for each score evaluation.
        #    It needs the output_data_dir_for_calculator_cache from the main engine.
        calculator = FactorCalculator(
            output_data_dir_for_cache=self.backtest_engine_instance.output_data_dir_for_calculator_cache
        )
        factor_df_cv = calculator.compute_factor_values(
            target_factor_instance_input=target_factor_instance,
            flattened_factors_input=flattened_factors,
            memory_bar_input=cv_split_memory_bar,
            num_data_rows_input=cv_split_num_rows,
            vt_symbols_for_run=self.vt_symbols
        )
        calculator.close()

        if factor_df_cv is None or factor_df_cv.is_empty():
            logger.warning(f"Estimator: Factor calculation on CV split yielded no data for params: {self.params_to_set}. Score: -inf")
            return -float('inf')

        # 6. Perform analysis to get Sharpe Ratio
        #    FactorAnalyser is also created fresh.
        analyser = FactorAnalyser(
             output_data_dir_for_reports=None # No full report needed during CV scoring
        )
        
        # Market close prices for this CV split
        market_close_cv = cv_split_memory_bar.get("close")
        if market_close_cv is None or market_close_cv.is_empty():
            logger.warning("Estimator: Market close data for CV split is missing. Score: -inf")
            analyser.close(); return -float('inf')

        # Use factor_df_cv's datetime column as the reference for returns alignment
        ref_dt_series = factor_df_cv[DEFAULT_DATETIME_COL]

        if not analyser.prepare_symbol_returns(market_close_cv, ref_dt_series):
            logger.warning(f"Estimator: Failed to prepare returns for CV split. Params: {self.params_to_set}. Score: -inf")
            analyser.close(); return -float('inf')
        
        if not analyser.perform_long_short_analysis(
            factor_data_df=factor_df_cv,
            returns_look_ahead_period=self.returns_look_ahead_period,
            long_percentile_threshold=self.long_percentile_threshold,
            short_percentile_threshold=self.short_percentile_threshold
        ):
            logger.warning(f"Estimator: L/S analysis failed for CV split. Params: {self.params_to_set}. Score: -inf")
            analyser.close(); return -float('inf')

        # Calculate Sharpe ratio from L/S portfolio returns
        sharpe = -float('inf') # Default to very bad score
        if analyser.long_short_portfolio_returns_df is not None and not analyser.long_short_portfolio_returns_df.is_empty():
            try:
                # Convert to pandas series for quantstats
                pd_returns = analyser.long_short_portfolio_returns_df.select([
                    pl.col(DEFAULT_DATETIME_COL),
                    pl.col("ls_portfolio_return").fill_null(0.0) # Fill NaNs before passing to qs
                ]).to_pandas().set_index(DEFAULT_DATETIME_COL)["ls_portfolio_return"]
                
                if not pd_returns.empty:
                    # Ensure index is datetime
                    if not isinstance(pd_returns.index, pd.DatetimeIndex):
                        pd_returns.index = pd.to_datetime(pd_returns.index)
                    # Handle potential all-NaN or all-zero series after fill_null if original was all null
                    if pd_returns.count() > 1 and not (pd_returns == 0).all(): # count() excludes NaNs
                        sharpe_value = qs.stats.sharpe(pd_returns, smart=True, annualize=True) # Annualize if appropriate for data freq
                        sharpe = float(sharpe_value) if not np.isnan(sharpe_value) and not np.isinf(sharpe_value) else -float('inf')
                    else:
                        sharpe = 0.0 # Or some other neutral score for zero returns/not enough data
            except Exception as e:
                logger.warning(f"Estimator: Error calculating Sharpe for CV split: {e}. Params: {self.params_to_set}. Score: -inf")
                sharpe = -float('inf')
        
        logger.info(f"Estimator Score (Sharpe): {sharpe:.4f} for params: {self.params_to_set}")
        analyser.close()
        
        # Store the factor instance that produced this score if it's the best so far for this estimator instance
        # Note: GridSearchCV manages the overall best_estimator_. This is for inspection if needed.
        if not hasattr(self, '_current_best_score') or sharpe > self._current_best_score: # type: ignore
            self._current_best_score = sharpe
            self._best_factor_instance_for_this_estimator = target_factor_instance

        return sharpe

    def get_params(self, deep=True):
        # Required by sklearn: return constructor parameters
        return {
            "backtest_engine_instance": self.backtest_engine_instance,
            "base_factor_definition": self.base_factor_definition,
            "full_training_memory_bar": self.full_training_memory_bar,
            "full_training_num_rows": self.full_training_num_rows,
            "vt_symbols": self.vt_symbols,
            "data_interval": self.data_interval,
            "factor_json_conf_path": self.factor_json_conf_path,
            "returns_look_ahead_period": self.returns_look_ahead_period,
            "long_percentile_threshold": self.long_percentile_threshold,
            "short_percentile_threshold": self.short_percentile_threshold,
            **self.params_to_set # Include parameters set by GridSearchCV
        }

    def set_params(self, **params):
        # Called by GridSearchCV to set parameters for the current trial
        # We store them in self.params_to_set so `score` can access them.
        # Filter out estimator's own init params from those to be set on the factor
        own_init_params = [
            "backtest_engine_instance", "base_factor_definition", 
            "full_training_memory_bar", "full_training_num_rows",
            "vt_symbols", "data_interval", "factor_json_conf_path",
            "returns_look_ahead_period", "long_percentile_threshold", "short_percentile_threshold"
        ]
        self.params_to_set = {k: v for k, v in params.items() if k not in own_init_params}
        
        # Also update the estimator's own attributes if they are in params (e.g., analysis params)
        for param_name, value in params.items():
            if hasattr(self, param_name) and param_name in own_init_params:
                setattr(self, param_name, value)
        return self

    def get_best_factor_definition_after_fit(self) -> Optional[Union[FactorTemplate, Dict]]:
        """
        Helper to retrieve the factor definition that achieved the best score
        WITHIN THIS SPECIFIC ESTIMATOR INSTANCE (which GridSearchCV makes copies of).
        GridSearchCV's `best_estimator_` attribute will hold the estimator instance
        that found the globally best parameters.
        """
        if self._best_factor_instance_for_this_estimator:
            # Return a serializable form or the instance itself
            return self._best_factor_instance_for_this_estimator # Or .to_setting()
        return None


class FactorOptimizer:
    """
    Optimizes factor parameters using GridSearchCV and FactorBacktestEstimator.
    """
    engine_name = APP_NAME + "_FactorOptimizer"

    def __init__(self, backtest_engine: BacktestEngine): # Requires a BacktestEngine instance
        self.backtest_engine = backtest_engine # The orchestrator
        self._write_log(f"{self.engine_name} initialized.", level=INFO)

    @staticmethod
    def _split_data_dict(
        data_dict: Dict[str, pl.DataFrame],
        test_size_ratio: float,
        dt_col: str 
    ) -> Tuple[Dict[str, pl.DataFrame], Dict[str, pl.DataFrame], int, int]:
        """
        Splits data dictionary. Returns train_dict, test_dict, train_rows, test_rows.
        (Simplified version for optimizer context)
        """
        if not (0.0 <= test_size_ratio < 1.0):
            raise ValueError("test_size_ratio must be between 0.0 and 1.0 (exclusive).")
        if "close" not in data_dict or data_dict["close"].is_empty():
            logger.warning("Optimizer split: 'close' data missing or empty.")
            return {}, {}, 0, 0

        n_total = data_dict["close"].height
        n_test = int(n_total * test_size_ratio)
        n_train = n_total - n_test
        
        train_d, test_d = {}, {}
        for k, df in data_dict.items():
            if isinstance(df, pl.DataFrame) and df.height == n_total:
                train_d[k] = df.slice(0, n_train)
                test_d[k] = df.slice(n_train, n_test) if n_test > 0 else df.clear()
            else: # Non-DataFrame or mismatched rows
                train_d[k] = df 
                test_d[k] = df.clear() if isinstance(df, pl.DataFrame) else (type(df)() if callable(type(df)) else None)
        return train_d, test_d, n_train, n_test


    def optimize_factor(
        self,
        factor_definition_template: Union[FactorTemplate, Dict], # Base definition to optimize
        parameter_grid: Dict[str, List[Any]], # Optimizer grid
        start_datetime: datetime,
        end_datetime: datetime,
        vt_symbols: List[str],
        data_interval: Interval,
        factor_json_conf_path: Optional[str] = None, # If definition_template is a key string (less ideal for optimizer)
        test_size_ratio: float = 0.2,
        n_cv_splits: int = 3,
        # Analysis parameters for scoring objective (L/S Sharpe) & final report
        returns_look_ahead_period: int = 1,
        long_percentile_threshold: float = 0.7,
        short_percentile_threshold: float = 0.3,
        report_filename_prefix: str = "optimized_factor_report"
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[Path]]:
        """
        Performs grid search optimization for the given factor definition.

        Returns:
            A tuple: (best_params, grid_search_cv_results, path_to_final_report_on_test_set)
        """
        self._write_log(f"Starting optimization for factor. Symbols: {vt_symbols}", level=INFO)

        # 1. Load full data ONCE using the BacktestEngine's loader
        if not self.backtest_engine._load_bar_data_engine(start_datetime, end_datetime, data_interval, vt_symbols):
            self._write_log("Optimizer: Failed to load full bar data. Aborting optimization.", ERROR)
            return None, None, None
        if self.backtest_engine.num_data_rows == 0:
            self._write_log("Optimizer: No bar data rows loaded. Aborting optimization.", WARNING)
            return None, None, None
        
        full_memory_bar = copy.deepcopy(self.backtest_engine.memory_bar) # Use a copy
        full_num_rows = self.backtest_engine.num_data_rows

        # 2. Split data into training and testing sets
        train_memory_bar, test_memory_bar, train_num_rows, test_num_rows = \
            FactorOptimizer._split_data_dict(full_memory_bar, test_size_ratio, DEFAULT_DATETIME_COL)

        if train_num_rows == 0:
            self._write_log("Optimizer: Training data is empty after split. Cannot proceed with optimization.", ERROR)
            return None, None, None
        
        self._write_log(f"Data split: Train rows={train_num_rows}, Test rows={test_num_rows}", INFO)

        # 3. Setup FactorBacktestEstimator
        estimator = FactorBacktestEstimator(
            backtest_engine_instance=self.backtest_engine, # Pass the orchestrator
            base_factor_definition=factor_definition_template,
            full_training_memory_bar=train_memory_bar, # Estimator works with training data
            full_training_num_rows=train_num_rows,
            vt_symbols=vt_symbols, # Symbols relevant for this training data
            data_interval=data_interval,
            factor_json_conf_path=factor_json_conf_path,
            returns_look_ahead_period=returns_look_ahead_period,
            long_percentile_threshold=long_percentile_threshold,
            short_percentile_threshold=short_percentile_threshold
        )

        # 4. Setup GridSearchCV
        if train_num_rows < n_cv_splits + 1 and n_cv_splits > 0: # TimeSeriesSplit needs enough samples
            self._write_log(f"Optimizer: Not enough samples ({train_num_rows}) in training data for {n_cv_splits} CV splits. Reducing CV splits or increasing data.", WARNING)
            # Adjust n_cv_splits if possible, or raise error if too few for any reasonable CV
            n_cv_splits = max(1, train_num_rows -1) if train_num_rows > 1 else 0 # Or handle as error
            if n_cv_splits == 0:
                 self._write_log("Optimizer: Cannot perform CV with current training data size. Aborting.", ERROR)
                 return None, None, None

        cv_splitter = TimeSeriesSplit(n_splits=n_cv_splits) if n_cv_splits > 0 else None # Handle no CV case if desired

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring=None, # Relies on estimator's score method
            cv=cv_splitter,
            verbose=2,
            n_jobs=1 # Crucial for safety with shared BacktestEngine state if not designed for parallel estimator runs
        )
        
        self._write_log(f"Optimizer: Starting GridSearchCV.fit on training data ({train_num_rows} samples, {n_cv_splits} CV splits)...", INFO)
        try:
            # X for GridSearchCV is typically features. Estimator uses pre-loaded training data.
            # Pass indices for TimeSeriesSplit to work correctly.
            training_indices = np.arange(train_num_rows)
            grid_search.fit(X=training_indices, y=None)
        except Exception as e_grid:
            self._write_log(f"Optimizer: Error during GridSearchCV.fit: {e_grid}", ERROR)
            self._write_log(traceback.format_exc(), DEBUG)
            return None, None, None

        best_params = grid_search.best_params_
        cv_results = grid_search.cv_results_
        self._write_log(f"Optimizer: GridSearchCV completed. Best Score (Sharpe): {grid_search.best_score_:.4f}", INFO)
        self._write_log(f"Optimizer: Best Parameters found: {best_params}", INFO)

        # 5. Final Evaluation on Test Set using best parameters
        final_report_path: Optional[Path] = None
        if test_num_rows > 0:
            self._write_log(f"Optimizer: Evaluating best parameters on test set ({test_num_rows} rows)...", INFO)
            
            # Create the factor definition with the best parameters
            final_factor_definition: Union[FactorTemplate, Dict]
            if isinstance(factor_definition_template, FactorTemplate):
                # It's often better to create a new instance from settings with best_params
                # or ensure deepcopy and set_nested_params is robust.
                # For simplicity, assuming set_nested_params works on a copy or original template.
                final_factor_definition = copy.deepcopy(factor_definition_template)
                final_factor_definition.set_nested_params_for_optimizer(best_params)
            elif isinstance(factor_definition_template, dict):
                final_factor_definition = apply_params_to_definition_dict(
                    copy.deepcopy(factor_definition_template), best_params
                )
            else: # Should not happen if initial checks are done
                self._write_log("Optimizer: Invalid factor_definition_template type for final evaluation.", ERROR)
                return best_params, cv_results, None

            # Temporarily set BacktestEngine's data to the test set for the final run
            original_engine_memory_bar = self.backtest_engine.memory_bar
            original_engine_num_rows = self.backtest_engine.num_data_rows
            
            self.backtest_engine.memory_bar = test_memory_bar
            self.backtest_engine.num_data_rows = test_num_rows
            
            # Determine test set start/end datetimes for the report
            test_start_dt = test_memory_bar["close"][DEFAULT_DATETIME_COL].min() if test_num_rows > 0 else end_datetime
            test_end_dt = test_memory_bar["close"][DEFAULT_DATETIME_COL].max() if test_num_rows > 0 else end_datetime


            # Run a single backtest with the best parameters on the test data
            # The run_single_factor_backtest will use the (now test) data in self.backtest_engine.memory_bar
            final_report_path = self.backtest_engine.run_single_factor_backtest(
                factor_definition=final_factor_definition,
                start_datetime=test_start_dt, # Reporting purpose, actual data is test_memory_bar
                end_datetime=test_end_dt,     # Reporting purpose
                vt_symbols_for_factor=vt_symbols, # Symbols are consistent
                factor_json_conf_path=factor_json_conf_path, # If original def was key
                data_interval=data_interval, # Consistent interval
                num_quantiles=estimator.get_params()["num_quantiles"] if hasattr(estimator, "num_quantiles") else 5, # Use analysis params from estimator
                returns_look_ahead_period=returns_look_ahead_period,
                long_percentile_threshold=long_percentile_threshold,
                short_percentile_threshold=short_percentile_threshold,
                report_filename_prefix=report_filename_prefix + "_TEST_SET"
            )
            
            # Restore BacktestEngine's original data state if it was modified
            self.backtest_engine.memory_bar = original_engine_memory_bar
            self.backtest_engine.num_data_rows = original_engine_num_rows

            if final_report_path:
                self._write_log(f"Optimizer: Final evaluation on test set complete. Report: {final_report_path}", INFO)
            else:
                self._write_log("Optimizer: Final evaluation on test set failed to produce a report.", WARNING)
        else:
            self._write_log("Optimizer: No test data to evaluate. Skipping final test set evaluation.", INFO)
            # Optionally, could run on full training data again if no test set
            # final_report_path = self.backtest_engine.run_single_factor_backtest(...) on training data

        return best_params, cv_results, final_report_path

    def _write_log(self, msg: str, level: int = INFO) -> None:
        log_msg = f"[{self.engine_name}] {msg}"
        level_map = {DEBUG: logger.debug, INFO: logger.info, WARNING: logger.warning, ERROR: logger.error}
        log_func = level_map.get(level, logger.info)
        log_func(log_msg)
