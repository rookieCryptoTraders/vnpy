# File: vnpy/factor/optimizer.py

from __future__ import annotations # For forward reference of BacktestEngine if needed

import numpy as np
from sklearn.base import BaseEstimator
from typing import Dict, Any, Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    # This avoids circular import issues if BacktestEngine might import this file later,
    # or for cleaner type hinting.
    from vnpy.factor.backtesting.backtesting import BacktestEngine

from logging import INFO, DEBUG, WARNING, ERROR # For log levels
import polars as pl


class FactorBacktestEstimator(BaseEstimator):
    """
    A scikit-learn compatible custom estimator for factor parameter optimization
    using GridSearchCV within the VnPy factor backtesting engine.
    """

    def __init__(
        self,
        backtesting_engine: BacktestEngine,
        factor_class_name: str,
        full_bar_data_for_slicing: Dict[str, pl.DataFrame], # New parameter
        **factor_params: Any
    ):
        self.backtesting_engine = backtesting_engine
        self.factor_class_name = factor_class_name
        self.full_bar_data_for_slicing = full_bar_data_for_slicing # Store this
        
        self._factor_param_names: List[str] = []
        for key, value in factor_params.items():
            setattr(self, key, value)
            self._factor_param_names.append(key)
        
        self.current_score: float = -np.inf

    # get_params and set_params will be inherited from BaseEstimator.
    # BaseEstimator's get_params collects all __init__ parameters.
    # BaseEstimator's set_params updates attributes directly.

    def fit(self, X_indices_for_fold: np.ndarray, y: Any = None) -> FactorBacktestEstimator:
        self.current_score = -np.inf

        current_factor_params_for_trial = {
            key: getattr(self, key) for key in self._factor_param_names
        }
        
        self.backtesting_engine.write_log(f"Estimator.fit: Class '{self.factor_class_name}', Params: {current_factor_params_for_trial}", DEBUG)

        # Helper to slice the dict of DataFrames
        def slice_bar_data_dict(bar_data_dict: Dict[str, pl.DataFrame], slice_indices: np.ndarray) -> Dict[str, pl.DataFrame]:
            sliced_dict = {}
            if not bar_data_dict or not isinstance(slice_indices, np.ndarray) or slice_indices.size == 0:
                self.backtesting_engine.write_log("slice_bar_data_dict: Invalid input or empty indices.", WARNING)
                return sliced_dict # Return empty if inputs are problematic

            for key, df in bar_data_dict.items():
                if isinstance(df, pl.DataFrame) and not df.is_empty():
                    if slice_indices.max() < df.height:
                        sliced_dict[key] = df[slice_indices]
                    else:
                        self.backtesting_engine.write_log(f"slice_bar_data_dict: Indices out of bounds for key {key}. Max index: {slice_indices.max()}, DF height: {df.height}", WARNING)
                        # Return empty dict if any slice fails, to signal data issue for this fold.
                        return {} 
                else:
                    # Pass through non-DF or empty DF as is, or handle more strictly if needed
                    # For now, passing through allows flexibility if some dict items aren't main bar data
                    sliced_dict[key] = df 
            return sliced_dict

        current_data_slice_dict = slice_bar_data_dict(self.full_bar_data_for_slicing, X_indices_for_fold)

        if not current_data_slice_dict or "close" not in current_data_slice_dict or current_data_slice_dict["close"].is_empty():
            self.backtesting_engine.write_log("Estimator.fit: Data slice is invalid or 'close' data is missing/empty after slicing.", ERROR)
            return self

        # Configure BacktestEngine for Data Slice (current_data_slice_dict)
        dt_col_name = self.backtesting_engine.factor_datetime_col
        if dt_col_name not in current_data_slice_dict["close"].columns:
            self.backtesting_engine.write_log(f"Estimator.fit: Datetime col '{dt_col_name}' not in sliced X['close'].", ERROR)
            return self
            
        current_vt_symbols = [col for col in current_data_slice_dict["close"].columns if col != dt_col_name]
        if not current_vt_symbols:
            self.backtesting_engine.write_log("Estimator.fit: No symbol columns in sliced X['close'].", ERROR)
            return self

        self.backtesting_engine.memory_bar = {k: v.clone() for k, v in current_data_slice_dict.items() if isinstance(v, pl.DataFrame)}
        self.backtesting_engine.num_data_rows = current_data_slice_dict["close"].height
        
        # Clear previous run's factor-specific data
        self.backtesting_engine.stacked_factors.clear()
        self.backtesting_engine.flattened_factors.clear()
        self.backtesting_engine.sorted_factor_keys.clear()
        self.backtesting_engine.dask_tasks.clear()
        self.backtesting_engine.factor_memory_instances.clear()

        # Prepare Factor Definition
        factor_definition_dict = {
            "class_name": self.factor_class_name,
            "factor_name": self.factor_class_name, 
            "params": current_factor_params_for_trial,
        }

        # Execute Backtesting Pipeline
        if not self.backtesting_engine.init_single_factor(factor_definition_dict, vt_symbols_for_factor=current_vt_symbols):
            self.backtesting_engine.write_log("Estimator.fit: init_single_factor failed.", WARNING); return self
        factor_data_df = self.backtesting_engine.calculate_single_factor_data()
        if factor_data_df is None or factor_data_df.is_empty():
            self.backtesting_engine.write_log("Estimator.fit: Factor calculation failed/empty.", WARNING); return self
        if not self.backtesting_engine.prepare_symbol_returns(reference_datetime_series=current_data_slice_dict["close"][dt_col_name]):
            self.backtesting_engine.write_log("Estimator.fit: Preparing symbol returns failed.", WARNING); return self
        if not self.backtesting_engine.perform_long_short_analysis(factor_data_df):
            self.backtesting_engine.write_log("Estimator.fit: L-S analysis failed.", WARNING); return self
        if hasattr(self.backtesting_engine, 'long_short_portfolio_returns_df') and \
           self.backtesting_engine.long_short_portfolio_returns_df is not None and \
           not self.backtesting_engine.long_short_portfolio_returns_df.is_empty():
            if not self.backtesting_engine.calculate_performance_metrics():
                self.backtesting_engine.write_log("Estimator.fit: Metrics calculation failed.", WARNING); return self
        else:
            self.backtesting_engine.write_log("Estimator.fit: L-S returns unavailable, skipping metrics.", INFO); return self

        # Store Score
        if hasattr(self.backtesting_engine, 'performance_metrics') and self.backtesting_engine.performance_metrics:
            sharpe_ratio = self.backtesting_engine.performance_metrics.get('sharpe')
            if sharpe_ratio is not None and np.isfinite(sharpe_ratio):
                self.current_score = sharpe_ratio
                self.backtesting_engine.write_log(f"Estimator.fit successful. Sharpe: {self.current_score:.4f}", DEBUG)
            else:
                self.current_score = -np.inf
                self.backtesting_engine.write_log(f"Estimator.fit: Sharpe None/non-finite ({sharpe_ratio}). Score -inf.", DEBUG)
        else:
            self.current_score = -np.inf
            self.backtesting_engine.write_log("Estimator.fit: No metrics. Score -inf.", DEBUG)
        
        return self

    def score(self, X: Any, y: Any = None) -> float:
        """
        Returns the score (e.g., Sharpe Ratio) calculated by the last `fit` call.
        This method will be called by GridSearchCV after `fit`.

        Args:
            X: Data slice (not directly used here as score is based on last fit).
            y: Target variable (not used).
            
        Returns:
            The calculated score (e.g., Sharpe Ratio).
        """
        return self.current_score
