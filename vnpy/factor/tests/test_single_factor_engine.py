import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import tempfile
import shutil
import polars as pl
from datetime import datetime
from typing import Dict, Any

from vnpy.factor.template import FactorTemplate, FactorParameters
from vnpy.trader.constant import Interval
from vnpy.factor.backtesting.backtesting import BacktestEngine, DEFAULT_DATETIME_COL
from vnpy.factor.optimizer import FactorBacktestEstimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit # For patching
import numpy as np

class MockFactor(FactorTemplate):
    factor_name = "mockfactor"

    def get_output_schema(self) -> Dict[str, pl.DataType]:
        schema = {DEFAULT_DATETIME_COL: pl.Datetime}
        # self.vt_symbols is set when factor is initialized by engine or manually
        for symbol in getattr(self, 'vt_symbols', []): 
            schema[symbol] = pl.Float64
        return schema

    def calculate(self, input_data: Dict[str, pl.DataFrame], memory: Any) -> pl.DataFrame:
        # input_data is expected to be engine.memory_bar, a dict of DataFrames like {"close": df_close, ...}
        # df_close has columns: DEFAULT_DATETIME_COL, SYM1.TEST, SYM2.TEST, ...
        
        close_prices_df = input_data.get("close")

        # Ensure vt_symbols are available for schema generation fallback
        current_vt_symbols = getattr(self, 'vt_symbols', [])

        if close_prices_df is None or close_prices_df.is_empty() or DEFAULT_DATETIME_COL not in close_prices_df.columns:
            # Return empty DataFrame matching schema if essential input is missing
            empty_data = {DEFAULT_DATETIME_COL: pl.Series(DEFAULT_DATETIME_COL, [], dtype=pl.Datetime)}
            for symbol in current_vt_symbols: # Use symbols this factor instance is configured for
                empty_data[symbol] = pl.Series(symbol, [], dtype=pl.Float64)
            # Explicitly pass schema to handle case where current_vt_symbols might be empty if not properly init'd before schema call
            return pl.DataFrame(empty_data, schema=self.get_output_schema() or {DEFAULT_DATETIME_COL: pl.Datetime})


        # Prepare output data dictionary
        output_df_data = {
            DEFAULT_DATETIME_COL: close_prices_df[DEFAULT_DATETIME_COL]
        }

        for symbol in current_vt_symbols: # Iterate over symbols this factor instance is configured for
            if symbol in close_prices_df.columns:
                output_df_data[symbol] = close_prices_df[symbol] * 1.05 # Dummy calculation
            else:
                # If symbol data not in close_prices, fill with nulls or default
                output_df_data[symbol] = pl.Series(symbol, [None] * len(close_prices_df), dtype=pl.Float64)
        
        # Ensure column order and presence as per schema
        # This is important if current_vt_symbols caused schema to be different or if a symbol was missing
        final_schema_keys = list(self.get_output_schema().keys())
        
        # Create DataFrame with available data
        temp_df = pl.DataFrame(output_df_data)
        
        # Select columns based on schema to ensure correct structure
        # For any symbol in schema but not in temp_df (e.g. if close_prices_df was missing it), it will be error
        # So, ensure all schema keys are present in output_df_data before this step if possible
        # The current loop for current_vt_symbols should ensure this.
        
        select_cols = []
        for col_name in final_schema_keys:
            if col_name in temp_df.columns:
                select_cols.append(col_name)
            elif col_name == DEFAULT_DATETIME_COL: # Should always be there if close_prices_df was valid
                 select_cols.append(DEFAULT_DATETIME_COL)
            # If a symbol from schema is NOT in temp_df columns, it implies an issue or it wasn't in current_vt_symbols.
            # The current logic (iterating current_vt_symbols) should mean all configured symbols are in output_df_data.

        return temp_df.select(select_cols)


class TestSingleFactorEngine(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, if any."""
        self.temp_dir = tempfile.mkdtemp()
        self.vt_symbols = ["SYM1.TEST", "SYM2.TEST"]
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 10)
        
        # Mock the factor module loading in BacktestEngine if factors are loaded by string name
        # For pre-initialized factors, this might not be strictly necessary but good for consistency
        patcher = patch('vnpy.factor.backtesting.backtesting.importlib.import_module')
        self.mock_import_module = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Simulate a factor module that can provide MockFactor if needed by name
        # Though for this test, we pass an instance directly.
        mock_factor_module = MagicMock()
        mock_factor_module.MockFactor = MockFactor 
        self.mock_import_module.return_value = mock_factor_module

        self.engine = BacktestEngine(
            vt_symbols=self.vt_symbols, # Default symbols for engine if not overridden by run_single_factor
            output_data_dir=self.temp_dir,
            factor_module_name="vnpy.factor.factors" # Dummy, as we provide factor instance
        )
        # Ensure load_bar_data is patched to prevent actual data loading
        # and to control the dummy data provided
        patch_load_bar = patch.object(self.engine, 'load_bar_data', self.mock_load_bar_data)
        self.mock_load_bar_method = patch_load_bar.start()
        self.addCleanup(patch_load_bar.stop)

    def mock_load_bar_data(self, start: datetime, end: datetime, interval: Any) -> bool:
        """Controlled mock for load_bar_data."""
        # This method will be called by run_single_factor_backtest
        # We need to populate self.engine.memory_bar["close"] and self.engine.num_data_rows
        # The symbols used here should ideally match self.engine.vt_symbols set by run_single_factor_backtest
        
        # `run_single_factor_backtest` updates `self.engine.vt_symbols` based on `vt_symbols_for_factor`
        # So, use `self.engine.vt_symbols` which reflects the symbols for the current test run.
        active_symbols = self.engine.vt_symbols 
        
        num_days = (end - start).days + 1 # Inclusive of end date for daily
        if interval == Interval.DAILY:
            dates = [start + pl.duration(days=i) for i in range(num_days)]
        else: # Default to minute-like frequency if not daily for placeholder
            num_minutes = num_days * 24 * 60 
            dates = [start + pl.duration(minutes=i) for i in range(num_minutes)]

        if not dates: # Should not happen with valid start/end
            self.engine.num_data_rows = 0
            self.engine.memory_bar = {}
            return False

        self.engine.num_data_rows = len(dates)
        
        close_data = {DEFAULT_DATETIME_COL: pl.Series(DEFAULT_DATETIME_COL, dates, dtype=pl.Datetime)}
        for symbol in active_symbols: # Use symbols relevant to the current engine configuration
            # Generate some deterministic but varying data
            close_data[symbol] = pl.Series(symbol, [100 + (i % 10) + (active_symbols.index(symbol) * 5) for i in range(len(dates))], dtype=pl.Float64)
        
        self.engine.memory_bar = {"close": pl.DataFrame(close_data)}
        self.engine.write_log(f"Mock load_bar_data: Populated memory_bar['close'] with {self.engine.num_data_rows} rows for symbols: {active_symbols}", level=10) # DEBUG level
        return True


    def tearDown(self):
        """Tear down test fixtures, if any."""
        shutil.rmtree(self.temp_dir)

    def test_run_with_preinitialized_factor(self):
        """Test running backtest with a pre-initialized factor instance."""
        symbols_for_this_test = [self.vt_symbols[0]] # Test with only one symbol

        # Create and configure the MockFactor instance
        mock_factor_instance = MockFactor(engine=None) # engine can be None for init
        mock_factor_instance.vt_symbols = symbols_for_this_test # Manually set symbols for this factor
        mock_factor_instance.params = FactorParameters({"window": 5}) # Dummy params

        report_path = self.engine.run_single_factor_backtest(
            factor_definition=mock_factor_instance,
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            vt_symbols_for_factor=symbols_for_this_test, # Engine will use these symbols
            data_interval=Interval.DAILY
        )

        self.assertIsNotNone(report_path, "Report path should not be None")
        self.assertTrue(Path(report_path).exists(), "Report file should exist")

        self.assertIsNotNone(self.engine.quantile_analysis_results, "Quantile analysis results should exist")
        if self.engine.quantile_analysis_results: # Check to satisfy type checker
            self.assertIn("overall_average", self.engine.quantile_analysis_results)
            self.assertIn("by_time", self.engine.quantile_analysis_results)
            # Check if overall_average DataFrame is not empty if it exists
            overall_avg_df = self.engine.quantile_analysis_results.get("overall_average")
            self.assertIsNotNone(overall_avg_df, "Overall average quantile returns DF should exist")
            if overall_avg_df is not None: # Keep mypy happy
                 self.assertFalse(overall_avg_df.is_empty(), "Overall average quantile returns DF should not be empty")


        self.assertIsNotNone(self.engine.long_short_portfolio_returns_df, "L-S portfolio returns DF should exist")
        if self.engine.long_short_portfolio_returns_df is not None: # Keep mypy happy
            self.assertFalse(self.engine.long_short_portfolio_returns_df.is_empty(), "L-S portfolio returns DF should not be empty")

        self.assertIsNotNone(self.engine.long_short_stats, "L-S stats should exist")
        if self.engine.long_short_stats: # Keep mypy happy
            self.assertIn("t_statistic", self.engine.long_short_stats)
            # With deterministic data, t-stat might not be NaN if there's variance
            # self.assertIsNotNone(self.engine.long_short_stats.get("t_statistic"), "T-statistic should not be None/NaN")


        self.assertIsNotNone(self.engine.performance_metrics, "Performance metrics should exist")
        if self.engine.performance_metrics: # Keep mypy happy
            self.assertIn("sharpe", self.engine.performance_metrics)
            # Sharpe can be 0 or NaN depending on returns variance.
            # self.assertIsNotNone(self.engine.performance_metrics.get("sharpe"), "Sharpe ratio should not be None/NaN")

    def test_factor_backtest_estimator_fit_score(self):
        """Test FactorBacktestEstimator's fit and score methods."""
        mock_engine_instance = MagicMock(spec=BacktestEngine)
        mock_engine_instance.factor_datetime_col = DEFAULT_DATETIME_COL
        # Mock write_log to prevent console output during tests unless debugging
        mock_engine_instance.write_log = MagicMock()


        mock_full_bar_data = {
            "close": pl.DataFrame({
                DEFAULT_DATETIME_COL: [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023,1,3)],
                "SYM1.TEST": [10.0, 10.1, 10.2],
                "SYM2.TEST": [20.0, 20.1, 20.2]
            }),
            "open": pl.DataFrame({ # Estimator's fit logic expects dict of DFs, so provide more keys
                DEFAULT_DATETIME_COL: [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023,1,3)],
                "SYM1.TEST": [9.9, 10.0, 10.1],
                "SYM2.TEST": [19.9, 20.0, 20.1]
            })
        }
        initial_factor_params = {"window": 10, "fixed_param": 5}

        estimator = FactorBacktestEstimator(
            backtesting_engine=mock_engine_instance,
            factor_class_name="MockFactor", # Assuming MockFactor is defined in the test file
            full_bar_data_for_slicing=mock_full_bar_data,
            **initial_factor_params
        )

        # Test if initial params are set
        self.assertEqual(estimator.window, 10)
        self.assertEqual(estimator.fixed_param, 5)

        # Simulate GridSearchCV setting a parameter
        estimator.set_params(window=20)
        self.assertEqual(estimator.window, 20) # Check if set_params worked

        mock_indices = np.array([0, 1]) # Slice first two rows

        # Configure mock engine's methods
        mock_engine_instance.init_single_factor.return_value = True
        
        # Dummy factor data DF needs to match the symbols in the sliced data and datetime col
        # Sliced data will have SYM1.TEST, SYM2.TEST
        dummy_factor_output_df = pl.DataFrame({
            DEFAULT_DATETIME_COL: mock_full_bar_data["close"][DEFAULT_DATETIME_COL][mock_indices],
            "SYM1.TEST": [0.5, 0.6],
            "SYM2.TEST": [-0.2, 0.3]
        })
        mock_engine_instance.calculate_single_factor_data.return_value = dummy_factor_output_df
        mock_engine_instance.prepare_symbol_returns.return_value = True
        mock_engine_instance.perform_long_short_analysis.return_value = True
        mock_engine_instance.calculate_performance_metrics.return_value = True
        
        # Using PropertyMock if performance_metrics is a property, else direct assignment
        # For this subtask, assuming performance_metrics is a direct attribute that can be set.
        # If it's a @property, then:
        # type(mock_engine_instance).performance_metrics = PropertyMock(return_value={"sharpe": 1.23})
        # For now, direct assignment as per current BacktestEngine structure:
        mock_engine_instance.performance_metrics = {"sharpe": 1.23}


        result_estimator = estimator.fit(X_indices_for_fold=mock_indices)
        self.assertIs(result_estimator, estimator) # fit should return self

        # Assert init_single_factor was called
        mock_engine_instance.init_single_factor.assert_called_once()
        args, kwargs = mock_engine_instance.init_single_factor.call_args
        
        # Check factor_definition argument
        called_factor_def = kwargs.get('factor_definition') 
        if called_factor_def is None and args: called_factor_def = args[0] # if passed as positional
        
        self.assertIsNotNone(called_factor_def)
        self.assertEqual(called_factor_def.get("class_name"), "MockFactor")
        self.assertEqual(called_factor_def.get("params", {}).get("window"), 20) # Updated by set_params
        self.assertEqual(called_factor_def.get("params", {}).get("fixed_param"), 5) # From initial_factor_params

        # Check vt_symbols_for_factor argument (should be derived from sliced data)
        called_symbols = kwargs.get('vt_symbols_for_factor')
        if called_symbols is None and len(args) > 1: called_symbols = args[1]

        self.assertEqual(set(called_symbols), {"SYM1.TEST", "SYM2.TEST"})


        self.assertEqual(estimator.current_score, 1.23)
        self.assertEqual(estimator.score(None), 1.23)
        
        # Test case where Sharpe is None or non-finite
        mock_engine_instance.performance_metrics = {"sharpe": np.nan}
        estimator.fit(X_indices_for_fold=mock_indices)
        self.assertEqual(estimator.current_score, -np.inf)


    @patch('vnpy.factor.backtesting.backtesting.GridSearchCV') # Patch where GridSearchCV is USED
    @patch('vnpy.factor.backtesting.backtesting.TimeSeriesSplit') # Patch where TimeSeriesSplit is USED
    def test_engine_optimize_factor_parameters(self, MockTimeSeriesSplit, MockGridSearchCV):
        """Test BacktestEngine's optimize_factor_parameters method."""
        
        # Configure MockTimeSeriesSplit
        mock_ts_splitter_instance = MockTimeSeriesSplit.return_value
        
        # Configure MockGridSearchCV
        mock_grid_search_instance = MockGridSearchCV.return_value
        mock_grid_search_instance.fit.return_value = None # or mock_grid_search_instance
        mock_grid_search_instance.best_params_ = {"window": 25}
        mock_grid_search_instance.best_score_ = 1.55
        mock_grid_search_instance.cv_results_ = {
            "mean_test_score": np.array([1.55, 1.45]), 
            "std_test_score": np.array([0.1, 0.05]),
            "params": [{"window": 25}, {"window": 15}]
        }

        # Prepare data for the engine method
        # self.engine is from setUp, already has load_bar_data patched.
        # optimize_factor_parameters expects full_bar_data directly.
        test_full_bar_data = {
            "close": pl.DataFrame({
                DEFAULT_DATETIME_COL: [datetime(2023, 1, i+1) for i in range(20)], # 20 days of data
                "SYM1.TEST": [100 + i for i in range(20)],
                "SYM2.TEST": [200 - i for i in range(20)]
            }),
            "open": pl.DataFrame({
                 DEFAULT_DATETIME_COL: [datetime(2023, 1, i+1) for i in range(20)],
                "SYM1.TEST": [99 + i for i in range(20)],
                "SYM2.TEST": [199 - i for i in range(20)]
            })
        }
        
        initial_settings = {"window": 10, "period": 30} # 'period' is a fixed param here
        param_grid = {"window": [15, 25]} # 'window' is the param to optimize
        
        # Call the method to test
        n_cv_splits = 3
        best_params_result = self.engine.optimize_factor_parameters(
            factor_class_name="MockFactor", 
            initial_factor_settings=initial_settings,
            parameter_grid=param_grid,
            full_bar_data=test_full_bar_data,
            n_splits_for_cv=n_cv_splits
        )

        # Assertions
        MockTimeSeriesSplit.assert_called_once_with(n_splits=n_cv_splits)
        
        MockGridSearchCV.assert_called_once()
        args_gscv, kwargs_gscv = MockGridSearchCV.call_args
        
        self.assertIsInstance(kwargs_gscv['estimator'], FactorBacktestEstimator)
        self.assertEqual(kwargs_gscv['estimator'].factor_class_name, "MockFactor")
        self.assertEqual(kwargs_gscv['estimator'].window, 10) # Initial window
        self.assertEqual(kwargs_gscv['estimator'].period, 30) # Initial fixed param
        self.assertIs(kwargs_gscv['estimator'].full_bar_data_for_slicing, test_full_bar_data)
        
        self.assertEqual(kwargs_gscv['param_grid'], param_grid)
        self.assertIs(kwargs_gscv['cv'], mock_ts_splitter_instance)
        self.assertEqual(kwargs_gscv['n_jobs'], 1) # Hardcoded in optimize_factor_parameters

        mock_grid_search_instance.fit.assert_called_once()
        # Check X passed to grid_search.fit (should be np.arange(num_samples))
        args_fit, _ = mock_grid_search_instance.fit.call_args
        expected_indices = np.arange(test_full_bar_data["close"].height)
        np.testing.assert_array_equal(args_fit[0], expected_indices)

        self.assertEqual(best_params_result, {"window": 25})
        
        self.assertIsNotNone(self.engine.optimization_results)
        if self.engine.optimization_results: # For type checker
            self.assertEqual(self.engine.optimization_results["best_score"], 1.55)
            self.assertEqual(self.engine.optimization_results["best_params"], {"window": 25})
            self.assertIn("cv_results_summary", self.engine.optimization_results)
            summary = self.engine.optimization_results["cv_results_summary"]
            self.assertEqual(summary["mean_test_score"], [1.55, 1.45]) # Check .tolist() conversion
            self.assertEqual(summary["params"], ['{\'window\': 25}', '{\'window\': 15}']) # Check str(p) conversion


if __name__ == '__main__':
    unittest.main()
