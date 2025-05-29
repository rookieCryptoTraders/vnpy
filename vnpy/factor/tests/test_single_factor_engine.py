import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import polars as pl
from datetime import datetime
from typing import Dict, Any

from vnpy.factor.template import FactorTemplate, FactorParameters
from vnpy.trader.constant import Interval
from vnpy.factor.backtesting.backtesting import BacktestEngine, DEFAULT_DATETIME_COL

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


if __name__ == '__main__':
    unittest.main()
