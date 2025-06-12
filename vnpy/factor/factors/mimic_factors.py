from typing import Dict, List
import polars as pl
from vnpy.factor.template import FactorTemplate
from vnpy.trader.object import BarData # Not strictly needed if input_data is already DataFrame

class MimicDummyFactor(FactorTemplate):
    author = "Jules Test"
    factor_name = "MimicDummyFactor" # Used for default key generation

    # Define parameters if any; for a dummy factor, maybe none are needed
    # or perhaps a parameter for which price to output (open, high, low, close)
    # For simplicity, let's assume it always outputs close price.
    # parameters = ["price_source_column"]
    # price_source_column: str = "close"

    def __init__(self, setting: dict | None = None, dependencies_module_lookup: object | None = None, **kwargs):
        super().__init__(setting, dependencies_module_lookup, **kwargs)
        # self.price_source_column = self.get_param("price_source_column", "close")

    def get_output_schema(self) -> Dict[str, pl.DataType]:
        # Output schema: datetime, vt_symbol, and the factor value (which will be the close price)
        # FactorEngine's default datetime column name is 'datetime'
        # This factor will output in a "long" format, one row per symbol per datetime.
        return {
            "datetime": pl.Datetime(time_unit="us"),
            "vt_symbol": pl.Utf8,
            "factor_value": pl.Float64  # The column RankedFactorStrategy will use
        }

    def calculate(
        self,
        input_data: Dict[str, pl.DataFrame], # Expects OHLCV data from FactorEngine.memory_bar
        memory: 'FactorMemory',
        *args,
        **kwargs
    ) -> pl.DataFrame:
        # input_data from FactorEngine.memory_bar is Dict[str, pl.DataFrame]
        # where keys are "open", "high", "low", "close", "volume".
        # Each DataFrame is wide: index=datetime, columns=vt_symbols
        # We need to transform this to a long format: datetime, vt_symbol, factor_value

        close_prices_wide_df = input_data.get("close") # This is a Polars DataFrame

        if close_prices_wide_df is None or close_prices_wide_df.is_empty():
            # self.write_log("Close price data not found or empty in input_data.", "WARNING") # FactorTemplate has no write_log
            print(f"{self.factor_key}: Close price data not found or empty.")
            return pl.DataFrame(schema=self.get_output_schema())

        if "datetime" not in close_prices_wide_df.columns:
            print(f"{self.factor_key}: 'datetime' column missing in close prices df.")
            return pl.DataFrame(schema=self.get_output_schema())

        # Ensure vt_symbols are available (e.g., from self.vt_symbols if set by engine, or from columns)
        # FactorTemplate has self.vt_symbols, which should be populated by FactorEngine
        # The columns of close_prices_wide_df (excluding 'datetime') are the vt_symbols
        symbols_to_process = [col for col in close_prices_wide_df.columns if col != "datetime"]
        if not symbols_to_process:
            print(f"{self.factor_key}: No symbol columns found in close prices df.")
            return pl.DataFrame(schema=self.get_output_schema())

        # Take the latest row from the wide DataFrame
        latest_close_prices_wide = close_prices_wide_df.tail(1)
        if latest_close_prices_wide.is_empty():
            return pl.DataFrame(schema=self.get_output_schema())

        # Melt the latest row to long format
        # Resulting columns from melt: 'datetime', 'variable' (symbol), 'value' (close price)
        latest_datetime = latest_close_prices_wide.get_column("datetime")[0] # Get the single datetime value

        data_for_long_df = []
        for vt_symbol in symbols_to_process:
            if vt_symbol in latest_close_prices_wide.columns:
                close_price = latest_close_prices_wide.get_column(vt_symbol)[0]
                if close_price is not None: # Check for missing data for a symbol
                     data_for_long_df.append({
                         "datetime": latest_datetime,
                         "vt_symbol": vt_symbol,
                         "factor_value": float(close_price) # Ensure it's float
                     })

        if not data_for_long_df:
            return pl.DataFrame(schema=self.get_output_schema())

        long_df = pl.DataFrame(data_for_long_df, schema=self.get_output_schema())

        # The FactorEngine expects the *entire* historical DataFrame to be returned.
        # The 'memory' object can be used to get existing data and append.
        # For simplicity in this dummy factor for testing, we'll just return the latest calculated values.
        # A real factor would do:
        # current_factor_history = memory.get_data()
        # updated_history = pl.concat([current_factor_history, long_df]).unique(subset=["datetime", "vt_symbol"], keep="last").sort("datetime")
        # return updated_history
        # However, FactorEngine.execute_calculation calls fm_instance.update_data(result_df),
        # and FactorMemory.update_data handles merging & truncation.
        # So, returning just the new row(s) should be fine.
        return long_df
