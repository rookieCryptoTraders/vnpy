# my_factor_library.py

import polars as pl

from ..memory import FactorMemory

# Assuming FactorTemplate, FactorMemory are correctly importable
from ..template import FactorTemplate

# https://github.com/Yvictor/polars_ta_extension
# using ta-lib based on polars

DEFAULT_DATETIME_COL = "datetime"


class EMAFactor(FactorTemplate):
    """
    Calculates the Exponential Moving Average (EMA) for each symbol.
    """

    author = "VN Trader Community"
    factor_name = "emafactor"  # Default, can be overridden by factor_name in settings

    def __init__(
        self,
        setting: dict | None = None,
        vt_symbols: list[str] | None = None,
        **kwargs,
    ):
        """
        Initializes EMAFactor.
        Expects 'period' in parameters.
        `vt_symbols` is a list of symbol strings for which the factor will generate outputs.
        """
        super().__init__(setting, **kwargs)  # FactorTemplate.__init__ handles params
        self.vt_symbols: list[str] = vt_symbols if vt_symbols else []

        if not hasattr(self.params, "period"):
            raise ValueError(
                f"EMAFactor ({self.factor_key}) requires 'period' parameter."
            )
        try:
            self.period: int = int(self.params.period)
            if self.period <= 0:
                raise ValueError("Period must be positive.")
        except ValueError as e:
            print(f"EMAFactor ({self.factor_key}): Invalid 'period' parameter: {e}")

    def calculate(
        self,
        input_data: dict[str, pl.DataFrame],  # Expects {"close": df_close_wide, ...}
        memory: FactorMemory,
    ) -> pl.DataFrame:
        """
        Calculates EMA based on 'close' prices.
        `input_data` provides historical OHLCV data in wide format.
        Returns a wide DataFrame: datetime | SYM1_ema | SYM2_ema | ...
        """
        df_close = input_data.get("close")
        output_schema = self.get_output_schema()

        if df_close is None or df_close.is_empty():
            return pl.DataFrame(data={}, schema=output_schema)

        if DEFAULT_DATETIME_COL not in df_close.columns:
            print(
                f"Warning: '{DEFAULT_DATETIME_COL}' missing in 'close' data for {self.factor_key}. Returning empty."
            )
            return pl.DataFrame(data={}, schema=output_schema)

        # Get the datetime column directly as a Series
        datetime_s = df_close.get_column(DEFAULT_DATETIME_COL)

        # Start a list of Series that will form the columns of the new DataFrame
        all_series_for_new_df: list[pl.Series] = [datetime_s]

        for symbol_col_name in df_close.columns:
            if symbol_col_name == DEFAULT_DATETIME_COL:
                continue

            if self.vt_symbols and symbol_col_name not in self.vt_symbols:
                continue

            ema_series = (
                df_close.get_column(symbol_col_name)
                .ewm_mean(span=self.period, adjust=False)
                .alias(symbol_col_name)
            )
            all_series_for_new_df.append(ema_series)

        # If only the datetime series is present (no EMAs calculated because no matching symbols found)
        if len(all_series_for_new_df) == 1:
            if datetime_s.is_empty():  # Should have been caught by df_close.is_empty()
                return pl.DataFrame(data={}, schema=output_schema)
            else:
                # Create a DataFrame from the datetime series first
                temp_df = datetime_s.to_frame()
                # Add other expected columns from the schema as nulls
                for col_name_in_schema, col_type_in_schema in output_schema.items():
                    if (
                        col_name_in_schema != DEFAULT_DATETIME_COL
                    ):  # If it's not the datetime col we already have
                        temp_df = temp_df.with_columns(
                            pl.lit(None, dtype=col_type_in_schema).alias(
                                col_name_in_schema
                            )
                        )
                # Ensure correct column order
                return temp_df.select(list(output_schema.keys()))

        # Construct the DataFrame directly from the list of Series
        # Each Series in all_series_for_new_df will become a column
        result_df = pl.DataFrame(all_series_for_new_df)

        # The pl.DataFrame constructor should maintain the order of series given.
        # If strict adherence to output_schema column order is paramount and might differ,
        # a final .select() could be used, but usually isn't necessary if all_series_for_new_df
        # is built in the desired order. The "no EMAs calculated" case above handles this.
        # result_df = result_df.select(list(output_schema.keys())) # Optional: if strict order needed and not guaranteed

        return result_df


class MACDFactor(FactorTemplate):
    """
    Calculates MACD (Moving Average Convergence Divergence).
    Depends on two EMAFactor instances (fast and slow).
    """

    author = "VN Trader Community"
    factor_name = "macdfactor"

    def __init__(
        self,
        setting: dict | None = None,
        vt_symbols: list[str] | None = None,
        **kwargs,
    ):
        """
        Initializes MACDFactor.
        Expects 'fast_period', 'slow_period', 'signal_period' in parameters.
        `vt_symbols` is a list of symbol strings for which the factor will generate outputs.
        """
        super().__init__(setting, **kwargs)
        self.vt_symbols: list[str] = vt_symbols if vt_symbols else []

        for factora_instance in self.dependencies_factor:
            if factora_instance.factor_name == 'fast_ema':
                self.fast_ema = factora_instance
            elif factora_instance.factor_name == 'slow_ema':
                self.slow_ema = factora_instance

        self.signal_period: int = int(self.params.signal_period)

    def calculate(
        self,
        input_data: dict[
            str, pl.DataFrame
        ],  # Expects {ema_fast_key: df_ema_fast, ema_slow_key: df_ema_slow}
        memory: FactorMemory,
    ) -> pl.DataFrame:
        """
        Calculates MACD lines using input from two dependency EMA factors,
        but outputs only the histogram.
        Returns a wide DataFrame: datetime | SYM1_histogram | SYM2_histogram | ...
        """
        output_schema = self.get_output_schema()  # Get the expected output schema

        if len(self.dependencies_factor) < 2:
            print(
                f"Warning: MACDFactor ({self.factor_key}) requires two EMA dependency factors. Returning empty."
            )
            return pl.DataFrame(data={}, schema=output_schema)

        df_ema_fast = input_data.get(self.fast_ema.factor_key)
        df_ema_slow = input_data.get(self.slow_ema.factor_key)

        if (
            df_ema_fast is None
            or df_ema_fast.is_empty()
            or df_ema_slow is None
            or df_ema_slow.is_empty()
        ):
            print(
                f"Warning: MACDFactor ({self.factor_key}) received empty or missing data from one or both EMA dependencies. Returning empty."
            )
            return pl.DataFrame(data={}, schema=output_schema)

        if (
            DEFAULT_DATETIME_COL not in df_ema_fast.columns
            or DEFAULT_DATETIME_COL not in df_ema_slow.columns
        ):
            print(
                f"Warning: '{DEFAULT_DATETIME_COL}' missing in dependency EMA data for {self.factor_key}. Returning empty."
            )
            return pl.DataFrame(data={}, schema=output_schema)

        # Get the datetime column as a Series (assuming both EMAs are aligned and have the same datetime index)
        datetime_s = df_ema_fast.get_column(DEFAULT_DATETIME_COL)

        # List to hold all Series that will form the columns of the new DataFrame
        all_series_for_new_df: list[pl.Series] = [datetime_s]

        # Determine common symbols present in both EMA DataFrames
        symbols_in_fast = set(df_ema_fast.columns) - {DEFAULT_DATETIME_COL}
        symbols_in_slow = set(df_ema_slow.columns) - {DEFAULT_DATETIME_COL}

        common_symbols = sorted(list(symbols_in_fast.intersection(symbols_in_slow)))

        # Filter by vt_symbols if provided for this MACD factor instance
        symbols_to_process = common_symbols
        if self.vt_symbols:
            symbols_to_process = [s for s in common_symbols if s in self.vt_symbols]

        if not symbols_to_process:
            print(
                f"Warning: MACDFactor ({self.factor_key}) found no common symbols to process after filtering. Input fast symbols count: {len(symbols_in_fast)}, slow count: {len(symbols_in_slow)}, configured vt_symbols count: {len(self.vt_symbols)}"
            )

        for symbol in symbols_to_process:
            fast_ema_values = df_ema_fast.get_column(symbol)
            slow_ema_values = df_ema_slow.get_column(symbol)

            # MACD line and signal line are intermediate calculations
            macd_line_values = fast_ema_values - slow_ema_values
            signal_line_values = macd_line_values.ewm_mean(
                span=self.signal_period, adjust=False
            )

            # Only the histogram is aliased and appended for the output
            histogram_values = (macd_line_values - signal_line_values).alias(symbol)

            all_series_for_new_df.append(histogram_values)

        # If only the datetime series is present (no MACD histogram values were calculated)
        if len(all_series_for_new_df) == 1:
            if (
                datetime_s.is_empty()
            ):  # Should be caught by earlier checks on df_ema_fast/slow
                return pl.DataFrame(data={}, schema=output_schema)
            else:
                # Create a DataFrame from the datetime series first
                temp_df = datetime_s.to_frame()
                # Add other expected columns from the schema (which are only histogram columns now) as nulls
                for col_name_in_schema, col_type_in_schema in output_schema.items():
                    if (
                        col_name_in_schema != DEFAULT_DATETIME_COL
                    ):  # If it's not the datetime col we already have
                        temp_df = temp_df.with_columns(
                            pl.lit(None, dtype=col_type_in_schema).alias(
                                col_name_in_schema
                            )
                        )
                # Ensure correct column order
                return temp_df.select(list(output_schema.keys()))

        # Construct the DataFrame directly from the list of Series
        result_df = pl.DataFrame(all_series_for_new_df)

        # Optional: Ensure column order matches the schema if not already guaranteed.
        # The construction of all_series_for_new_df (datetime first, then histogram per symbol)
        # should align with a typical schema order. If exact schema order is critical and might diverge:
        # result_df = result_df.select(list(output_schema.keys()))

        return result_df
