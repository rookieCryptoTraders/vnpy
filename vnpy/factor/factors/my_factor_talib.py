import polars as pl
import polars_talib as plta  # Import the new technical analysis extension

from vnpy.factor.memory import FactorMemory
from vnpy.factor.template import FactorTemplate

# Assume this is defined globally or in a constants file
DEFAULT_DATETIME_COL = "datetime"


class EMAFactor(FactorTemplate):
    """
    Calculates the Exponential Moving Average (EMA) for each symbol using
    the high-performance polars_ta_extension library.
    """

    author = "VN Trader Community"
    factor_name = "EMAFactor"

    def __init__(
        self,
        setting: dict | None = None,
        vt_symbols: list[str] | None = None,
        **kwargs,
    ):
        """
        Initializes the EMAFactor.

        This factor requires a 'period' parameter to be defined in its settings.
        """
        # The parent __init__ handles loading parameters from the 'setting' dict
        super().__init__(setting, **kwargs)
        self.vt_symbols: list[str] = vt_symbols if vt_symbols else []

        # Validate that the required 'period' parameter exists and is valid
        if not hasattr(self.params, "period"):
            raise ValueError(
                f"EMAFactor ({self.factor_key}) requires a 'period' parameter."
            )
        try:
            # The library expects 'timeperiod', so we store the validated parameter
            self.period: int = int(self.params.period)
            if self.period <= 0:
                raise ValueError("Period must be a positive integer.")
        except (ValueError, TypeError) as e:
            raise e

    def calculate(
        self,
        input_data: dict[str, pl.DataFrame],
        memory: FactorMemory,
    ) -> pl.DataFrame:
        """
        Calculates the EMA for all relevant symbols in a single, vectorized operation.

        Args:
            input_data: A dictionary containing historical data, expects a "close" key.
            memory: The FactorMemory instance for this factor.

        Returns:
            A Polars DataFrame with the datetime column and EMA values for each symbol.
        """
        df_close = input_data.get("close")

        # --- Input Validation ---
        if df_close is None or df_close.is_empty():
            return pl.DataFrame(data={}, schema=self.get_output_schema())

        if DEFAULT_DATETIME_COL not in df_close.columns:
            self.write_log(
                f"Required column '{DEFAULT_DATETIME_COL}' is missing in 'close' data.",
                level="ERROR",
            )
            return pl.DataFrame(data={}, schema=self.get_output_schema())

        # --- Core Calculation Logic ---
        # Identify which columns to calculate the EMA on (all except datetime)
        symbol_columns = [
            col for col in df_close.columns if col != DEFAULT_DATETIME_COL
        ]

        # If the factor is scoped to specific symbols, filter the columns
        if self.vt_symbols:
            symbol_columns = [col for col in symbol_columns if col in self.vt_symbols]

        # If no symbols are left to process, return an empty frame with the correct schema
        if not symbol_columns:
            return pl.DataFrame(data={}, schema=self.get_output_schema())

        # Create a list of EMA calculation expressions for all symbols at once.
        # Note: The extension library uses 'timeperiod' as the argument name.
        ema_expressions = [
            plta.ema(col, timeperiod=self.period).alias(col) for col in symbol_columns
        ]

        # Execute all expressions and select the result columns in a single operation
        result_df = df_close.select(pl.col(DEFAULT_DATETIME_COL), *ema_expressions)

        return result_df
