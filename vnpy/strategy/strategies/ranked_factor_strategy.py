from typing import Dict, List, Any, Tuple

import pandas as pd
import polars as pl

from vnpy.trader.object import OrderRequest
from vnpy.strategy.template import StrategyTemplate
from vnpy.trader.constant import Direction, Offset, OrderType # Ensure OrderType is imported
from vnpy.trader.utility import round_to # Import round_to


class RankedFactorStrategy(StrategyTemplate):
    """
    A strategy that ranks symbols based on a single factor and goes long on
    the top N symbols and short on the bottom M symbols.
    """
    author: str = "Jules"

    # Parameters for the strategy, configured via strategy_config.json
    ranking_factor_name: str = "default_factor"  # Name of the factor to rank by
    factor_value_column: str = "factor_value"    # Column name in the factor DataFrame that holds the value
    num_long: int = 5                            # Number of symbols to long
    num_short: int = 5                           # Number of symbols to short

    percent_capital_per_position: float = 0.02   # e.g., 0.02 for 2% of portfolio capital per chosen symbol
    limit_order_price_offset_ticks: int = 0      # Price offset in ticks for limit orders

    # List of parameters to be saved and loaded
    parameters: List[str] = StrategyTemplate.parameters + [
        "ranking_factor_name",
        "factor_value_column",
        "num_long",
        "num_short",
        "percent_capital_per_position",
        "limit_order_price_offset_ticks"
    ]

    # List of variables to be saved and loaded
    variables: List[str] = StrategyTemplate.variables + []

    def __init__(
        self,
        engine_interface: Any,
        settings: Dict[str, Any],
    ) -> None:
        """
        Initialize the strategy with settings.
        """
        super().__init__(engine_interface, settings)

        # Initialize strategy-specific parameters from settings
        self.ranking_factor_name = settings.get("ranking_factor_name", self.ranking_factor_name)
        self.factor_value_column = settings.get("factor_value_column", self.factor_value_column)
        self.num_long = settings.get("num_long", self.num_long)
        self.num_short = settings.get("num_short", self.num_short)
        self.percent_capital_per_position = settings.get("percent_capital_per_position", self.percent_capital_per_position)
        self.limit_order_price_offset_ticks = settings.get("limit_order_price_offset_ticks", self.limit_order_price_offset_ticks)

        if self.ranking_factor_name:
            self.required_factor_keys = [self.ranking_factor_name]
        else:
            self.required_factor_keys = []
            self.write_log("No ranking_factor_name provided. Strategy may not function.", "WARNING")

        self.write_log(f"RankedFactorStrategy initialized with factor: {self.ranking_factor_name}, "
                       f"long: {self.num_long}, short: {self.num_short}, "
                       f"cap_alloc: {self.percent_capital_per_position*100}%, "
                       f"price_offset: {self.limit_order_price_offset_ticks} ticks", "INFO")

    def _transform_latest_factors(self, latest_factor_data_map: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Extracts the relevant factor DataFrame.
        The input DataFrame is assumed to have vt_symbol and the factor value column.
        """
        if not self.ranking_factor_name:
            self.write_log("ranking_factor_name is not set. Cannot transform factors.", "ERROR")
            return pl.DataFrame()

        factor_df = latest_factor_data_map.get(self.ranking_factor_name)

        if factor_df is None:
            self.write_log(f"Factor data for '{self.ranking_factor_name}' not found in input.", "WARNING")
            return pl.DataFrame()

        if factor_df.is_empty():
            self.write_log(f"Factor data for '{self.ranking_factor_name}' is empty.", "WARNING")
            return pl.DataFrame()

        if "vt_symbol" not in factor_df.columns:
            self.write_log(f"Factor DataFrame for '{self.ranking_factor_name}' is missing 'vt_symbol' column.", "ERROR")
            return pl.DataFrame()

        if self.factor_value_column not in factor_df.columns:
            self.write_log(f"Factor DataFrame for '{self.ranking_factor_name}' is missing "
                           f"'{self.factor_value_column}' column.", "ERROR")
            return pl.DataFrame()

        self.write_log(f"Transformed factor data for {self.ranking_factor_name} successfully.", "DEBUG")
        return factor_df

    def predict_from_model(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        No ML model is used. This method returns the input data as is.
        """
        self.write_log(f"Predict from model (passthrough) with data shape: {data.shape}", "DEBUG")
        return data

    def generate_signals_from_prediction(
        self, model_output: pd.DataFrame, symbol_feature_df: pd.DataFrame
    ) -> List[OrderRequest]:
        """
        Generates buy/sell signals based on ranked factor values.
        model_output is a Pandas DataFrame with 'vt_symbol' as index and factor values in columns.
        """
        self.write_log(f"Generating signals from prediction. Input shape: {model_output.shape}", "DEBUG")
        order_requests: List[OrderRequest] = []

        if model_output.empty:
            self.write_log("Model output is empty. No signals generated.", "WARNING")
            return order_requests

        if self.factor_value_column not in model_output.columns:
            self.write_log(f"Factor value column '{self.factor_value_column}' not found in model output. "
                           f"Available columns: {model_output.columns.tolist()}", "ERROR")
            return order_requests

        portfolio_total_value = None
        if self.portfolio_result:
            portfolio_total_value = self.portfolio_result.get_total_value()

        if not portfolio_total_value or portfolio_total_value <= 0:
            self.write_log("Portfolio total value is not available or non-positive. Skipping order generation.", "WARNING")
            return order_requests

        ranked_df = model_output.reset_index().sort_values(by=self.factor_value_column, ascending=False)
        self.write_log(f"Ranked DataFrame head:\n{ranked_df.head()}", "DEBUG") # Escaped


        # Generate long orders
        long_symbols = ranked_df.head(self.num_long)["vt_symbol"].tolist()
        for vt_symbol in long_symbols:
            direction = Direction.LONG # Define direction for this loop
            contract = self.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Contract not found for long symbol: {vt_symbol}", "WARNING")
                continue

            price_for_volume_calc = 0.0
            base_limit_price = 0.0

            tick = self.get_tick(vt_symbol)
            if tick and tick.last_price > 0:
                self.write_log(f"Using tick data for {vt_symbol} (Long): LastPrice={tick.last_price}, Ask1={tick.ask_price_1}", "DEBUG")
                price_for_volume_calc = tick.last_price
                base_limit_price = tick.ask_price_1 if tick.ask_price_1 > 0 else tick.last_price
            else:
                if vt_symbol in model_output.index and self.factor_value_column in model_output.columns:
                    price_from_factor = model_output.loc[vt_symbol, self.factor_value_column]
                    if price_from_factor > 0:
                        self.write_log(f"Tick data not available for {vt_symbol} (Long). Using price from factor data: {price_from_factor}", "DEBUG")
                        price_for_volume_calc = price_from_factor
                        base_limit_price = price_from_factor
                    else:
                        self.write_log(f"Price from factor data for {vt_symbol} (Long) is not positive ({price_from_factor}). Skipping.", "WARNING")
                        continue
                else:
                    self.write_log(f"Tick data and factor price not available for {vt_symbol} (Long). Skipping.", "WARNING")
                    continue

            pricetick = contract.pricetick if contract.pricetick > 0 else 0.000001

            capital_for_position = portfolio_total_value * self.percent_capital_per_position
            if price_for_volume_calc <= 0: # Should be caught by earlier checks, but as safeguard
                 self.write_log(f"Price for volume calculation is not positive for {vt_symbol} (Long): {price_for_volume_calc}. Skipping.", "WARNING")
                 continue
            calculated_volume = capital_for_position / price_for_volume_calc
            volumetick = contract.volumetick if contract.volumetick > 0 else 1.0
            rounded_volume = round_to(calculated_volume, volumetick)

            if rounded_volume < contract.min_volume:
                self.write_log(f"Calculated volume {rounded_volume} for {vt_symbol} (Long) is less than min_volume {contract.min_volume}. Skipping.", "WARNING")
                continue

            limit_price = base_limit_price + (self.limit_order_price_offset_ticks * pricetick)
            limit_price = round_to(limit_price, pricetick)

            req = OrderRequest(
                symbol=contract.symbol,
                exchange=contract.exchange,
                direction=direction,
                offset=Offset.NONE,
                type=OrderType.LIMIT,
                price=limit_price,
                volume=rounded_volume,
                reference=self.strategy_name
            )
            order_requests.append(req)
            self.write_log(f"Generated LONG LIMIT order for {vt_symbol} at {limit_price}, vol {rounded_volume}", "INFO")

        # Generate short orders
        short_symbols = ranked_df.tail(self.num_short)["vt_symbol"].tolist()
        for vt_symbol in short_symbols:
            direction = Direction.SHORT # Define direction for this loop
            contract = self.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Contract not found for short symbol: {vt_symbol}", "WARNING")
                continue

            price_for_volume_calc = 0.0
            base_limit_price = 0.0

            tick = self.get_tick(vt_symbol)
            if tick and tick.last_price > 0:
                self.write_log(f"Using tick data for {vt_symbol} (Short): LastPrice={tick.last_price}, Bid1={tick.bid_price_1}", "DEBUG")
                price_for_volume_calc = tick.last_price
                base_limit_price = tick.bid_price_1 if tick.bid_price_1 > 0 else tick.last_price
            else:
                if vt_symbol in model_output.index and self.factor_value_column in model_output.columns:
                    price_from_factor = model_output.loc[vt_symbol, self.factor_value_column]
                    if price_from_factor > 0:
                        self.write_log(f"Tick data not available for {vt_symbol} (Short). Using price from factor data: {price_from_factor}", "DEBUG")
                        price_for_volume_calc = price_from_factor
                        base_limit_price = price_from_factor
                    else:
                        self.write_log(f"Price from factor data for {vt_symbol} (Short) is not positive ({price_from_factor}). Skipping.", "WARNING")
                        continue
                else:
                    self.write_log(f"Tick data and factor price not available for {vt_symbol} (Short). Skipping.", "WARNING")
                    continue

            pricetick = contract.pricetick if contract.pricetick > 0 else 0.000001

            capital_for_position = portfolio_total_value * self.percent_capital_per_position
            if price_for_volume_calc <= 0: # Should be caught by earlier checks, but as safeguard
                 self.write_log(f"Price for volume calculation is not positive for {vt_symbol} (Short): {price_for_volume_calc}. Skipping.", "WARNING")
                 continue
            calculated_volume = capital_for_position / price_for_volume_calc
            volumetick = contract.volumetick if contract.volumetick > 0 else 1.0
            rounded_volume = round_to(calculated_volume, volumetick)

            if rounded_volume < contract.min_volume:
                self.write_log(f"Calculated volume {rounded_volume} for {vt_symbol} (Short) is less than min_volume {contract.min_volume}. Skipping.", "WARNING")
                continue

            limit_price = base_limit_price - (self.limit_order_price_offset_ticks * pricetick)
            limit_price = round_to(limit_price, pricetick)

            req = OrderRequest(
                symbol=contract.symbol,
                exchange=contract.exchange,
                direction=direction,
                offset=Offset.NONE,
                type=OrderType.LIMIT,
                price=limit_price,
                volume=rounded_volume,
                reference=self.strategy_name
            )
            order_requests.append(req)
            self.write_log(f"Generated SHORT LIMIT order for {vt_symbol} at {limit_price}, vol {rounded_volume}", "INFO")

        return order_requests

    def prepare_training_data(
        self, historical_factor_data_map: Dict[str, pl.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.write_log("prepare_training_data called (no-op)", "DEBUG")
        return pd.DataFrame(), pd.Series(dtype='float64')

    def retrain_model(self, factor_memories_for_training: Dict[str, Any]) -> None:
        self.write_log("retrain_model called (no-op)", "DEBUG")
        pass
