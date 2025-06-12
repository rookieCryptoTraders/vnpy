from typing import Dict, List, Any, Tuple

import pandas as pd
import polars as pl

from vnpy.trader.object import OrderRequest, BarData # Ensure BarData is imported
from vnpy.strategy.template import StrategyTemplate
from vnpy.trader.constant import Direction, Offset, OrderType
from vnpy.trader.utility import round_to


class RankedFactorStrategy(StrategyTemplate):
    """
    A strategy that ranks symbols based on a single factor and goes long on
    the top N symbols and short on the bottom M symbols.
    """
    author: str = "Jules"

    ranking_factor_name: str = "default_factor"
    factor_value_column: str = "factor_value" # This is the name of the column AFTER melting wide factor data
    num_long: int = 5
    num_short: int = 5
    percent_capital_per_position: float = 0.02
    limit_order_price_offset_ticks: int = 0

    parameters: List[str] = StrategyTemplate.parameters + [
        "ranking_factor_name",
        "factor_value_column",
        "num_long",
        "num_short",
        "percent_capital_per_position",
        "limit_order_price_offset_ticks"
    ]
    variables: List[str] = StrategyTemplate.variables + []

    def __init__(
        self,
        engine_interface: Any,
        settings: Dict[str, Any],
    ) -> None:
        super().__init__(engine_interface, settings)

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
                       f"factor_value_column (for ranked values after melt): {self.factor_value_column}, "
                       f"long: {self.num_long}, short: {self.num_short}, "
                       f"cap_alloc: {self.percent_capital_per_position*100}%, "
                       f"price_offset: {self.limit_order_price_offset_ticks} ticks", "INFO")

    def _transform_latest_factors(self, latest_factor_data_map: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Transforms the wide factor DataFrame (datetime | SYM1 | SYM2 | ...) from
        latest_factor_data_map into a long format DataFrame (datetime, vt_symbol, factor_value_column).
        """
        if not self.ranking_factor_name:
            self.write_log("ranking_factor_name is not set. Cannot transform factors.", "ERROR")
            return pl.DataFrame()

        wide_factor_df = latest_factor_data_map.get(self.ranking_factor_name)

        if wide_factor_df is None or wide_factor_df.is_empty():
            self.write_log(f"Factor data for '{self.ranking_factor_name}' not found or is empty in input map.", "WARNING")
            return pl.DataFrame()

        if "datetime" not in wide_factor_df.columns:
            self.write_log(f"Factor DataFrame for '{self.ranking_factor_name}' is missing 'datetime' column.", "ERROR")
            return pl.DataFrame()

        latest_wide_row_df = wide_factor_df.tail(1)
        if latest_wide_row_df.is_empty():
            self.write_log(f"Latest row of factor data for '{self.ranking_factor_name}' is empty.", "WARNING")
            return pl.DataFrame()

        current_datetime = latest_wide_row_df.get_column("datetime")[0]
        symbol_columns = [col for col in latest_wide_row_df.columns if col != "datetime"]

        if not symbol_columns:
            self.write_log(f"No symbol columns found in factor data for '{self.ranking_factor_name}'.", "WARNING")
            return pl.DataFrame()

        try:
            long_df = latest_wide_row_df.melt(
                id_vars=["datetime"],
                value_vars=symbol_columns,
                variable_name="vt_symbol",
                value_name=self.factor_value_column # Use the parameter here
            )
        except Exception as e:
            self.write_log(f"Error melting DataFrame for factor '{self.ranking_factor_name}': {e}", "ERROR")
            return pl.DataFrame()

        if self.vt_symbols: # Filter by strategy's subscribed symbols if specified
            long_df = long_df.filter(pl.col("vt_symbol").is_in(self.vt_symbols))

        if long_df.is_empty():
            self.write_log(f"Long format factor data is empty after melt/filter for '{self.ranking_factor_name}'.", "WARNING")

        self.write_log(f"Successfully transformed (melted) factor data for {self.ranking_factor_name}. Shape: {long_df.shape}", "DEBUG")
        return long_df

    def predict_from_model(self, data: pd.DataFrame) -> pd.DataFrame:
        self.write_log(f"Predict from model (passthrough) with data shape: {data.shape}", "DEBUG")
        return data

    def generate_signals_from_prediction(
        self, model_output: pd.DataFrame, symbol_feature_df: pd.DataFrame
    ) -> List[OrderRequest]:
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
        self.write_log(f"Ranked DataFrame head:\n{ranked_df.head()}", "DEBUG")

        # Loop for LONG orders
        long_symbols = ranked_df.head(self.num_long)["vt_symbol"].tolist()
        for vt_symbol in long_symbols:
            direction = Direction.LONG
            contract = self.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Contract not found for long symbol: {vt_symbol}", "WARNING"); continue

            price_for_volume_calc = 0.0
            base_limit_price = 0.0
            tick = self.get_tick(vt_symbol)

            if tick and tick.last_price > 0:
                self.write_log(f"Using tick data for {vt_symbol} ({direction}): LastPrice={tick.last_price}, Ask1={tick.ask_price_1}", "DEBUG")
                price_for_volume_calc = tick.last_price
                base_limit_price = tick.ask_price_1 if tick.ask_price_1 > 0 else tick.last_price
            else:
                latest_bar = self.get_latest_bar(vt_symbol)
                if latest_bar and latest_bar.close_price > 0:
                    self.write_log(f"Tick data not available for {vt_symbol} ({direction}). Using latest bar close: {latest_bar.close_price}", "DEBUG")
                    price_for_volume_calc = latest_bar.close_price
                    base_limit_price = latest_bar.close_price
                else:
                    # Fallback to price from factor data (model_output) THIS IS THE LAST RESORT
                    # This part of logic was present in previous version from subtask 009, re-integrating carefully.
                    if vt_symbol in model_output.index and self.factor_value_column in model_output.columns:
                        price_from_factor = model_output.loc[vt_symbol, self.factor_value_column]
                        if price_from_factor > 0:
                            self.write_log(f"Tick and Bar data not available for {vt_symbol} ({direction}). Using price from factor data: {price_from_factor}", "DEBUG")
                            price_for_volume_calc = price_from_factor
                            base_limit_price = price_from_factor
                        else:
                            self.write_log(f"Price from factor data for {vt_symbol} ({direction}) is not positive ({price_from_factor}). Skipping.", "WARNING")
                            continue
                    else:
                        self.write_log(f"Tick, Bar, and factor price not available for {vt_symbol} ({direction}). Skipping order.", "WARNING")
                        continue

            pricetick = contract.pricetick if contract.pricetick > 0 else 0.000001
            capital_for_position = portfolio_total_value * self.percent_capital_per_position
            if price_for_volume_calc <= 0:
                self.write_log(f"Invalid price_for_volume_calc ({price_for_volume_calc}) for {vt_symbol} ({direction}). Skipping.", "WARNING"); continue

            calculated_volume = capital_for_position / price_for_volume_calc
            volumetick = contract.volumetick if contract.volumetick > 0 else 1.0
            rounded_volume = round_to(calculated_volume, volumetick)

            if rounded_volume < contract.min_volume:
                self.write_log(f"Calculated volume {rounded_volume} for {vt_symbol} ({direction}) < min_volume {contract.min_volume}. Skipping.", "WARNING"); continue

            limit_price = base_limit_price + (self.limit_order_price_offset_ticks * pricetick)
            limit_price = round_to(limit_price, pricetick)
            if limit_price <= 0:
                self.write_log(f"Calculated invalid limit price ({limit_price}) for {vt_symbol} ({direction}). Skipping.", "WARNING"); continue

            req = OrderRequest(symbol=contract.symbol, exchange=contract.exchange, direction=direction, offset=Offset.NONE, type=OrderType.LIMIT, price=limit_price, volume=rounded_volume, reference=self.strategy_name)
            order_requests.append(req)
            self.write_log(f"Generated {direction} LIMIT order for {vt_symbol} at {limit_price}, vol {rounded_volume}", "INFO")

        # Loop for SHORT orders
        short_symbols = ranked_df.tail(self.num_short)["vt_symbol"].tolist()
        for vt_symbol in short_symbols:
            direction = Direction.SHORT
            contract = self.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Contract not found for short symbol: {vt_symbol}", "WARNING"); continue

            price_for_volume_calc = 0.0
            base_limit_price = 0.0
            tick = self.get_tick(vt_symbol)

            if tick and tick.last_price > 0:
                self.write_log(f"Using tick data for {vt_symbol} ({direction}): LastPrice={tick.last_price}, Bid1={tick.bid_price_1}", "DEBUG")
                price_for_volume_calc = tick.last_price
                base_limit_price = tick.bid_price_1 if tick.bid_price_1 > 0 else tick.last_price
            else:
                latest_bar = self.get_latest_bar(vt_symbol)
                if latest_bar and latest_bar.close_price > 0:
                    self.write_log(f"Tick data not available for {vt_symbol} ({direction}). Using latest bar close: {latest_bar.close_price}", "DEBUG")
                    price_for_volume_calc = latest_bar.close_price
                    base_limit_price = latest_bar.close_price
                else:
                    if vt_symbol in model_output.index and self.factor_value_column in model_output.columns:
                        price_from_factor = model_output.loc[vt_symbol, self.factor_value_column]
                        if price_from_factor > 0:
                            self.write_log(f"Tick and Bar data not available for {vt_symbol} ({direction}). Using price from factor data: {price_from_factor}", "DEBUG")
                            price_for_volume_calc = price_from_factor
                            base_limit_price = price_from_factor
                        else:
                            self.write_log(f"Price from factor data for {vt_symbol} ({direction}) is not positive ({price_from_factor}). Skipping.", "WARNING")
                            continue
                    else:
                        self.write_log(f"Tick, Bar, and factor price not available for {vt_symbol} ({direction}). Skipping order.", "WARNING")
                        continue

            pricetick = contract.pricetick if contract.pricetick > 0 else 0.000001
            capital_for_position = portfolio_total_value * self.percent_capital_per_position
            if price_for_volume_calc <= 0:
                self.write_log(f"Invalid price_for_volume_calc ({price_for_volume_calc}) for {vt_symbol} ({direction}). Skipping.", "WARNING"); continue

            calculated_volume = capital_for_position / price_for_volume_calc
            volumetick = contract.volumetick if contract.volumetick > 0 else 1.0
            rounded_volume = round_to(calculated_volume, volumetick)

            if rounded_volume < contract.min_volume:
                self.write_log(f"Calculated volume {rounded_volume} for {vt_symbol} ({direction}) < min_volume {contract.min_volume}. Skipping.", "WARNING"); continue

            limit_price = base_limit_price - (self.limit_order_price_offset_ticks * pricetick)
            limit_price = round_to(limit_price, pricetick)
            if limit_price <= 0:
                self.write_log(f"Calculated invalid limit price ({limit_price}) for {vt_symbol} ({direction}). Skipping.", "WARNING"); continue

            req = OrderRequest(symbol=contract.symbol, exchange=contract.exchange, direction=direction, offset=Offset.NONE, type=OrderType.LIMIT, price=limit_price, volume=rounded_volume, reference=self.strategy_name)
            order_requests.append(req)
            self.write_log(f"Generated {direction} LIMIT order for {vt_symbol} at {limit_price}, vol {rounded_volume}", "INFO")

        return order_requests

    def prepare_training_data(
        self, historical_factor_data_map: Dict[str, pl.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.write_log("prepare_training_data called (no-op)", "DEBUG")
        return pd.DataFrame(), pd.Series(dtype='float64')

    def retrain_model(self, factor_memories_for_training: Dict[str, Any]) -> None:
        self.write_log("retrain_model called (no-op)", "DEBUG")
        pass
