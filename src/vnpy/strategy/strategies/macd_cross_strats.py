from logging import INFO, WARNING, ERROR
from typing import Any

import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from vnpy.strategy.engine import StrategyEngine
from vnpy.trader.object import OrderRequest, Direction, Offset, OrderType

# Import the base strategy template
from vnpy.strategy.template import StrategyTemplate

# --- Constants ---
DEFAULT_DATETIME_COL = "datetime"
DEFAULT_VT_SYMBOL_COL = "vt_symbol"


class MacdCrossStrategy(StrategyTemplate):
    """
    A generic strategy that uses a predictive model (e.g., Logistic Regression)
    on a dynamically defined set of features from the configuration file.
    """

    author = "Quant Developer"

    def __init__(self, engine_interface: "StrategyEngine", settings: dict[str, Any]):
        """
        Initializes the generic strategy.
        """
        super().__init__(engine_interface, settings)

        # Get feature mapping from model_params in the JSON config
        self.model_params: dict = settings.get("model_params", {})
        self.feature_map: dict = self.model_params.get("features", {})

        # Initialize a default model if one isn't loaded from a file
        if self.model is None:
            sklearn_params = self.model_params.get(
                "sklearn_params", {"random_state": 42}
            )
            self.model = LogisticRegression(**sklearn_params)
            self.write_log(
                f"Initialized with a new {self.model_params.get('model_name')} model.",
                INFO,
            )

    # -----------------------------------------------------------------
    # --- Implementation of Abstract Methods from StrategyTemplate ---
    # -----------------------------------------------------------------

    def _transform_latest_factors(
        self, latest_factor_data_map: dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Dynamically combines all required factor components into a single feature DataFrame.
        """
        if not self.feature_map:
            self.write_log(
                "No features defined in model_params. Cannot transform data.", ERROR
            )
            return pl.DataFrame()

        # Join all feature dataframes together
        all_features_long = []
        for feature_alias, factor_key in self.feature_map.items():
            df_feature = latest_factor_data_map.get(factor_key)
            if df_feature is None:
                self.write_log(
                    f"Feature '{feature_alias}' with key '{factor_key}' not found in received data.",
                    WARNING,
                )
                return pl.DataFrame()  # Return empty if any required feature is missing

            # Reshape from wide to long and rename the value column to the feature alias
            df_long = df_feature.melt(
                id_vars=[DEFAULT_DATETIME_COL],
                variable_name=DEFAULT_VT_SYMBOL_COL,
                value_name=feature_alias,
            )
            all_features_long.append(df_long)

        # Sequentially join the long DataFrames
        if not all_features_long:
            return pl.DataFrame()

        final_features_df = all_features_long[0]
        for i in range(1, len(all_features_long)):
            final_features_df = final_features_df.join(
                all_features_long[i],
                on=[DEFAULT_DATETIME_COL, DEFAULT_VT_SYMBOL_COL],
                how="inner",
            )

        return final_features_df

    def predict_from_model(self, data: pd.DataFrame) -> pd.Series:
        """
        Uses the trained model to predict the trading direction based on the
        dynamically provided features.
        """
        if not hasattr(self.model, "classes_"):
            self.write_log("Model has not been trained yet. Cannot predict.", WARNING)
            return pd.Series()

        # Get the feature names from the config to ensure correct column order
        feature_names = list(self.feature_map.keys())

        # Ensure all required feature columns are in the dataframe
        if not all(f in data.columns for f in feature_names):
            self.write_log(
                f"Missing one or more required features in prediction data. Have: {data.columns}, Need: {feature_names}",
                ERROR,
            )
            return pd.Series()

        features_for_model = data[feature_names]
        predictions = self.model.predict(features_for_model)

        return pd.Series(predictions, index=data.index)

    def generate_signals_from_prediction(
        self, model_output: pd.Series, symbol_feature_df: pd.DataFrame
    ) -> list[OrderRequest]:
        """
        Generates trade orders based on the model's predictions (1 for long, -1 for short).
        """
        orders = []
        trade_size = self.params.get_parameter("trading_config", {}).get(
            "trade_size", 1.0
        )
        portfolio_engine = self.strategy_engine.portfolio_engine

        for vt_symbol, prediction in model_output.items():
            current_position = portfolio_engine.get_position(
                vt_symbol, self.strategy_name
            )

            if prediction == 1 and current_position <= 0:  # Long signal
                orders.append(
                    self.create_order(
                        vt_symbol, Direction.LONG, trade_size + abs(current_position)
                    )
                )
            elif prediction == -1 and current_position >= 0:  # Short signal
                orders.append(
                    self.create_order(
                        vt_symbol, Direction.SHORT, trade_size + abs(current_position)
                    )
                )

        return orders

    def create_order(
        self, vt_symbol: str, direction: Direction, volume: float
    ) -> OrderRequest:
        """Helper function to create a market order request."""
        contract = self.get_contract(vt_symbol)
        if not contract:
            return None
        return OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            type=OrderType.MARKET,
            volume=volume,
            offset=Offset.NONE,
            reference=self.strategy_name,
        )

    def prepare_training_data(
        self, historical_factor_data_map: dict[str, pl.DataFrame]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepares historical features (X) and labels (y) for model training.
        """
        # 1. Get historical features using the same transformation logic
        features_df_pl = self._transform_latest_factors(historical_factor_data_map)
        if features_df_pl.is_empty():
            return pd.DataFrame(), pd.Series()

        # 2. Create Labels (y) using future returns
        retraining_config = self.params.get_parameter("retraining_config", {})
        lookahead_period = retraining_config.get("label_lookahead_periods", 10)
        label_factor_key = retraining_config.get(
            "label_factor_key"
        )  # e.g., "factor.1m.CLOSE"

        if not label_factor_key:
            self.write_log(
                "`label_factor_key` not defined in retraining_config. Cannot create labels.",
                ERROR,
            )
            return pd.DataFrame(), pd.Series()

        df_close = historical_factor_data_map.get(label_factor_key)
        if df_close is None:
            self.write_log(
                f"Label data with key '{label_factor_key}' not found.", ERROR
            )
            return pd.DataFrame(), pd.Series()

        df_close_long = df_close.melt(
            id_vars=[DEFAULT_DATETIME_COL],
            variable_name=DEFAULT_VT_SYMBOL_COL,
            value_name="label_price",
        )

        df_close_long = df_close_long.with_columns(
            pl.col("label_price")
            .shift(-lookahead_period)
            .over(DEFAULT_VT_SYMBOL_COL)
            .alias("future_price")
        )
        df_close_long = df_close_long.with_columns(
            pl.when(pl.col("future_price") > pl.col("label_price"))
            .then(pl.lit(1))
            .when(pl.col("future_price") < pl.col("label_price"))
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("label")
        )

        # 3. Combine features and labels
        full_training_df_pl = features_df_pl.join(
            df_close_long.select(
                [DEFAULT_DATETIME_COL, DEFAULT_VT_SYMBOL_COL, "label"]
            ),
            on=[DEFAULT_DATETIME_COL, DEFAULT_VT_SYMBOL_COL],
            how="inner",
        ).drop_nulls()

        # 4. Final preparation for scikit-learn
        full_training_df_pd = full_training_df_pl.to_pandas()
        if full_training_df_pd.empty:
            return pd.DataFrame(), pd.Series()

        feature_names = list(self.feature_map.keys())
        X = full_training_df_pd[feature_names]
        y = full_training_df_pd["label"]

        return X, y

    def on_init(self) -> None:
        """
        This strategy requires historical data for training labels, so it needs
        to subscribe to the close price factor in addition to its features.
        """
        super().on_init()

        # Dynamically add the label factor to the list of required factors
        retraining_config = self.params.get_parameter("retraining_config", {})
        label_factor_key = retraining_config.get("label_factor_key")
        if label_factor_key and label_factor_key not in self.feature_map.values():
            self.write_log(
                f"Also tracking '{label_factor_key}' for training labels.", INFO
            )
            # Explicitly subscribe to the label factor key to ensure data is provided
            if hasattr(self.strategy_engine, "subscribe_factor"):
                self.strategy_engine.subscribe_factor(label_factor_key)
            else:
                self.write_log(
                    f"Strategy engine does not support factor subscription for '{label_factor_key}'.",
                    WARNING,
                )
