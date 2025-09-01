"""
Provides a refactored, parameter-driven base template for portfolio-level
trading strategies. This template standardizes parameter management and
improves configuration consistency.
"""

import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
import polars as pl

from vnpy.factor.memory import FactorMemory
from vnpy.trader.constant import Direction, Offset, OrderType
from vnpy.trader.object import (
    ContractData,
    OrderData,
    OrderRequest,
    TickData,
    TradeData,
)
from vnpy.trader.utility import round_to, virtual

try:
    from vnpy.strategy.setting import DATA_PATH, MODEL_PATH
except ImportError:
    # Fallback paths for standalone use or if settings are not configured
    MODEL_PATH = Path("./models")
    DATA_PATH = Path("./data")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    DATA_PATH.mkdir(parents=True, exist_ok=True)

if TYPE_CHECKING:
    from vnpy.strategy.engine import StrategyEngine


class StrategyParameters:
    """
    A flexible container for storing and managing strategy parameters,
    providing attribute-style access and easy updates. This class is
    analogous to FactorParameters for framework consistency.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initializes parameters from a dictionary."""
        if params is not None:
            for key, value in params.items():
                setattr(self, key, value)

    def __contains__(self, item: str) -> bool:
        """Checks if a parameter exists."""
        return hasattr(self, item)

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Sets or updates multiple parameters."""
        for key, value in params.items():
            setattr(self, key, value)

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Gets a single parameter's value, with a default."""
        return getattr(self, key, default)

    def get_all_parameters(self) -> dict[str, Any]:
        """Returns a copy of all parameters as a dictionary."""
        return self.__dict__.copy()

    def update(self, other_params: dict[str, Any] | "StrategyParameters") -> None:
        """Updates parameters from another dictionary or StrategyParameters instance."""
        params_to_update = (
            other_params
            if isinstance(other_params, dict)
            else other_params.get_all_parameters()
        )
        self.set_parameters(params_to_update)


class StrategyTemplate(ABC):
    """
    Abstract base class for portfolio trading strategies.
    Refactored to use a parameter-driven design similar to FactorTemplate.
    """

    author: str = "ccccc"

    # variables are for runtime state that gets saved/loaded
    variables: list[str] = [
        "inited",
        "trading",
        "latest_factor_update_time_iso",
        "last_retrain_time_iso",
    ]

    def __init__(
        self,
        engine_interface: "StrategyEngine",
        settings: dict[str, Any],
    ) -> None:
        """Initializes the strategy instance from a settings dictionary."""
        self.strategy_engine: StrategyEngine = engine_interface

        # Core identifiers
        self.strategy_name: str = settings.get("strategy_name", "UnnamedStrategy")
        self.class_name: str = self.__class__.__name__

        # Parameters object for all strategy-specific settings
        self.params: StrategyParameters = StrategyParameters()
        if "params" in settings and isinstance(settings["params"], dict):
            self.params.update(settings["params"])

        # Model-specific parameters, separated for clarity
        self.model_params: dict = settings.get("model_params", {})

        # Explicitly load vt_symbols for easier access
        self.vt_symbols: list[str] = self.params.get_parameter("vt_symbols", [])

        # Engine access and initial state
        self.get_tick: Callable[[str], TickData | None] = self.strategy_engine.get_tick
        self.get_contract: Callable[[str], ContractData | None] = (
            self.strategy_engine.get_contract
        )
        self.inited: bool = False
        self.trading: bool = False

        # Data and model attributes
        self.latest_factor_data: pl.DataFrame | None = None
        self.latest_factor_update_time: datetime | None = None
        self.model: Any = None
        self.last_retrain_time: datetime | None = None

        # Order management
        self.active_order_ids: set[str] = set()

        # Load model immediately on initialization if path is provided in params
        model_path = self.params.get_parameter("model_path")
        if model_path:
            self.load_model(model_path)

        self.write_log(
            f"Strategy instance '{self.strategy_name}' initialized.", level=DEBUG
        )

    # --------------------------------
    # Abstract Methods to be Implemented by User
    # --------------------------------
    @abstractmethod
    def _transform_latest_factors(
        self, latest_factor_data_map: dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """Transforms new factor data into a feature matrix for model prediction."""
        pass

    @abstractmethod
    def predict_from_model(self, data: pd.DataFrame) -> Any:
        """Generates raw predictions from the loaded model."""
        pass

    @abstractmethod
    def generate_signals_from_prediction(
        self, model_output: Any, symbol_feature_df: pd.DataFrame
    ) -> list[OrderRequest]:
        """Converts raw model predictions into specific trade orders."""
        pass

    @abstractmethod
    def prepare_training_data(
        self, historical_factor_data_map: dict[str, pl.DataFrame]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepares features (X) and labels (y) for model retraining."""
        pass

    # --------------------------------
    # Lifecycle and Event Methods
    # --------------------------------
    @virtual
    def on_factor_update(
        self, factor_memories: dict[str, FactorMemory]
    ) -> list[OrderRequest] | None:
        """Core logic triggered on every new factor data update."""
        if not self.trading or not self.inited:
            return None

        self.latest_factor_update_time = self.strategy_engine.get_current_datetime()

        # Get all required factor keys (for features and labels)
        required_keys = self.get_all_required_factor_keys()
        if not required_keys:
            self.write_log("No required factor keys defined. Cannot process.", WARNING)
            return None

        latest_polars_data_map: dict[str, pl.DataFrame] = {}
        for factor_key in required_keys:
            factor_memory = factor_memories.get(factor_key)
            if not factor_memory:
                self.write_log(f"Required factor '{factor_key}' not found.", WARNING)
                return None
            latest_rows_df = factor_memory.get_latest_rows(N=1)
            if latest_rows_df is None or latest_rows_df.is_empty():
                self.write_log(f"Factor '{factor_key}' provided empty data.", WARNING)
                return None
            latest_polars_data_map[factor_key] = latest_rows_df

        try:
            if self.model is None:
                self.write_log("Model not loaded. Cannot predict.", WARNING)
                return None

            transformed_pl_df = self._transform_latest_factors(latest_polars_data_map)
            if transformed_pl_df is None or transformed_pl_df.is_empty():
                return None

            self.latest_factor_data = transformed_pl_df
            pandas_features_df = transformed_pl_df.to_pandas().set_index("vt_symbol")

            model_prediction_output = self.predict_from_model(pandas_features_df)
            if model_prediction_output is None:
                return None

            return self.generate_signals_from_prediction(
                model_prediction_output, pandas_features_df
            )
        except Exception as e:
            self.write_log(
                f"Error in on_factor_update pipeline: {e}\n{traceback.format_exc()}",
                ERROR,
            )
            return None

    @virtual
    def retrain_model(self, factor_memories: dict[str, FactorMemory]) -> None:
        """Orchestrates model retraining using provided FactorMemory instances."""
        self.write_log("Starting model retraining process...", INFO)
        try:
            historical_data = {
                key: mem.get_data()
                for key, mem in factor_memories.items()
                if key in self.get_all_required_factor_keys()
            }
            features_df, labels_series = self.prepare_training_data(historical_data)

            if (
                features_df is None
                or features_df.empty
                or labels_series is None
                or labels_series.empty
            ):
                self.write_log(
                    "Training data prep resulted in empty features/labels.", WARNING
                )
                return

            if self.model is None or not hasattr(self.model, "fit"):
                self.write_log(
                    "Model is not initialized or does not have a 'fit' method.", ERROR
                )
                return

            self.write_log(f"Training model with {features_df.shape[0]} samples.", INFO)
            self.model.fit(features_df, labels_series)
            self.write_log("Model training completed.", INFO)

            self.save_model()
            self.last_retrain_time = self.strategy_engine.get_current_datetime()
            self.write_log(
                f"Model retraining finished. Last retrain: {self.last_retrain_time}",
                INFO,
            )
        except Exception as e:
            self.write_log(
                f"Error during model retraining: {e}\n{traceback.format_exc()}", ERROR
            )

    # --------------------------------
    # Model and State Management
    # --------------------------------
    def load_model(self, path: str | None = None) -> None:
        load_path_str = path or self.params.get_parameter("model_path")
        if not load_path_str:
            return
        load_path = Path(load_path_str)
        if not load_path.is_absolute():
            load_path = MODEL_PATH / load_path
        if not load_path.exists():
            self.write_log(f"Model file not found at {load_path}.", WARNING)
            return
        try:
            with open(load_path, "rb") as f:
                self.model = joblib.load(f)
            self.write_log(f"Model loaded from {load_path}", INFO)
        except Exception as e:
            self.write_log(f"Failed to load model from {load_path}: {e}", ERROR)

    def save_model(self, path: str | None = None) -> None:
        save_path_str = path or self.params.get_parameter("model_path")
        if not save_path_str or self.model is None:
            return
        save_path = Path(save_path_str)
        if not save_path.is_absolute():
            save_path = MODEL_PATH / save_path
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                joblib.dump(self.model, f)
            self.write_log(f"Model saved to {save_path}", INFO)
        except Exception as e:
            self.write_log(f"Failed to save model to {save_path}: {e}", ERROR)

    def to_setting(self) -> dict:
        model_params = self.model_params.copy()
        if self.model and hasattr(self.model, "get_params"):
            try:
                model_params["sklearn_params"] = self.model.get_params()
            except Exception as e:
                self.write_log(f"Could not get sklearn_params: {e}", WARNING)
        return {
            "strategy_name": self.strategy_name,
            "class_name": self.class_name,
            "params": self.params.get_all_parameters(),
            "model_params": model_params,
        }

    def get_data(self) -> dict:
        data = {"active_order_ids": list(self.active_order_ids)}
        for name in self.variables:
            value = getattr(self, name, None)
            if isinstance(value, datetime):
                data[name] = value.isoformat()
            else:
                data[name] = value
        return data

    def load_data(self, data: dict) -> None:
        self.active_order_ids = set(data.get("active_order_ids", []))
        time_fields = {
            "latest_factor_update_time_iso": "latest_factor_update_time",
            "last_retrain_time_iso": "last_retrain_time",
        }
        for key, attr in time_fields.items():
            if time_str := data.get(key):
                try:
                    setattr(self, attr, datetime.fromisoformat(time_str))
                except Exception:
                    self.write_log(f"Could not parse saved time for {attr}", WARNING)
        for name in self.variables:
            if name in data and name not in time_fields:
                setattr(self, name, data[name])
        self.write_log("Runtime data loaded.", DEBUG)

    # --------------------------------
    # Order and Trading Methods
    # --------------------------------
    def send_order(
        self,
        vt_symbol: str,
        direction: Direction,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
        offset: Offset = Offset.NONE,
    ) -> list[str]:
        """A convenience method to send an order."""
        if not self.trading:
            return []

        contract = self.get_contract(vt_symbol)
        if not contract:
            self.write_log(
                f"Order rejected: Contract not found for {vt_symbol}.", ERROR
            )
            return []

        # Round price and volume to contract specifications
        price = round_to(price, contract.pricetick or 1e-8)
        volume = round_to(volume, contract.volumetick or 1e-8)

        if volume < (contract.min_volume or 1e-8):
            self.write_log(
                f"Order rejected: Vol {volume} < min_vol {contract.min_volume}", WARNING
            )
            return []

        req = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            type=order_type,
            price=price,
            volume=volume,
            offset=offset,
            reference=self.strategy_name,
        )
        return self.strategy_engine.send_order(self.strategy_name, req)

    def cancel_order(self, vt_orderid: str) -> None:
        """Cancels a specific active order."""
        if not self.trading:
            return
        order = self.strategy_engine.main_engine.get_order(vt_orderid)
        if order:
            self.strategy_engine.cancel_order(order.create_cancel_request())

    def cancel_all_orders(self) -> None:
        """Cancels all active orders for this strategy."""
        if not self.trading or not self.active_order_ids:
            return
        self.write_log(
            f"Cancelling all {len(self.active_order_ids)} active orders...", INFO
        )
        for vt_orderid in list(self.active_order_ids):
            self.cancel_order(vt_orderid)

    # --------------------------------
    # Portfolio Integration Methods (NEW)
    # --------------------------------
    def get_position(self, vt_symbol: str) -> float:
        """
        Gets the current net position for a specific symbol within this
        strategy's portfolio slice.

        Returns:
            float: The number of contracts held. Positive for long,
                   negative for short, 0 for flat.
        """
        if self.strategy_engine.portfolio_engine:
            return self.strategy_engine.portfolio_engine.get_position(
                self.strategy_name, vt_symbol
            )
        self.write_log("PortfolioEngine not available, cannot get position.", WARNING)
        return 0.0

    def get_portfolio_value(self) -> float:
        """
        Gets the total current market value of this strategy's entire portfolio slice.
        """
        if self.strategy_engine.portfolio_engine:
            result = self.strategy_engine.portfolio_engine.get_portfolio_result(
                self.strategy_name
            )
            return result.value if result else 0.0
        self.write_log(
            "PortfolioEngine not available, cannot get portfolio value.", WARNING
        )
        return 0.0

    # --------------------------------
    # Virtual Methods
    # --------------------------------
    @virtual
    def on_init(self) -> None:
        self.inited = True
        self.write_log("Strategy initialized.")

    @virtual
    def on_start(self) -> None:
        self.trading = True
        self.write_log("Strategy started.")

    @virtual
    def on_stop(self) -> None:
        self.trading = False
        self.cancel_all_orders()  # <-- NEW: Cancel all orders on stop
        self.write_log("Strategy stopped.")

    @virtual
    def on_order(self, order: OrderData) -> None:
        """
        Callback for order updates. Automatically manages active_order_ids.
        Subclasses can override this but should call super().on_order(order).
        """
        if order.is_active():
            self.active_order_ids.add(order.vt_orderid)
        else:
            self.active_order_ids.discard(order.vt_orderid)

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        pass

    @virtual
    def on_timer(self) -> None:
        pass

    @virtual
    def get_all_required_factor_keys(self) -> set[str]:
        """
        Returns a set of all factor keys this strategy needs to function.
        """
        feature_map = self.model_params.get("features", {})
        required_keys = set(feature_map.values())
        retraining_config = self.params.get_parameter("retraining_config", {})
        if label_key := retraining_config.get("label_factor_key"):
            required_keys.add(label_key)
        return required_keys

    def write_log(self, msg: str, level: int = INFO) -> None:
        self.strategy_engine.write_log(msg, self, level=level)

    def put_event(self) -> None:
        self.strategy_engine.put_strategy_update_event(self)
