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
from vnpy.trader.object import (
    ContractData,
    OrderData,
    OrderRequest,
    TickData,
    TradeData,
)
from vnpy.trader.utility import virtual

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

    author: str = "Unknown"

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

        required_keys = self.params.get_parameter("required_factor_keys", [])
        if not required_keys:
            self.write_log(
                "No required_factor_keys defined in params. Cannot process.", WARNING
            )
            return None

        latest_polars_data_map: dict[str, pl.DataFrame] = {}
        for factor_key in required_keys:
            factor_memory = factor_memories.get(factor_key)
            if not factor_memory:
                self.write_log(
                    f"Required factor '{factor_key}' not found in received data.",
                    WARNING,
                )
                return None
            latest_rows_df = factor_memory.get_latest_rows(N=1)
            if latest_rows_df is None or latest_rows_df.is_empty():
                self.write_log(
                    f"Factor '{factor_key}' provided empty data. Skipping.", WARNING
                )
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

    # --------------------------------
    # Model Management
    # --------------------------------
    def load_model(self, path: str | None = None) -> None:
        """Loads a model from the specified path or the path in parameters."""
        load_path_str = path or self.params.get_parameter("model_path")
        if not load_path_str:
            self.write_log("No model_path in params. Skipping model load.", DEBUG)
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
            self.write_log(f"Model loaded successfully from {load_path}", INFO)
        except Exception as e:
            self.write_log(f"Failed to load model from {load_path}: {e}", ERROR)

    def save_model(self, path: str | None = None) -> None:
        """Saves the current model to the specified path or the path in parameters."""
        save_path_str = path or self.params.get_parameter("model_path")
        if not save_path_str or self.model is None:
            self.write_log("Skipping save: No save path or model is None.", DEBUG)
            return

        save_path = Path(save_path_str)
        if not save_path.is_absolute():
            save_path = MODEL_PATH / save_path

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                joblib.dump(self.model, f)
            self.write_log(f"Model saved successfully to {save_path}", INFO)
        except Exception as e:
            self.write_log(f"Failed to save model to {save_path}: {e}", ERROR)

    def check_retraining_schedule(self, current_datetime: datetime) -> bool:
        """Checks if model retraining is due based on configuration."""
        retraining_config = self.params.get_parameter("retraining_config", {})
        interval_days = retraining_config.get("frequency_days", 0)

        if interval_days <= 0:
            return False
        if self.last_retrain_time is None:
            return True

        days_since = (
            current_datetime.replace(tzinfo=None)
            - self.last_retrain_time.replace(tzinfo=None)
        ).days
        return days_since >= interval_days

    # --------------------------------
    # State Serialization
    # --------------------------------
    def to_setting(self) -> dict:
        """
        Serializes the strategy's configuration to a dictionary,
        including model-specific parameters for traceability.
        """
        model_params = {}
        if self.model:
            # Add support for scikit-learn models
            if hasattr(self.model, "get_params") and callable(self.model.get_params):
                try:
                    model_params["sklearn_params"] = self.model.get_params()
                except Exception as e:
                    self.write_log(f"Could not get sklearn_params: {e}", WARNING)
            # Add more `elif` blocks here for other model types (Keras, PyTorch, etc.)

        settings = {
            "strategy_name": self.strategy_name,
            "class_name": self.class_name,
            "params": self.params.get_all_parameters(),
            "model_params": model_params,
        }
        return settings

    def get_data(self) -> dict:
        """Serializes the strategy's runtime state."""
        data = {"active_order_ids": list(self.active_order_ids)}
        for name in self.variables:
            value = getattr(self, name, None)
            if isinstance(value, datetime):
                data[name] = value.isoformat()
            elif name.endswith("_iso") and isinstance(value, str):
                data[name] = value
            else:
                data[name] = value
        return data

    def load_data(self, data: dict) -> None:
        """Loads the strategy's runtime state from a dictionary."""
        self.write_log("Loading runtime data into strategy...", DEBUG)
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

    def update_setting(self, settings: dict) -> None:
        """Updates strategy parameters dynamically from a new settings dictionary."""
        self.write_log(
            f"Updating strategy settings for '{self.strategy_name}'...", DEBUG
        )
        if params_to_update := settings.get("params"):
            self.params.update(params_to_update)

        self.load_model()  # Reload model in case path changed
        self.write_log("Strategy settings updated.", INFO)
        self.put_event()

    # --------------------------------
    # Utility and Virtual Methods
    # --------------------------------
    def write_log(self, msg: str, level: int = INFO) -> None:
        """Writes a log message via the strategy engine."""
        self.strategy_engine.write_log(msg, self, level=level)

    def put_event(self) -> None:
        """Puts a strategy update event onto the event bus."""
        self.strategy_engine.put_strategy_update_event(self)

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
        self.write_log("Strategy stopped.")

    @virtual
    def on_order(self, order: OrderData) -> None:
        pass

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        pass

    @virtual
    def on_timer(self) -> None:
        pass
