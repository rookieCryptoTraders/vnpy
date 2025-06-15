"""
Defines the StrategyEngine, the core component for managing and executing
trading strategies within the vn.py framework.
"""

import importlib
import inspect
import pkgutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from logging import ERROR, INFO, WARNING
from pathlib import Path
from typing import Any
from collections.abc import Callable


# --- Strategy & Portfolio Specific Imports ---

# --- VnTrader Core Imports ---
from vnpy.event import Event, EventEngine
from vnpy.factor.memory import FactorMemory
from vnpy.strategy.template import StrategyTemplate
from vnpy.trader.constant import EngineType
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.event import (
    EVENT_FACTOR,
    EVENT_LOG,
    EVENT_ORDER,
    EVENT_TIMER,
    EVENT_TRADE,
)
from vnpy.trader.object import (
    CancelRequest,
    LogData,
    OrderRequest,
)
from vnpy.trader.utility import load_json, save_json

# --- Settings Integration ---
from vnpy.strategy.setting import (
    DATA_PATH,
    MODEL_PATH,
    STRATEGY_ENGINE_OPERATIONAL_PARAMS,
    get_strategy_instance_definitions_filename
)

# --- Constants ---
STRATEGY_ENGINE_APP_NAME: str = "StrategyEngine"


class StrategyEngine(BaseEngine):
    """
    Orchestrates the lifecycle of all trading strategies, from loading and
    initialization to event processing and shutdown.
    """

    engine_type: EngineType = EngineType.LIVE

    def __init__(
        self,
        main_engine: MainEngine,
        event_engine: EventEngine,
    ) -> None:
        super().__init__(
            main_engine, event_engine, engine_name=STRATEGY_ENGINE_APP_NAME
        )

        # Load configuration from settings
        self.strategy_code_module_path: str = STRATEGY_ENGINE_OPERATIONAL_PARAMS.get(
            "strategy_code_module_path", "strategies"
        )
        self.execution_gateway_name: str = STRATEGY_ENGINE_OPERATIONAL_PARAMS.get(
            "default_execution_gateway", "DEFAULT_GW"
        )
        self.init_max_workers: int = STRATEGY_ENGINE_OPERATIONAL_PARAMS.get(
            "init_max_workers", 1
        )
        self.definitions_filename: Path = get_strategy_instance_definitions_filename()

        # Core state
        self.strategy_classes: dict[str, type[StrategyTemplate]] = {}
        self.strategies: dict[str, StrategyTemplate] = {}

        # Factor data cache
        self.latest_factor_memories: dict[str, FactorMemory] = {}
        self.factor_update_time: datetime | None = None

        self.write_log(
            f"StrategyEngine initialized. Definitions name: {self.definitions_filename}",
            level=INFO,
        )

    def init_engine(self) -> None:
        """Main initialization sequence for the StrategyEngine."""
        self.write_log("Starting StrategyEngine initialization...")
        self._load_all_strategy_classes()
        self.init_strategies_from_configs()
        self.load_all_strategy_runtime_data()
        self.register_event()
        self.write_log(
            f"StrategyEngine ready. {len(self.strategies)} instances active.",
            level=INFO,
        )

    def close(self) -> None:
        """Shuts down the engine, stopping strategies and saving their settings."""
        self.write_log(f"Shutting down {self.engine_name}...")
        self.stop_all_strategies()
        self.save_all_strategy_settings()  # Save settings on close
        self.save_all_strategy_runtime_data()  # Save runtime state
        self.unregister_event()
        self.init_executor.shutdown(wait=True)
        self.write_log(f"{self.engine_name} shut down complete.", level=INFO)

    def register_event(self) -> None:
        """Registers all necessary event listeners."""
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_FACTOR, self.process_factor_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def unregister_event(self) -> None:
        """Unregisters all event listeners."""
        self.event_engine.unregister(EVENT_ORDER, self.process_order_event)
        self.event_engine.unregister(EVENT_TRADE, self.process_trade_event)
        self.event_engine.unregister(EVENT_FACTOR, self.process_factor_event)
        self.event_engine.unregister(EVENT_TIMER, self.process_timer_event)

    def _load_all_strategy_classes(self) -> None:
        """Discovers and loads all StrategyTemplate subclasses from the code module path."""
        self.write_log(
            f"Loading strategy classes from module: {self.strategy_code_module_path}",
            level=INFO,
        )
        try:
            for importer, modname, ispkg in pkgutil.walk_packages(
                path=[self.strategy_code_module_path]
            ):
                module = importlib.import_module(
                    f"{self.strategy_code_module_path}.{modname}"
                )
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, StrategyTemplate)
                        and obj is not StrategyTemplate
                    ):
                        self.strategy_classes[obj.__name__] = obj
        except Exception as e:
            self.write_log(
                f"Error loading strategy classes: {e}\n{traceback.format_exc()}",
                level=ERROR,
            )
        self.write_log(
            f"Loaded {len(self.strategy_classes)} strategy classes.", level=INFO
        )

    def init_strategies_from_configs(self) -> None:
        """Initializes strategy instances based on the definitions JSON file."""
        self.write_log(
            f"Initializing strategies from {self.definitions_filename}...", level=INFO
        )
        if not self.definitions_filename.exists():
            self.write_log("Definitions file not found. No strategies loaded.", WARNING)
            return

        strategy_settings = load_json(str(self.definitions_filename))
        for setting in strategy_settings:
            class_name = setting.get("class_name")
            strategy_name = setting.get("strategy_name")
            if not class_name or not strategy_name:
                self.write_log(f"Skipping invalid strategy config: {setting}", WARNING)
                continue

            if class_name in self.strategy_classes:
                strategy_class = self.strategy_classes[class_name]
                self.strategies[strategy_name] = strategy_class(self, setting)
            else:
                self.write_log(
                    f"Strategy class '{class_name}' not found for '{strategy_name}'.",
                    ERROR,
                )

    def process_factor_event(self, event: Event) -> None:
        """Processes factor updates and forwards them to all active strategies."""
        self.latest_factor_memories = event.data
        for strategy in self.strategies.values():
            if strategy.inited and strategy.trading:
                self.call_strategy_func(
                    strategy, strategy.on_factor_update, self.latest_factor_memories
                )

    def process_timer_event(self, event: Event) -> None:
        """Processes timer events for periodic tasks like model retraining."""
        now = self.get_current_datetime()
        for strategy in self.strategies.values():
            if strategy.inited and strategy.trading:
                self.call_strategy_func(strategy, strategy.on_timer)
                if strategy.check_retraining_schedule(now):
                    self.write_log(
                        f"Retraining triggered for '{strategy.strategy_name}'.", INFO
                    )
                    self.init_executor.submit(self.run_retraining, strategy)

    def run_retraining(self, strategy: StrategyTemplate) -> None:
        """Executes the model retraining process for a strategy in a background thread."""
        try:
            strategy.retrain_model(self.latest_factor_memories)
        except Exception as e:
            self.write_log(
                f"Error retraining model for {strategy.strategy_name}: {e}",
                ERROR,
                strategy,
            )

    def stop_all_strategies(self) -> None:
        """Stops all running strategies."""
        for strategy in self.strategies.values():
            if strategy.trading:
                self.call_strategy_func(strategy, strategy.on_stop)

    def save_all_strategy_settings(self) -> None:
        """Saves the configurations of all strategies to the definitions file."""
        settings_to_save = [s.to_setting() for s in self.strategies.values()]
        save_json(str(self.definitions_filename), settings_to_save)
        self.write_log(f"Saved settings for {len(settings_to_save)} strategies.", INFO)

    def save_all_strategy_runtime_data(self) -> None:
        """Saves the runtime state of all strategies."""
        all_data = {name: s.get_data() for name, s in self.strategies.items()}
        # Save this to a dedicated runtime state file
        runtime_file = self.definitions_filename.with_name("strategy_runtime_data.json")
        save_json(str(runtime_file), all_data)
        self.write_log(f"Saved runtime data for {len(all_data)} strategies.", INFO)

    def load_all_strategy_runtime_data(self) -> None:
        """Loads the runtime state for all initialized strategies."""
        runtime_file = self.definitions_filename.with_name("strategy_runtime_data.json")
        if not runtime_file.exists():
            return

        all_data = load_json(str(runtime_file))
        for name, data in all_data.items():
            if name in self.strategies:
                self.strategies[name].load_data(data)
        self.write_log("Loaded runtime data for strategies.", INFO)

    def call_strategy_func(
        self, strategy: StrategyTemplate, func: Callable, *args, **kwargs
    ) -> Any:
        """Safely calls a method on a strategy instance."""
        try:
            return func(*args, **kwargs)
        except Exception:
            self.write_log(
                f"Error calling {func.__name__} on {strategy.strategy_name}:\n{traceback.format_exc()}",
                ERROR,
            )

    def write_log(
        self, msg: str, strategy: StrategyTemplate | None = None, level: int = INFO
    ) -> None:
        """Writes a log message, optionally prefixing it with the strategy name."""
        prefix = f"[{strategy.strategy_name}] " if strategy else ""
        log_entry = LogData(
            msg=f"{prefix}{msg}", gateway_name=self.engine_name, level=level
        )
        self.event_engine.put(Event(EVENT_LOG, log_entry))

    def get_current_datetime(self) -> datetime:
        """Returns the current UTC datetime."""
        return datetime.now(timezone.utc)

    def put_strategy_update_event(self, strategy: StrategyTemplate) -> None:
        """Emits an event to notify the UI or other components of a strategy update."""
        # This is a placeholder for integration with a UI event if needed
        pass

    # --- Placeholders for methods delegated to ExecutionAgent ---
    def send_order(
        self,
        strategy_name: str,
        req: OrderRequest,
        lock: bool = False,
        net: bool = False,
    ) -> list[str]:
        req.reference = strategy_name  # Ensure reference is set
        # In a real system, you might pass more context to the agent
        return self.execution_agent.send_order(req, lock, net)

    def cancel_order(self, req: CancelRequest) -> None:
        self.execution_agent.cancel_order(req)
