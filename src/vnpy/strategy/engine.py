"""
Defines the StrategyEngine, the core component for managing and executing
trading strategies within the vn.py framework.
"""

import importlib
import inspect
import pkgutil
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from logging import ERROR, INFO, WARNING
from pathlib import Path
from typing import Any

# --- Strategy & Portfolio Specific Imports ---
from vnpy.app.portfolio_manager.engine import PortfolioEngine

# --- VnTrader Core Imports ---
from vnpy.event import Event, EventEngine
from vnpy.factor.memory import FactorMemory
from vnpy.strategy.setting import (
    STRATEGY_ENGINE_OPERATIONAL_PARAMS,
    get_strategy_instance_definitions_filename,
)
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
    ContractData,
    LogData,
    OrderData,
    OrderRequest,
    TickData,
    TradeData,
)
from vnpy.trader.utility import load_json, save_json

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

        # Corrected: Ensure the filename is a Path object for consistency
        self.definitions_filepath: Path = Path(
            get_strategy_instance_definitions_filename()
        )

        # Core state
        self.strategy_classes: dict[str, type[StrategyTemplate]] = {}
        self.strategies: dict[str, StrategyTemplate] = {}

        # PortfolioEngine integration
        self.portfolio_engine = PortfolioEngine(main_engine, event_engine)

        # Factor data cache
        self.latest_factor_memories: dict[str, FactorMemory] = {}
        self.factor_update_time: datetime | None = None

        self.write_log(
            f"StrategyEngine initialized. Definitions: {self.definitions_filepath}",
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
        self.save_all_strategy_settings()
        self.save_all_strategy_runtime_data()
        self.unregister_event()
        self.portfolio_engine.close()
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
        """
        Discovers and loads all StrategyTemplate subclasses from the code module path.
        This version robustly handles dot-notation paths.
        """
        self.write_log(
            f"Loading strategy classes from module path: {self.strategy_code_module_path}",
            level=INFO,
        )
        self.strategy_classes.clear()

        try:
            module = importlib.import_module(self.strategy_code_module_path)

            # Walk the package to find all modules
            for _, modname, _ in pkgutil.walk_packages(
                module.__path__, module.__name__ + "."
            ):
                try:
                    imported_module = importlib.import_module(modname)
                    # Find all StrategyTemplate subclasses in the module
                    for item_name in dir(imported_module):
                        item_value = getattr(imported_module, item_name)
                        if (
                            inspect.isclass(item_value)
                            and issubclass(item_value, StrategyTemplate)
                            and item_value is not StrategyTemplate
                            and not inspect.isabstract(item_value)
                        ):
                            self.strategy_classes[item_value.__name__] = item_value
                except Exception as e:
                    self.write_log(
                        f"Failed to import or inspect module {modname}: {e}", WARNING
                    )
        except ImportError as e:
            self.write_log(f"Failed to import base strategy module: {e}", ERROR)

        self.write_log(f"Loaded {len(self.strategy_classes)} strategy classes.", INFO)

    def init_strategies_from_configs(self) -> None:
        """Initializes strategy instances based on the definitions JSON file."""
        self.write_log(
            f"Initializing strategies from {self.definitions_filepath}...", level=INFO
        )
        if not self.definitions_filepath.exists():
            self.write_log("Definitions file not found. No strategies loaded.", WARNING)
            return

        strategy_settings = load_json(str(self.definitions_filepath))
        for setting in strategy_settings:
            class_name = setting.get("class_name")
            strategy_name = setting.get("strategy_name")

            # Use the new add_strategy method to create the instance
            self.add_strategy(class_name, strategy_name, setting)

    def add_strategy(self, class_name: str, strategy_name: str, setting: dict) -> bool:
        """
        Manually adds a new strategy instance to the engine.

        Args:
            class_name (str): The class name of the strategy to add.
            strategy_name (str): The unique name for this strategy instance.
            setting (dict): The configuration dictionary for this instance.

        Returns:
            bool: True if the strategy was added successfully, False otherwise.
        """
        if class_name not in self.strategy_classes:
            self.write_log(
                f"Failed to add strategy: Class '{class_name}' not found.", ERROR
            )
            return False

        if strategy_name in self.strategies:
            self.write_log(
                f"Failed to add strategy: Instance name '{strategy_name}' already exists.",
                ERROR,
            )
            return False

        # Ensure the setting dict has the correct identifiers
        setting["class_name"] = class_name
        setting["strategy_name"] = strategy_name

        strategy_class = self.strategy_classes[class_name]
        strategy = strategy_class(self, setting)
        self.strategies[strategy_name] = strategy

        # Initialize the strategy immediately after adding
        self.call_strategy_func(strategy, strategy.on_init)
        self.write_log(f"Strategy '{strategy_name}' added and initialized.", INFO)
        return True

    def process_factor_event(self, event: Event) -> None:
        """Processes factor updates and forwards them to all active strategies."""
        self.latest_factor_memories = event.data
        for strategy in self.strategies.values():
            if strategy.inited and strategy.trading:
                # Generate orders based on new factors
                orders = self.call_strategy_func(
                    strategy, strategy.on_factor_update, self.latest_factor_memories
                )
                # Send any generated orders
                if orders:
                    for req in orders:
                        self.send_order(strategy.strategy_name, req)

    def process_timer_event(self, event: Event) -> None:
        """Processes timer events for periodic tasks like model retraining."""
        now = self.get_current_datetime()
        for strategy in self.strategies.values():
            if strategy.inited and strategy.trading:
                self.call_strategy_func(strategy, strategy.on_timer)
                if strategy.check_retraining_schedule(now):
                    self.write_log(
                        f"Retraining triggered for '{strategy.strategy_name}'. Running synchronously.",
                        INFO,
                    )
                    # Changed to a direct, synchronous call
                    self.run_retraining(strategy)

    def run_retraining(self, strategy: StrategyTemplate) -> None:
        """Executes the model retraining process for a strategy."""
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
        for strategy_name in list(self.strategies.keys()):
            self.stop_strategy(strategy_name)

    def start_strategy(self, strategy_name: str) -> None:
        """Starts a specific strategy."""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            self.write_log(
                f"Cannot start strategy '{strategy_name}': Not found.", ERROR
            )
            return

        if strategy.trading:
            self.write_log(f"Strategy '{strategy_name}' is already running.", WARNING)
            return

        self.call_strategy_func(strategy, strategy.on_start)

    def stop_strategy(self, strategy_name: str) -> None:
        """Stops a specific strategy."""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            self.write_log(f"Cannot stop strategy '{strategy_name}': Not found.", ERROR)
            return

        if not strategy.trading:
            return

        self.call_strategy_func(strategy, strategy.on_stop)

    def save_all_strategy_settings(self) -> None:
        """Saves the configurations of all strategies to the definitions file."""
        settings_to_save = [s.to_setting() for s in self.strategies.values()]
        save_json(str(self.definitions_filepath), settings_to_save)
        self.write_log(f"Saved settings for {len(settings_to_save)} strategies.", INFO)

    def save_all_strategy_runtime_data(self) -> None:
        """Saves the runtime state of all strategies."""
        all_data = {name: s.get_data() for name, s in self.strategies.items()}
        runtime_file = self.definitions_filepath.with_name("strategy_runtime_data.json")
        save_json(str(runtime_file), all_data)
        self.write_log(f"Saved runtime data for {len(all_data)} strategies.", INFO)

    def load_all_strategy_runtime_data(self) -> None:
        """Loads the runtime state for all initialized strategies."""
        runtime_file = self.definitions_filepath.with_name("strategy_runtime_data.json")
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
        pass  # Placeholder for UI integration

    def send_order(
        self,
        strategy_name: str,
        req: OrderRequest,
        lock: bool = False,
        net: bool = False,
    ) -> list[str]:
        """Sends an order request to the main engine."""
        req.reference = strategy_name
        return self.main_engine.send_order(req, self.execution_gateway_name)

    def cancel_order(self, req: CancelRequest) -> None:
        self.main_engine.cancel_order(req, self.execution_gateway_name)

    def process_order_event(self, event: Event) -> None:
        """Process order event and forward to portfolio engine."""
        order: OrderData = event.data
        if order.reference in self.strategies:
            self.call_strategy_func(self.strategies[order.reference], "on_order", order)
        self.portfolio_engine.process_order_event(event)

    def process_trade_event(self, event: Event) -> None:
        """Process trade event and forward to portfolio engine."""
        trade: TradeData = event.data
        if trade.reference in self.strategies:
            self.call_strategy_func(self.strategies[trade.reference], "on_trade", trade)
        self.portfolio_engine.process_trade_event(event)

    def get_tick(self, vt_symbol: str) -> TickData | None:
        return self.main_engine.get_tick(vt_symbol)

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        return self.main_engine.get_contract(vt_symbol)
