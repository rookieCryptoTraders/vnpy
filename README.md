# vn.py Trading Framework

## 1. Overview

**vn.py** is a comprehensive, modular, event-driven quantitative trading framework written in Python. It provides a complete suite of tools for strategy development, backtesting, and live trading across various financial markets, including stocks, futures, and cryptocurrencies. The framework is designed with a decoupled architecture, allowing for high extensibility and easy integration of custom modules.

## 2. Project Structure

```
crypto_vnpy/
│
├── vnpy/                # Core trading engine and modules
├── vnpy_clickhouse/     # ClickHouse database integration
├── vnpy_datafeed/       # Market data feed connectors
├── tests/               # Unit and integration tests
├── venv/                # Python virtual environment (recommended)
└── README.md            # Project documentation (this file)
```

## 3. Core Architecture

The heart of vn.py is its event-driven architecture, which ensures that different components can communicate asynchronously without being tightly linked. This promotes modularity and scalability.

### 3.1. `EventEngine`
- **Location**: `vnpy/event/engine.py`
- **Role**: The central message bus of the entire platform. All modules communicate by putting `Event` objects into the `EventEngine` and listening for events they are interested in. This decouples all components, from data gateways to strategy execution.

### 3.2. `MainEngine`
- **Location**: `vnpy/trader/engine.py`
- **Role**: The central orchestrator of the framework. It is responsible for:
    - Initializing and managing the lifecycle of all other engines (e.g., `StrategyEngine`, `PortfolioEngine`).
    - Loading and managing gateway connections to external exchanges.
    - Providing a unified API for core trading actions like sending orders, subscribing to market data, and querying account information.

### 3.3. `Gateways`
- **Location**: `vnpy/gateway/`
- **Role**: Gateways are the bridges to external exchanges (e.g., Binance, OKX). Each gateway implements a standardized interface (`BaseGateway`) for:
    - Connecting to the exchange's API (REST and Websocket).
    - Subscribing to real-time market data (ticks, bars).
    - Sending, updating, and canceling orders.
    - Querying account balances, positions, and historical data.

## 4. Application Modules

vn.py includes several pre-built application modules that provide essential trading functionalities. These are managed by the `MainEngine`.

### 4.1. `PortfolioManager`
- **Location**: `vnpy/app/portfolio_manager/engine.py`
- **Role**: A passive, event-driven engine that tracks trading performance. It listens for trade and order events and automatically calculates PnL, positions, and key statistics for different portfolios. Portfolios are identified by a unique `reference` tag attached to each order, making it easy to track the performance of individual strategies.

### 4.2. `DataRecorder`
- **Location**: `vnpy/app/data_recorder/engine.py`
- **Role**: Responsible for capturing and storing market data. It can record tick-by-tick data, generate candlestick bars (`BarData`) from ticks, and save them to a database (e.g., ClickHouse) for historical analysis and backtesting.

### 4.3. `RiskManager`
- **Location**: `vnpy/app/risk_manager/engine.py`
- **Role**: Provides pre-trade risk control by intercepting all outgoing orders. It can enforce rules such as:
    - Limiting order flow rate.
    - Capping the size of a single order.
    - Setting a maximum for total traded volume.
    - Preventing potential self-trading.

## 5. Factor Engine Module

The Factor Engine is a powerful, data-centric module responsible for all feature engineering and data processing. It is designed for performance, using **Dask** for parallel computation and **Polars** for high-speed data manipulation.

- **Location**: `vnpy/factor/`
- **Key Components**:
    - `FactorEngine` (`engine.py`): The core of the calculation layer. It listens for raw market data (bars) and orchestrates the computation of all configured factors. It uses a Dask computational graph to manage dependencies and execute calculations efficiently.
    - `FactorTemplate` (`template.py`): The abstract base class for all factors. Any new factor must inherit from this class and implement the `calculate` method. It supports dependency injection, allowing factors to be built upon one another.
    - `FactorMemory` (`memory.py`): A dedicated memory management class for persisting and retrieving historical factor data, using Apache Arrow for efficient disk I/O.

- **Data Flow**:
    1. The `FactorEngine` receives a `BarData` event.
    2. It updates its internal bar memory (`memory_bar`).
    3. It triggers the Dask computational graph, where each `FactorTemplate` instance calculates its value based on either raw bar data or the output of its dependency factors.
    4. The calculated factor values are stored in their respective `FactorMemory` instances.
    5. Finally, the `FactorEngine` broadcasts an `EVENT_FACTOR` containing the latest calculated factor data for all strategies to consume.

## 6. Strategy Engine Module

The Strategy Engine is the decision-making layer of the framework. It hosts and executes the trading logic.

- **Location**: `vnpy/strategy/`
- **Key Components**:
    - `StrategyEngine` (`engine.py`): Manages the lifecycle of all strategy instances. It loads strategy configurations from a JSON file, feeds them the latest factor data from `EVENT_FACTOR` events, and sends any generated orders to the `MainEngine` for execution.
    - `StrategyTemplate` (`template.py`): The abstract base class for all model-driven strategies. It defines a standard pipeline that a strategy must implement:
        1. `_transform_latest_factors`: Prepare the feature matrix from incoming factor data.
        2. `predict_from_model`: Use a loaded model (e.g., from scikit-learn) to generate predictions.
        3. `generate_signals_from_prediction`: Convert predictions into `OrderRequest` objects.
        4. `prepare_training_data`: A method for preparing features (X) and labels (y) for model retraining.
    - `StrategyParameters` (`template.py`): A helper class that holds a strategy's configuration, ensuring that all tunable settings (like symbols, model paths, and trading thresholds) are managed via parameters and not hardcoded.

- **Integration with Portfolio Manager**:
    - When a `StrategyTemplate` creates an `OrderRequest`, it **must** set the `reference` field to its own `strategy_name`.
    - This `reference` tag is carried through the entire order lifecycle (order -> trade) and is used by the `PortfolioEngine` to attribute all PnL and performance metrics back to the originating strategy. This is the critical link that enables multi-strategy performance tracking.

## 8. Additional Documentation

### 8.1. Portfolio Manager Documentation

#### Overview
The Portfolio Manager is a VeighNa application designed to manage and track trading portfolios, calculating PnL (Profit and Loss) across multiple contracts and providing real-time portfolio monitoring.

#### Core Components

##### PortfolioEngine (`engine.py`)
The main engine handling portfolio management operations.

###### Key Features
- Real-time portfolio tracking
- Trade processing and PnL calculation
- Automatic market data subscription
- Data persistence for positions and orders
- Event-driven architecture integration

###### Key Methods
- `process_trade_event`: Handles incoming trade events
- `process_timer_event`: Regular portfolio updates and calculations
- `save_data/load_data`: Position data persistence
- `save_order/load_order`: Order reference data persistence
- `set_timer_interval`: Configure update frequency

##### ContractResult (`base.py`)
Manages individual contract positions and calculations.

###### Properties
- `reference`: Portfolio reference identifier
- `vt_symbol`: Contract symbol
- `open_pos`: Opening position
- `last_pos`: Current position
- `trading_pnl`: Realized PnL from trades
- `holding_pnl`: Unrealized PnL from positions
- `total_pnl`: Combined total PnL

###### Methods
- `update_trade`: Process new trades
- `calculate_pnl`: Calculate various PnL metrics
- `get_data`: Return current state as dictionary

##### PortfolioResult (`base.py`)
Aggregates results across multiple contracts within a portfolio.

###### Properties
- `reference`: Portfolio identifier
- `trading_pnl`: Total realized PnL
- `holding_pnl`: Total unrealized PnL
- `total_pnl`: Combined portfolio PnL

#### Usage Example

```python
# Initialize the engine
portfolio_engine = PortfolioEngine(main_engine, event_engine)

# Set update interval (in seconds)
portfolio_engine.set_timer_interval(5)

# Get portfolio results
portfolio = portfolio_engine.get_portfolio_result("PORTFOLIO_A")
print(f"Total PnL: {portfolio.total_pnl}")
```

### 8.2. Factor Backtesting Documentation

This document provides an overview of the Factor Backtesting component, its core classes, and how to use them for evaluating quantitative factors.

#### Purpose

The Factor Backtesting component is a powerful suite of tools designed for the rigorous evaluation of quantitative trading factors. It provides a complete workflow, from data loading and factor calculation to performance analysis and parameter optimization.

The primary goals of this component are:
- To provide a robust engine for running single-factor backtests on historical data.
- To offer a grid-search-based optimizer for tuning factor parameters.
- To generate comprehensive and insightful performance reports, including quantile analysis and long-short portfolio statistics.
- To ensure efficient and scalable computation using modern libraries like Polars and Dask.

#### Interface

The component is composed of several key classes that work together. The main user-facing classes are `BacktestEngine` and `FactorOptimizer`.

##### `BacktestEngine`
The central orchestrator for running a single factor backtest. It manages data loading, coordinates the `FactorCalculator` and `FactorAnalyser`, and produces a final report.

**Key Public Method:**
- `run_single_factor_backtest(factor_definition, start_datetime, end_datetime, vt_symbols, ...)`: Executes a complete backtest for a given factor definition over a specified period.

##### `FactorOptimizer`
Performs parameter optimization for a factor using a grid search methodology. It splits data into training and testing sets, finds the best parameters on the training set, and validates them on the out-of-sample test set.

**Key Public Method:**
- `optimize_factor(factor_definition_template, parameter_grid, start_datetime, ...)`: Runs the optimization process and returns the best parameters and a final evaluation report.

##### `FactorCalculator` (Internal Engine)
Responsible for the efficient, Dask-powered calculation of factor values based on a dependency graph. It is used internally by the `BacktestEngine` and `FactorOptimizer`.

##### `FactorAnalyser` (Internal Engine)
Performs in-depth analysis of factor performance, including quantile analysis and long-short portfolio construction. It uses `quantstats` to generate detailed metrics and reports. It is used internally by the other classes.

#### Usage Examples

##### Example 1: Running a Single Factor Backtest

This example shows how to set up and run a backtest for a simple Moving Average Crossover factor.

```python
from datetime import datetime
from vnpy.trader.constant import Interval
from vnpy.factor.backtesting import BacktestEngine

# 1. Initialize the backtest engine
engine = BacktestEngine()

# 2. Define the factor to be tested (e.g., as a dictionary)
factor_def = {
    "class_name": "MovingAverageCrossover",
    "params": {
        "fast_window": 10,
        "slow_window": 30
    }
}

# 3. Define the backtest parameters
start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31)
symbols = ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"]

# 4. Run the backtest
report_path = engine.run_single_factor_backtest(
    factor_definition=factor_def,
    start_datetime=start,
    end_datetime=end,
    vt_symbols_for_factor=symbols,
    data_interval=Interval.HOUR,
    report_filename_prefix="mac_crossover_test"
)

if report_path:
    print(f"Backtest complete. Report saved to: {report_path}")
```

##### Example 2: Optimizing a Factor's Parameters

This example demonstrates how to find the optimal `fast_window` and `slow_window` for the same factor.

```python
from datetime import datetime
from vnpy.trader.constant import Interval
from vnpy.factor.backtesting import BacktestEngine, FactorOptimizer

# 1. Initialize the backtest engine and optimizer
engine = BacktestEngine()
optimizer = FactorOptimizer(backtest_engine=engine)

# 2. Define the factor template and the parameter grid for the search
factor_template = {
    "class_name": "MovingAverageCrossover",
    "params": {} # Parameters will be filled by the optimizer
}

param_grid = {
    "fast_window": [5, 10, 15],
    "slow_window": [20, 30, 40]
}

# 3. Run the optimization
best_params, search_results, report_path = optimizer.optimize_factor(
    factor_definition_template=factor_template,
    parameter_grid=param_grid,
    start_datetime=datetime(2023, 1, 1),
    end_datetime=datetime(2023, 12, 31),
    vt_symbols=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
    data_interval=Interval.HOUR
)

print(f"Best parameters found: {best_params}")
print(f"Final report for best params on test set: {report_path}")
```

### 8.3. Twitter Gateway Documentation

#### Purpose
The `TwitterGateway` is designed to receive text content from Twitter in real-time. It streams tweets based on specified keywords and integrates with the event engine to process the incoming data.

#### Interface
##### Constructor
```python
TwitterGateway(event_engine: EventEngine, gateway_name: str)
```
- **event_engine**: The event engine instance.
- **gateway_name**: The name of the gateway.

##### Methods
###### `connect`
```python
def connect(self, setting: dict) -> None
```
Connects to the Twitter API and starts streaming tweets based on the provided keywords.
- **setting**: A dictionary containing connection settings, including `keywords` to track.

###### `subscribe`
```python
def subscribe(self, req: SubscribeRequest) -> None
```
This method is not applicable for the Twitter gateway.

###### `close`
```python
def close(self) -> None
```
Closes the Twitter stream and disconnects the gateway.

###### `write_log`
```python
def write_log(self, msg: str, level=logging.INFO) -> None
```
Writes log messages.
- **msg**: The log message.
- **level**: The logging level.

#### Usage Examples
##### Example 1: Initialize and Connect
```python
from vnpy.gateway.twitter import TwitterGateway
from vnpy.event import EventEngine

# Initialize event engine
event_engine = EventEngine()

# Create TwitterGateway instance
twitter_gateway = TwitterGateway(event_engine, "Twitter")

# Connect to Twitter API
settings = {
    "keywords": ["crypto", "trading"],
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "access_token": "your_access_token",
    "access_token_secret": "your_access_token_secret"
}
twitter_gateway.connect(settings)
```

### 8.4. Strategy Engine & Template Workflow

This document outlines the data flow and model lifecycle within the `StrategyEngine` and `StrategyTemplate` framework.

#### I. Data Flow: From Factor Event to Order Placement

1.  **Factor Calculation & Event**:
    *   External processes or a dedicated factor engine calculates factor data.
    *   Upon completion, an `EVENT_FACTOR` (or a similarly purposed event, e.g., `EVENT_FACTOR_UPDATE`) is emitted.
    *   This event carries a dictionary: `Dict[str, FactorMemory]`, where keys are unique factor identifiers (e.g., "factor.1m.EMA_Close_12") and values are `FactorMemory` instances containing the data for that factor across multiple symbols.

2.  **StrategyEngine: Event Reception & Dispatch**:
    *   The `StrategyEngine` (specifically, the modified `BaseStrategyEngine` in `vnpy/strategy/engine.py`) listens for the `EVENT_FACTOR`.
    *   Upon receiving the event, its `process_factor_event` method is triggered.
    *   The engine caches these `FactorMemory` instances (e.g., in `self.latest_factor_memories`) for potential use in retraining.
    *   It then iterates through all active (inited and trading) strategy instances.
    *   For each strategy instance, it calls the `strategy.on_factor(latest_factor_memories)` method, passing the complete dictionary of `FactorMemory` objects.

3.  **StrategyTemplate: Factor Processing & Prediction**:
    *   The `on_factor` method within the concrete strategy (inheriting from `StrategyTemplate` in `vnpy/strategy/template.py`) is invoked.
    *   **Data Fetching**: The strategy uses its `required_factor_keys` list to identify the factors it needs. For each required factor, it calls `factor_memory.get_latest_rows(1)` on the corresponding `FactorMemory` object (from the dictionary received from the engine) to get the latest data snapshot as a Polars DataFrame. This results in a `Dict[str, pl.DataFrame]`.
    *   **Transformation**: It calls its (abstract, implemented by concrete strategy) `_transform_latest_factors(latest_polars_data_map)` method. This method takes the dictionary of Polars DataFrames, pivots/transforms/combines them into a single Polars DataFrame where rows typically represent `vt_symbol` and columns represent factor features.
    *   **Prediction**:
        *   The transformed Polars DataFrame is converted to a Pandas DataFrame.
        *   It calls its (abstract, implemented by concrete strategy) `predict_from_model(pandas_feature_df)` method. This method uses the strategy's loaded model (`self.model`) to generate raw predictions (e.g., probabilities, class labels).
    *   **Signal Generation**: It calls its (abstract, implemented by concrete strategy) `generate_signals_from_prediction(model_output, pandas_feature_df)` method. This method converts the raw model predictions into actionable trading signals, typically a list of `OrderRequest` objects.

4.  **StrategyEngine: Order Execution**:
    *   The `StrategyEngine` receives the list of `OrderRequest` objects returned by the strategy's `on_factor` method.
    *   For each `OrderRequest`, the engine calls `self.send_order(strategy_name, order_request)`, which forwards the request to the appropriate trading gateway via `MainEngine.send_order()` (or the `ExecutionAgent`).

#### II. Model Retraining Loop

1.  **Scheduled Trigger (StrategyEngine)**:
    *   The `StrategyEngine` has a timer event (`_process_timer_event`) that fires periodically (e.g., every minute or hour).
    *   For each active strategy, it calls `strategy.check_retraining_schedule(current_datetime)`.
    *   The `check_retraining_schedule` method in `StrategyTemplate` checks if retraining is due based on `strategy.retraining_config["frequency_days"]` and `strategy.last_retrain_time`. It also considers if a model needs initial training (e.g. `self.model` is `None` and `model_load_path` was not successful).

2.  **Retraining Initiation (StrategyEngine)**:
    *   If `check_retraining_schedule` returns `True`, the `StrategyEngine` queues a call to `strategy.retrain_model()` in a separate thread (using `self.init_executor.submit(self._train_strategy_model_thread, strategy.strategy_name)`, where `_train_strategy_model_thread` now directly calls `strategy.retrain_model()`).

3.  **Data Preparation & Training (StrategyTemplate)**:
    *   The `retrain_model` method within the `StrategyTemplate` is executed.
    *   **Fetch Historical Data**: It calls its helper `_fetch_historical_training_factors(self.strategy_engine.latest_factor_memories)`. This helper iterates through the strategy's `required_factor_keys`, accesses the corresponding `FactorMemory` objects from the engine's cache, and calls `factor_memory.get_data()` on each to get full historical data as Polars DataFrames (`Dict[str, pl.DataFrame]`).
    *   **Prepare Training Data**: It calls its (abstract, implemented by concrete strategy) `prepare_training_data(historical_polars_data_map)` method. This method takes the dictionary of historical Polars DataFrames and transforms them into features (Pandas DataFrame `X`) and labels (Pandas Series `y`) suitable for model training. This step includes any necessary data cleaning, feature engineering from historical data, and label generation.
    *   **Model Fitting**: If valid training data (`X`, `y`) is produced, it trains its model: `self.model.fit(X, y)`.
    *   **Save Model**: After successful training, it calls `self.save_model()` to persist the updated model to `self.model_save_path`.
    *   **Update Timestamp**: It updates `self.last_retrain_time` to the current datetime.

4.  **Ongoing Predictions**:
    *   Once the model is retrained (i.e., `self.model` is updated), subsequent calls to `on_factor` will use the newly retrained model for predictions.

This workflow enables strategies to react to new factor data for generating trading signals and incorporates a managed lifecycle for their predictive models, including automated retraining.