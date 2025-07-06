# Factor Engine Framework

This document provides a comprehensive overview of the factor engine framework, a powerful system for creating, calculating, backtesting, and optimizing quantitative factors.

## 1. Core Concepts

The framework is built around several key concepts:

*   **Factor (`FactorTemplate`)**: The fundamental unit of logic. A factor is a Python class that inherits from `FactorTemplate` and implements a `calculate` method. It can be a simple calculation (e.g., moving average) or a complex one that depends on other factors.

*   **Factor Engine (`FactorEngine`)**: The live engine responsible for orchestrating real-time factor calculation. It manages data streams, triggers calculations, and stores results.

*   **Backtesting Engine (`BacktestEngine`)**: An offline engine that simulates the factor's performance on historical data. It coordinates data loading, calculation, and analysis.

*   **Factor Memory (`FactorMemory`)**: A persistent, memory-mapped storage for historical factor values, using Arrow IPC files for high performance. Each unique factor instance has its own `FactorMemory`.

*   **`factor_key` vs. `factor_name`**:
    *   `factor_key`: A globally unique identifier (GUID) for a specific factor instance, generated from its class, frequency, and parameters (e.g., `factor.emafactor.1m@period_12`). It's crucial for data storage and dependency resolution.
    *   `factor_name`: A human-readable alias or nickname, primarily used to clarify the role of a dependency within a parent factor's configuration.

## 2. Architecture and Data Flow

The system is designed with a clear separation of concerns, enabling both live trading and offline research.

### Live Calculation (`FactorEngine`)

1.  **Initialization**:
    *   The `FactorEngine` loads factor configurations from `factor_definition_setting.json`.
    *   It creates `FactorTemplate` instances for each configured factor.
    *   It resolves the dependency graph between factors, creating a flattened list of all unique factor instances.
    *   It initializes `FactorMemory` for each unique factor to store historical values.
    *   It builds a Dask computational graph to manage the execution order based on dependencies.

2.  **Execution**:
    *   The engine listens for incoming bar data (`EVENT_BAR`).
    *   On receiving a complete bar slice for all symbols, it updates its internal OHLCV `memory_bar`.
    *   It triggers the Dask graph, which executes the `calculate` method for each factor in the correct order.
    *   The output of each factor's calculation is a Polars DataFrame, which is then passed to its `FactorMemory` instance for atomic updating and persistence.
    *   The latest calculated factor values are broadcast via `EVENT_FACTOR`.

### Backtesting & Optimization

The backtesting and optimization workflow uses a similar set of components but in an offline context.

1.  **`BacktestEngine`**:
    *   Loads a complete historical dataset for a specified period.
    *   Initializes the target factor and its dependencies using `FactorInitializer`.
    *   Uses `FactorCalculator` to compute the factor values over the entire historical dataset in one batch.
    *   Passes the calculated factor values and market data to `FactorAnalyser`.

2.  **`FactorAnalyser`**:
    *   Performs a detailed performance analysis, including:
        *   Quantile analysis.
        *   Long-short portfolio return calculation.
        *   Performance metrics (Sharpe ratio, etc.) using `quantstats`.
    *   Generates comprehensive HTML and JSON reports.

3.  **`FactorOptimizer`**:
    *   Automates the process of finding the best parameters for a factor.
    *   Splits the historical data into training and testing sets.
    *   Uses either Grid Search or Bayesian Optimization to iterate through parameter combinations on the training set.
    *   The objective function for each trial is typically the Sharpe ratio of the long-short portfolio, calculated by the `BacktestEngine` and `FactorAnalyser`.
    *   After finding the best parameters, it runs a final evaluation on the out-of-sample test set and generates a report.

## 3. Directory Structure

The factor-related modules and files are organized as follows:

*   **`vnpy/vnpy/factor/`**: The root directory for the factor engine.
    *   `engine.py`: Contains the live `FactorEngine`.
    *   `template.py`: Defines the `FactorTemplate` abstract base class.
    *   `memory.py`: Contains the `FactorMemory` class for data persistence.
    *   `base.py`: Basic enumerations like `FactorMode`.
    *   `setting.py`: Manages loading of settings and defines paths.
    *   `exceptions.py`: Custom exceptions for the framework.
    *   **`backtesting/`**:
        *   `backtesting.py`: The `BacktestEngine`.
        *   `factor_calculator.py`: The `FactorCalculator` for batch computations.
        *   `factor_analyzer.py`: The `FactorAnalyser` for performance reporting.
        *   `factor_initializer.py`: Handles the creation of factor instances from configurations.
        *   `optimizer.py`: The `FactorOptimizer`.
    *   **`factors/`**: Where user-defined and generated factors reside.
        *   `__init__.py`: Makes factor classes available for import.
        *   `my_factor_lib.py`: Example of custom-written factors (`EMAFactor`, `MACDFactor`).
        *   **`ta_lib/`**: Sub-package for factors automatically generated from the `polars-talib` library.
    *   **`utils/`**:
        *   `factor_utils.py`: Helper functions for loading/saving settings and initializing factors.
        *   `factor_generator.py`: A script to auto-generate factor classes from `polars-talib`.
        *   `setting_populator.py`: A script to auto-generate a default `factor_definition_setting.json` file.
    *   **`docs/`**: Documentation files.
        *   `factor_name.md`: Explains `factor_key` vs. `factor_name`.
        *   `factor_params.md`: Guide to managing factor parameters.
        *   `optimization.md`: Details the optimization workflow.
    *   **Configuration Files**:
        *   `factor_settings.json`: General settings for the factor engine (e.g., module name, memory lengths).
        *   `factor_definition_setting.json`: A **list** of configurations for each top-level factor the engine should load.

## 4. How to Define a Factor

1.  **Inherit from `FactorTemplate`**: Create a new class that inherits from `vnpy.factor.template.FactorTemplate`.

2.  **Set `factor_name`**: Assign a unique, lowercase name to your factor class.

3.  **Implement `__init__`**:
    *   Call `super().__init__(setting, **kwargs)`.
    *   Define the parameters your factor needs. You can access them via `self.params`.
    *   If your factor depends on other factors, they will be available in `self.dependencies_factor` as a list of `FactorTemplate` instances.

4.  **Implement `get_output_schema`**:
    *   Define the structure of the DataFrame that your factor will output. It must include a datetime column and typically one column per symbol for the factor's value.

5.  **Implement `calculate`**:
    *   This is the core logic of your factor.
    *   It receives `input_data`, which is a dictionary containing the output DataFrames of its dependencies (or raw OHLCV data for leaf factors).
    *   It should return a single Polars DataFrame containing the calculated factor values, conforming to the schema you defined.

**Example: `EMAFactor`**
```python
import polars as pl
from ..template import FactorTemplate
from ..memory import FactorMemory

class EMAFactor(FactorTemplate):
    factor_name = "emafactor"

    def __init__(self, setting: dict | None = None, **kwargs):
        super().__init__(setting, **kwargs)
        self.period: int = int(self.params.period)

    def calculate(self, input_data: dict[str, pl.DataFrame], memory: FactorMemory) -> pl.DataFrame:
        df_close = input_data.get("close")
        # ... calculation logic using Polars ...
        # result_df = ...
        return result_df
```

## 5. Configuration

The engine is configured through two main JSON files located in your `.vntrader` directory.

### `factor_settings.json`

This file contains global settings for the factor engine.

```json
{
    "module_name": "vnpy.factor.factors",
    "datetime_col": "datetime",
    "max_memory_length_bar": 500,
    "max_memory_length_factor": 100,
    "error_threshold": 3
}
```

### `factor_definition_setting.json`

This file contains a **list** of dictionary configurations, where each dictionary defines a top-level factor to be loaded by the engine.

*   **`class_name`**: The Python class name of the factor.
*   **`factor_name`**: A human-readable nickname for this instance.
*   **`freq`**: The calculation frequency (e.g., "1m", "1d").
*   **`params`**: A dictionary of parameters for the factor's `__init__` method.
*   **`dependencies_factor`**: A list of configuration dictionaries for any factors this one depends on. This allows for creating nested dependency trees.

**Example Configuration for a MACD Factor:**

```json
[
    {
        "class_name": "MACDFactor",
        "factor_name": "my_macd_indicator",
        "freq": "1m",
        "params": {
            "signal_period": 9
        },
        "dependencies_factor": [
            {
                "class_name": "EMAFactor",
                "factor_name": "fast_ema",
                "params": { "period": 12 }
            },
            {
                "class_name": "EMAFactor",
                "factor_name": "slow_ema",
                "params": { "period": 26 }
            }
        ]
    }
]
```
This defines one top-level factor, `MACDFactor`, which itself depends on two `EMAFactor` instances, nicknamed "fast_ema" and "slow_ema". The engine will automatically resolve this entire tree.