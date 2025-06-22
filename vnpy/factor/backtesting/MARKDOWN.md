# Factor Backtesting Component

This document provides an overview of the Factor Backtesting component, its core classes, and how to use them for evaluating quantitative factors.

---

## 1. Purpose

The Factor Backtesting component is a powerful suite of tools designed for the rigorous evaluation of quantitative trading factors. It provides a complete workflow, from data loading and factor calculation to performance analysis and parameter optimization.

The primary goals of this component are:
- To provide a robust engine for running single-factor backtests on historical data.
- To offer a grid-search-based optimizer for tuning factor parameters.
- To generate comprehensive and insightful performance reports, including quantile analysis and long-short portfolio statistics.
- To ensure efficient and scalable computation using modern libraries like Polars and Dask.

---

## 2. Interface

The component is composed of several key classes that work together. The main user-facing classes are `BacktestEngine` and `FactorOptimizer`.

### `BacktestEngine`
The central orchestrator for running a single factor backtest. It manages data loading, coordinates the `FactorCalculator` and `FactorAnalyser`, and produces a final report.

**Key Public Method:**
- `run_single_factor_backtest(factor_definition, start_datetime, end_datetime, vt_symbols, ...)`: Executes a complete backtest for a given factor definition over a specified period.

### `FactorOptimizer`
Performs parameter optimization for a factor using a grid search methodology. It splits data into training and testing sets, finds the best parameters on the training set, and validates them on the out-of-sample test set.

**Key Public Method:**
- `optimize_factor(factor_definition_template, parameter_grid, start_datetime, ...)`: Runs the optimization process and returns the best parameters and a final evaluation report.

### `FactorCalculator` (Internal Engine)
Responsible for the efficient, Dask-powered calculation of factor values based on a dependency graph. It is used internally by the `BacktestEngine` and `FactorOptimizer`.

### `FactorAnalyser` (Internal Engine)
Performs in-depth analysis of factor performance, including quantile analysis and long-short portfolio construction. It uses `quantstats` to generate detailed metrics and reports. It is used internally by the other classes.

---

## 3. Usage Examples

### Example 1: Running a Single Factor Backtest

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

### Example 2: Optimizing a Factor's Parameters

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

---

## 4. Dependencies

### Internal Dependencies
- `vnpy.factor.template`: For the base `FactorTemplate`.
- `vnpy.factor.base`: For core constants and types.
- `vnpy.trader.constant`: For data interval definitions.
- `vnpy.trader.setting`: For default settings.

### External Dependencies
- `polars`: For high-performance data manipulation.
- `numpy`: For numerical operations.
- `dask`: For parallel execution of the factor calculation graph.
- `quantstats`: For generating performance metrics and reports.
- `loguru`: For logging.