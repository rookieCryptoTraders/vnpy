# Factor Parameter Optimization Workflow

This document outlines the process for optimizing parameters of a main factor and its nested dependencies. The framework provides two primary optimization methods: **Grid Search** and **Bayesian Optimization**. The key is to manage factor definitions, their unique identities (`factor_key`), and human-readable parameter paths (`factor_name` based nicknames) effectively through different stages.

## 1. Core Concepts Recap

Before diving into the workflow, let's briefly revisit key terms:

*   **Factor Definition Dictionary:** A JSON-like dictionary structure that describes a factor, its class, parameters, and its dependencies (which are also defined as dictionaries). This is the "blueprint."
*   **`FactorTemplate` Instance:** A live Python object created from a factor definition dictionary. It contains the logic for calculating factor values.
*   **`factor_key`:** A **globally unique identifier** for a `FactorTemplate` instance, automatically generated based on its class, calculation frequency, and specific parameter values (e.g., `"factor.emafactor.1m@period_12"`). This key is crucial for data caching (`FactorMemory`) and identifying unique computation nodes in Dask. **If a parameter that forms part of the `factor_key` changes, the `factor_key` itself must change, implying a new, distinct factor instance.**
*   **`factor_name` (Nickname):** A human-readable alias assigned to a factor instance, especially when it's a dependency of another factor. For example, a `MACDFactor` might have dependencies nicknamed `"fast_ema"` and `"slow_ema"`. This nickname is stored on the dependency instance's `factor_name` attribute.
*   **Parameter Path:** A dot-separated string used by the optimizer to refer to a specific parameter within a potentially nested factor structure (e.g., `"alpha"` for a main factor's parameter, or `"fast_ema.period"` for a parameter of a dependency nicknamed "fast\_ema").

## 2. The Optimization Workflow

The optimization process is orchestrated by the `FactorOptimizer`, which leverages the `BacktestEngine`, `FactorCalculator`, and `FactorAnalyser`.

### Step 1: Defining the Tunable Parameter Space

*   **Function:** `FactorTemplate.get_nested_params_for_optimizer()`
*   **Purpose:** To extract all tunable parameters from a factor instance (and its entire dependency tree) into a flat dictionary. The keys in this dictionary are the "parameter paths" that the optimizer will use.
*   **Path Generation:**
    *   Parameters of the main factor get simple keys: e.g., `"alpha"`.
    *   Parameters of direct dependencies use their nickname: e.g., if a dependency is nicknamed `"short_ema"`, its `period` parameter path becomes `"short_ema.period"`.
    *   Parameters of nested dependencies build on this: e.g., if `"short_ema"` itself has a dependency nicknamed `"smoother"`, its `window` parameter path becomes `"short_ema.smoother.window"`.
    *   **Nickname De-duplication:** If a factor has multiple direct dependencies with the *same* nickname (e.g., two dependencies both nicknamed "filter"), `get_nested_params_for_optimizer` appends a counter to make the path segment unique (e.g., `"filter_0.param"`, `"filter_1.param"`).

### Step 2: Setting Up and Running the Optimization

1.  **Initialization:** An instance of `FactorOptimizer` is created, taking a `BacktestEngine` instance.
2.  **Method Call:** You call either `optimizer.optimize_factor(...)` for grid search or `optimizer.optimize_factor_bayes(...)` for Bayesian optimization. You provide:
    *   `factor_definition_template`: A **dictionary** representing the base structure and initial parameters of the factor.
    *   `parameter_grid` (for grid search) or `parameter_bounds` (for Bayesian). These use the nickname-based paths from Step 1 as keys.
    *   Data parameters (`start_datetime`, `end_datetime`, `vt_symbols`, `data_interval`).
    *   A `test_size_ratio` to split the data.

3.  **Data Handling:**
    *   The `FactorOptimizer` calls `backtest_engine._load_bar_data_engine()` **once** to load the full historical market data.
    *   This data is then split into a **training set** and a **test set**. The optimization process runs exclusively on the training set.

### Step 3: The Core Evaluation Loop

The optimizer iterates through different parameter combinations. For each combination, it evaluates its performance on the **training data**.

1.  **`_calculate_factor_score`**: This internal method is the heart of the evaluation. For each parameter set, it performs the following steps:
    a.  **Prepare Factor Definition:**
        *   It creates a fresh, deep copy of the original `factor_definition_template`.
        *   It calls `apply_params_to_definition_dict()` to apply the current trial's parameters (e.g., `{"entry_signal_ema.period": 12}`) to the copied dictionary.
    b.  **Re-initialize Factor Tree:**
        *   `target_instance, flattened_factors = self.backtest_engine._init_and_flatten_factor(modified_def_dict, ...)`:
            *   The `BacktestEngine` takes the `modified_def_dict`.
            *   It creates **brand new `FactorTemplate` instances** for the target factor and all its dependencies.
            *   Crucially, each new instance calculates its `factor_key` based on its *current parameters*, ensuring `factor_key` integrity for caching and calculation.
    c.  **Calculate Factor Values:**
        *   A `FactorCalculator` instance computes the factor values for the newly instantiated factor graph over the **training data**.
    d.  **Calculate Objective Score:**
        *   A `FactorAnalyser` instance takes the calculated factor values and market data from the training set.
        *   It performs a long-short portfolio analysis and calculates the **Sharpe ratio**.
    e.  **Return Score:** The calculated Sharpe ratio is returned to the optimizer as the score for that parameter combination.

### Step 4: Finalizing Optimization

1.  **Best Parameters:** After trying all combinations (grid search) or completing its iterations (Bayesian), the optimizer identifies the `best_params` that yielded the highest score on the training data.
2.  **Evaluation on Test Set:** The `FactorOptimizer` then performs a final, crucial step:
    *   It creates a final factor definition using the `best_params`.
    *   It calls `self.backtest_engine.run_single_factor_backtest(...)`, providing this final definition and instructing it to use the **hold-out test data set**.
    *   This run generates a comprehensive report on the **out-of-sample performance** of the factor with the optimized parameters, giving a more realistic view of its potential effectiveness.

## Why the Dictionary-Update and Re-initialization Approach is Critical

This approach of modifying a *factor definition dictionary* and then *completely re-initializing* the factor tree for each trial is critical for several reasons:

1.  **`factor_key` Integrity:** It ensures that every `FactorTemplate` instance used in a trial has a `factor_key` that accurately reflects its parameters for that specific trial. This is vital because `factor_key` is used for data caching (`FactorMemory`) and uniquely identifying computation nodes.
2.  **Clean State per Trial:** Re-instantiating factors from their definition guarantees that each trial starts with a fresh set of objects, free from any state that might have been carried over from previous trials.
3.  **Robustness:** It correctly handles cases where changing a parameter might fundamentally alter the nature or identity of a factor or its dependencies.

By using `apply_params_to_definition_dict`, the optimizer can cleanly modify the "blueprint" (the dictionary) before each trial's "construction" (factor instantiation), ensuring that the `FactorCalculator` always works with a factor graph that is perfectly consistent with the parameters being evaluated.
