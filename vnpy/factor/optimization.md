# Factor Parameter Optimization Workflow Documentation

This document outlines the process for optimizing parameters of a main factor and its nested dependencies using `sklearn`'s `GridSearchCV`, integrated with the custom factor backtesting framework. The key is to manage factor definitions, their unique identities (`factor_key`), and human-readable parameter paths (`factor_name` based nicknames) effectively through different stages.

## 1. Core Concepts Recap

Before diving into the workflow, let's briefly revisit key terms:

* **Factor Definition Dictionary:** A JSON-like dictionary structure that describes a factor, its class, parameters, and its dependencies (which are also defined as dictionaries). This is the "blueprint."
* **`FactorTemplate` Instance:** A live Python object created from a factor definition dictionary. It contains the logic for calculating factor values.
* **`factor_key`:** A **globally unique identifier** for a `FactorTemplate` instance, automatically generated based on its class, calculation frequency, and specific parameter values (e.g., `"factor.emafactor.1m@period_12"`). This key is crucial for data caching (`FactorMemory`) and identifying unique computation nodes in Dask. **If a parameter that forms part of the `factor_key` changes, the `factor_key` itself must change, implying a new, distinct factor instance.**
* **`factor_name` (Nickname):** A human-readable alias assigned to a factor instance, especially when it's a dependency of another factor. For example, a `MACDFactor` might have dependencies nicknamed `"fast_ema"` and `"slow_ema"`. This nickname is stored on the dependency instance's `factor_name` attribute.
* **Parameter Path:** A dot-separated string used by the optimizer to refer to a specific parameter within a potentially nested factor structure (e.g., `"alpha"` for a main factor's parameter, or `"fast_ema.period"` for a parameter of a dependency nicknamed "fast\_ema").

## 2. The Optimization Workflow

The optimization process revolves around the `FactorOptimizer`, `FactorBacktestEstimator`, and the `BacktestEngine` (orchestrator), `FactorCalculator`, and `FactorAnalyser`.

### Step 1: Defining the Tunable Parameter Space

* **Function:** `FactorTemplate.get_nested_params_for_optimizer()`
* **Purpose:** To extract all tunable parameters from a factor instance (and its entire dependency tree) into a flat dictionary. The keys in this dictionary are the "parameter paths" that the optimizer will use.
* **Path Generation:**
    * Parameters of the main factor (on which the method is called) get simple keys: e.g., `"alpha"`.
    * Parameters of direct dependencies use their nickname: e.g., if a dependency is nicknamed `"short_ema"`, its `period` parameter path becomes `"short_ema.period"`.
    * Parameters of nested dependencies build on this: e.g., if `"short_ema"` itself has a dependency nicknamed `"smoother"`, its `window` parameter path becomes `"short_ema.smoother.window"`.
    * **Nickname De-duplication:** If a factor has multiple direct dependencies with the *same* nickname (e.g., two dependencies both nicknamed "filter"), `get_nested_params_for_optimizer` appends a counter to make the path segment unique (e.g., `"filter_0.param"`, `"filter_1.param"`).

* **Example:**
    Given a `ComplexFactor` instance that has:
    * An own parameter `smoothing_level`.
    * A dependency `EMAFactor` instance, nicknamed `"entry_signal_ema"`, with a `period` parameter.
    * Another dependency `StdDevFactor` instance, nicknamed `"volatility_filter"`, which itself depends on an `EMAFactor` nicknamed `"vol_ema"` with a `period` parameter.

    Calling `complex_factor_instance.get_nested_params_for_optimizer()` might return:
    ```json
    {
        "smoothing_level": 0.5,
        "entry_signal_ema.period": 10,
        "volatility_filter.window": 20, // Assuming StdDevFactor has a 'window' param
        "volatility_filter.vol_ema.period": 15
    }
    ```

### Step 2: Setting Up the Optimization (`FactorOptimizer`)

1.  **Initialization:** An instance of `FactorOptimizer` is created, typically taking a `BacktestEngine` instance.
2.  **`optimize_factor` Call:** You call `optimizer.optimize_factor(...)`, providing:
    * `factor_definition_template`: A **dictionary** representing the base structure and initial parameters of the factor you want to optimize.
    * `parameter_grid`: A dictionary compatible with `sklearn.GridSearchCV`, using the nickname-based paths generated in Step 1 as keys.
        ```python
        parameter_grid = {
            "smoothing_level": [0.5, 0.7],
            "entry_signal_ema.period": [8, 10, 12],
            "volatility_filter.vol_ema.period": [15, 20]
        }
        ```
    * Data parameters (`start_datetime`, `end_datetime`, `vt_symbols`, `data_interval`).
    * Optimization settings (`test_size_ratio`, `n_cv_splits`).
    * Analysis parameters (for calculating the objective score, e.g., L/S Sharpe).

3.  **Data Handling:**
    * The `FactorOptimizer` calls `backtest_engine._load_bar_data_engine()` **once** to load the full historical market data.
    * This data is then split into a training set and a test set using a helper like `FactorOptimizer._split_data_dict()`.

4.  **Estimator Creation:** A `FactorBacktestEstimator` instance is created. It's initialized with:
    * The `BacktestEngine` instance.
    * The `factor_definition_template` (the dictionary).
    * The **training portion** of the market data.
    * Other necessary parameters for running backtests within its `score` method.

### Step 3: `GridSearchCV` and `FactorBacktestEstimator.score` (The Core Loop)

1.  `GridSearchCV` is initialized with the `FactorBacktestEstimator`, the `parameter_grid`, and a `TimeSeriesSplit` for cross-validation on the training data.
2.  `GridSearchCV` iterates through each parameter combination defined in `parameter_grid`. For each combination and each CV split:
    * It calls `estimator.set_params(**current_param_combination)` on the `FactorBacktestEstimator`. This method stores the `current_param_combination` (which uses nickname paths) within the estimator.
    * It then calls `estimator.score(X_cv_split_indices, y=None)`.

3.  **Inside `FactorBacktestEstimator.score(self, X_cv_split_indices, y=None)`:**
    This method is the heart of evaluating one parameter combination on one CV split of the training data.
    a.  **Prepare Factor Definition:**
        * `current_def_dict = copy.deepcopy(self.base_factor_definition_dict)`: Creates a fresh copy of the original factor definition dictionary.
        * `modified_def_dict = apply_params_to_definition_dict_nickname_paths(current_def_dict, self.params_to_set)`:
            * This crucial utility function takes the `current_def_dict` and the `self.params_to_set` (e.g., `{"entry_signal_ema.period": 12}`).
            * It navigates the `current_def_dict` using the nickname paths and updates the parameter values directly within this dictionary structure.
            * It returns the `modified_def_dict`.
    b.  **Re-initialize Factor Tree:**
        * `target_instance, flattened_factors = self.backtest_engine_instance._init_and_flatten_factor(modified_def_dict, vt_symbols_for_this_cv_split, ...)`:
            * The `BacktestEngine` takes the `modified_def_dict` (which now reflects the current trial's parameters).
            * It creates **brand new `FactorTemplate` instances** for the target factor and all its dependencies.
            * During this re-initialization, each new factor instance calculates its `factor_key` based on its *current parameters from `modified_def_dict`*. This ensures `factor_key`s are always correct.
            * The full dependency tree is resolved and returned as `target_instance` and `flattened_factors` (a dictionary of all unique factor instances, keyed by their correct, new `factor_key`s).
    c.  **Calculate Factor Values:**
        * A new `FactorCalculator` instance is created.
        * `factor_values_cv = calculator.compute_factor_values(target_instance, flattened_factors, cv_split_training_data, ...)`: The calculator processes the *newly instantiated factor graph* on the current CV split of the training data.
    d.  **Calculate Objective Score:**
        * A new `FactorAnalyser` instance is created.
        * The analyser uses `factor_values_cv` and the corresponding CV split of market data to:
            * Prepare symbol returns.
            * Perform long-short portfolio analysis.
            * Calculate the Sharpe ratio of this L/S portfolio.
    e.  **Return Score:** The calculated Sharpe ratio is returned to `GridSearchCV`.

### Step 4: Finalizing Optimization

1.  **Best Parameters:** After trying all combinations, `GridSearchCV` identifies the `best_params_` (a dictionary with nickname paths and optimal values) and the `best_estimator_`.
2.  **Evaluation on Test Set:** The `FactorOptimizer`:
    * Takes the `best_params_`.
    * Creates a `final_optimized_definition_dict` by calling `apply_params_to_definition_dict_nickname_paths(copy.deepcopy(base_factor_definition_dict), best_params)`.
    * Calls `self.backtest_engine.run_single_factor_backtest(...)`, providing this `final_optimized_definition_dict` and instructing it to use the **test data set**.
    * This final run generates a comprehensive report on the out-of-sample performance of the factor with optimized parameters.

## Why the Dictionary-Update and Re-initialization Approach?

This approach of modifying a *factor definition dictionary* and then *completely re-initializing* the factor tree for each trial in `GridSearchCV` (Steps 3a, 3b, 3c) is critical for several reasons:

1.  **`factor_key` Integrity:** It ensures that every `FactorTemplate` instance used in a trial has a `factor_key` that accurately reflects its parameters for that specific trial. This is vital because `factor_key` is used for data caching (`FactorMemory`) and uniquely identifying computation nodes in Dask.
2.  **Clean State per Trial:** Re-instantiating factors from their definition guarantees that each trial starts with a fresh set of objects, free from any state that might have been carried over from previous trials.
3.  **Robustness:** It correctly handles cases where changing a parameter might fundamentally alter the nature or identity of a factor or its dependencies.

By using `apply_params_to_definition_dict_nickname_paths`, the optimizer can cleanly modify the "blueprint" (the dictionary) before each trial's "construction" (factor instantiation), ensuring that the `FactorCalculator` always works with a factor graph that is perfectly consistent with the parameters being evaluated.