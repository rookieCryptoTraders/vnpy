# Guide to Managing Factor Parameters

This document explains how to inspect and modify parameters for `FactorTemplate` instances and their underlying dictionary definitions. This is crucial for manual tuning, programmatic adjustments, and building robust optimization routines. We'll focus on using nickname-based paths for clarity and stability.

## 1. Understanding Parameter Paths (Nickname-Based)

To uniquely identify and modify parameters within a potentially nested structure of factors (where one factor can depend on another, which can depend on another, and so on), we use a dot-separated path.

* **Parameter of the Main Factor:** Accessed directly by its name (e.g., `"alpha"`).
* **Parameter of a Direct Dependency:** Accessed by `dependency_nickname.parameter_name` (e.g., `"ema_short.period"`). The "dependency nickname" is the `factor_name` you assign to a dependency within its parent factor's configuration.
* **Parameter of a Nested Dependency:** Accessed by chaining nicknames: `dep_nickname1.nested_dep_nickname2.parameter_name` (e.g., `"my_macd.fast_ema_of_macd.period"`).

**Nickname De-duplication Suffix:**
If a factor has multiple direct dependencies that were unfortunately given the same nickname in the configuration (e.g., two dependencies both nicknamed "filter"), the system appends a counter (`_0`, `_1`, etc.) to these nicknames when generating parameter paths to ensure uniqueness. For example:
* `"filter_0.window"`
* `"filter_1.window"`
The functions described below are designed to work with these (potentially suffixed) effective nicknames in the paths.

## 2. Method 1: Getting All Parameters from a Factor Instance

This method allows you to extract a complete, flat dictionary of all tunable parameters from a live `FactorTemplate` instance and its entire dependency tree.

* **Function:** `FactorTemplate.get_nested_params_for_optimizer()`
* **Called On:** An instance of `FactorTemplate` (e.g., your main factor object).
* **Input:** None (or an optional `current_path_prefix` for internal recursion).
* **Output:** A flat dictionary.
    * Keys: Dot-separated nickname paths to each parameter.
    * Values: The current values of these parameters.
* **Use Cases:**
    * Inspecting the complete current configuration of a complex factor graph.
    * Providing the initial set of parameters to an optimization algorithm (like `GridSearchCV`).
    * Serializing the "tunable state" of a factor graph if needed elsewhere.

* **Example:**

    ```python
    # Assume 'complex_factor' is an initialized FactorTemplate instance:
    # complex_factor (params: {"level": 0.5})
    #   L-> dependency_A (nicknamed "ema_entry", EMAFactor, params: {"period": 10})
    #   L-> dependency_B (nicknamed "vol_filter", StdDevFactor, params: {"window": 20})
    #         L-> dependency_B1 (nicknamed "base_ema", EMAFactor, params: {"period": 15})

    all_params = complex_factor.get_nested_params_for_optimizer()
    # all_params might look like:
    # {
    #     "level": 0.5,
    #     "ema_entry.period": 10,
    #     "vol_filter.window": 20,
    #     "vol_filter.base_ema.period": 15
    # }
    ```

## 3. Method 2: Setting Parameters on Live Factor Instances

This method allows you to programmatically change parameters on an *existing, live* `FactorTemplate` instance and its already instantiated dependencies.

* **Function:** `FactorTemplate.set_nested_params_for_optimizer(nested_params_dict)`
* **Called On:** An instance of `FactorTemplate`.
* **Input:**
    * `nested_params_dict`: A flat dictionary where keys are the nickname-based parameter paths and values are the new parameter values you want to set.
* **Action:**
    * The method parses the path keys.
    * It recursively navigates through the `self.dependencies_factor` list (which contains live `FactorTemplate` instances of dependencies), matching the nicknames in the path to the `factor_name` attributes of these dependency instances.
    * It updates the `params` object of the targeted factor instances (both the main one and its dependencies).
* **Use Cases:**
    * Programmatically adjusting parameters of an active factor graph based on external inputs or conditions.
    * Interactive tuning or "what-if" scenarios where you directly manipulate live objects.
    * Certain optimization strategies that might involve mutating instances (though with caveats).

* **Important Considerations When Mutating Live Instances:**
    * **`factor_key` Staleness:** The `factor_key` of a `FactorTemplate` instance is typically generated during its `__init__` based on its initial parameters. If you use `set_nested_params_for_optimizer` to change a parameter that is part of the `factor_key`'s definition (e.g., an EMA's `period`), the `factor_key` attribute on that live instance will **not automatically update**. It becomes "stale."
        * This can lead to issues if other parts of your system rely on `factor_key` for data caching (like `FactorMemory`) or for uniquely identifying computation nodes in Dask. The system might use the old key for new data or fail to recompute when it should.
    * **Re-calculation Needed:** After changing parameters, the factor (and any parent factors that depend on it) will need to be explicitly re-calculated for the changes to take effect in their output values.
    * **Clean State:** For rigorous backtesting or optimization, starting each trial with freshly initialized instances is often preferred to avoid unintended state carry-over from previous mutations.

* **Example:**

    ```python
    # Assume 'main_factor' is an initialized FactorTemplate instance.
    # It has a dependency nicknamed "ema_short" which is an EMAFactor.
    # It also has a dependency "macd_main" which has its own dependency "ema_fast".

    params_to_set = {
        "alpha": 0.75,  # Update 'alpha' param of main_factor
        "ema_short.period": 12,  # Update 'period' of direct dependency "ema_short"
        "macd_main.ema_fast.period": 10 # Update 'period' of a nested dependency
    }

    main_factor.set_nested_params_for_optimizer(params_to_set)

    # Now, main_factor.params["alpha"] would be 0.75.
    # The 'ema_short' dependency instance would have its params["period"] updated to 12.
    # The 'ema_fast' (dependency of 'macd_main') instance would have its params["period"] updated to 10.
    # Note: Their factor_key attributes remain unchanged from their initial values.
    ```

## 4. Method 3: Modifying a Factor Definition Dictionary

This utility function works on the dictionary representation of a factor (e.g., as loaded from a JSON file or generated by `FactorTemplate.to_setting()`), not on live instances. It's particularly crucial for robust optimization workflows.

* **Function:** `apply_params_to_definition_dict_nickname_paths(definition_dict, params_with_paths)`
    * (This function would typically reside in your `factor_utils.py` or a similar utility module.)
* **Input:**
    * `definition_dict`: The factor configuration *dictionary*.
    * `params_with_paths`: A flat dictionary with nickname-based paths as keys and new parameter values.
* **Action:**
    * Creates a **deep copy** of the input `definition_dict`.
    * Parses the nickname-based paths in `params_with_paths`.
    * Navigates the copied dictionary structure by matching nicknames in the path with the `factor_name` fields within the `dependencies_factor` lists of the dictionary. It also handles the `_0`, `_1` suffixes for de-duplicated nicknames if they were generated by `get_nested_params_for_optimizer`.
    * Updates the parameter values in the corresponding `"params"` sub-dictionaries within the copied `definition_dict`.
* **Output:** A new, modified factor definition dictionary. The original `definition_dict` is unchanged.
* **Primary Use Case: Parameter Optimization (e.g., with `FactorBacktestEstimator` in `GridSearchCV`)**
    1.  You start with a `base_factor_definition_dict`.
    2.  In each trial of your optimizer (e.g., `FactorBacktestEstimator.score`):
        a.  Create a `current_trial_definition_dict = copy.deepcopy(base_factor_definition_dict)`.
        b.  Get the parameter set for the current trial from `GridSearchCV` (e.g., `{"ema_short.period": 12, "alpha": 0.2}`).
        c.  `modified_definition_dict = apply_params_to_definition_dict_nickname_paths(current_trial_definition_dict, trial_params)`.
        d.  **Crucially, you then re-initialize the entire factor tree from this `modified_definition_dict`**:
            `target_instance, flattened_factors = BacktestEngine._init_and_flatten_factor(modified_definition_dict, ...)`
            This creates **new** `FactorTemplate` instances. Each new instance will generate its `factor_key` based on the parameters now present in `modified_definition_dict`, ensuring `factor_key` integrity.
    * This ensures each optimization trial works with a clean, correctly keyed set of factor instances.
* **Other Use Cases:**
    * Programmatically generating variations of a base factor configuration before saving them.
    * Updating stored factor configurations before they are loaded and instantiated.

* **Example:**

    ```python
    # Assume base_definition_dict is loaded from JSON or from factor_instance.to_setting()
    base_definition_dict = {
        "class_name": "MyMainFactor",
        "factor_name": "Main",
        "params": {"alpha": 0.1},
        "dependencies_factor": [
            {
                "class_name": "EMAFactor",
                "factor_name": "ema_short", # Nickname
                "params": {"period": 10},
                "factor_key": "factor.emafactor.1m@period_10" # Example original key
            },
            {
                "class_name": "AnotherFactor",
                "factor_name": "filter_0", # Potentially de-duplicated if original nickname was "filter"
                "params": {"length": 5}
            }
        ]
    }

    params_to_update_in_dict = {
        "alpha": 0.2,
        "ema_short.period": 12,
        "filter_0.length": 7 
    }

    # Assume apply_params_to_definition_dict_nickname_paths is imported
    modified_def_dict = apply_params_to_definition_dict_nickname_paths(
        base_definition_dict,
        params_to_update_in_dict
    )

    # modified_def_dict will now be:
    # {
    #     "class_name": "MyMainFactor",
    #     "factor_name": "Main",
    #     "params": {"alpha": 0.2}, # Updated
    #     "dependencies_factor": [
    #         {
    #             "class_name": "EMAFactor",
    #             "factor_name": "ema_short",
    #             "params": {"period": 12}, # Updated
    #             "factor_key": "factor.emafactor.1m@period_10" # Original key, will be regenerated on init
    #         },
    #         {
    #             "class_name": "AnotherFactor",
    #             "factor_name": "filter_0", 
    #             "params": {"length": 7} # Updated
    #         }
    #     ]
    # }

    # Next, you would re-initialize your factor from modified_def_dict:
    # target_instance, flattened_factors = BacktestEngine._init_and_flatten_factor(modified_def_dict, ...)
    # The new 'target_instance' and its dependencies (in 'flattened_factors')
    # will have their factor_keys correctly reflecting period=12 for "ema_short", etc.
    ```

## Choosing the Right Method

* **`FactorTemplate.get_nested_params_for_optimizer()`**:
    * **Use when:** You need to *inspect* the current parameters of a live factor instance and its entire dependency tree, or to get an initial flat parameter set for an optimizer.

* **`FactorTemplate.set_nested_params_for_optimizer()`**:
    * **Use when:** You want to *directly modify parameters on existing, live factor instances* for quick experiments or dynamic adjustments where you are aware of and can manage the implications of potential `factor_key` staleness.

* **`apply_params_to_definition_dict_nickname_paths()` (from utils):**
    * **Use when:** You need to modify the parameters in a factor's *dictionary definition* (its blueprint). This is the **recommended approach for optimization loops** because it allows you to create a modified blueprint and then re-initialize the entire factor tree from it, ensuring each trial starts with a clean state and correctly generated `factor_key`s for all instances.

By using these methods appropriately, you can effectively manage and tune complex factor structures.