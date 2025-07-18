# Code Review Report: vnpy/vnpy/factor/engine.py

**Date:** July 18, 2025
**Reviewer:** Gemini
**Status:** Revised and Expanded

## 1. Overall Summary

The `FactorEngine` is a powerful and well-architected component for event-driven factor calculation. It effectively leverages Dask for building and executing computational graphs and Polars for efficient in-memory data manipulation. The design shows a strong emphasis on robustness, with detailed error handling, dependency resolution, and resource monitoring.

This revised review provides more precise, actionable recommendations for the critical issues identified previously, along with an analysis of a new concern regarding data stream latency. The primary goals are to **unlock parallel performance**, **increase reliability through deterministic processing**, and **improve responsiveness in live trading**.

## 2. Strengths

-   **Modern Architecture:** The engine demonstrates a clean separation of concerns. `FactorTemplate` holds the calculation logic, `FactorMemory` manages state and persistence, and the `FactorEngine` acts as the central orchestrator.
-   **Advanced Computation Model:** The use of `dask.delayed` to build a lazy, directed acyclic graph (DAG) of factor dependencies is excellent. This allows for efficient, declarative, and potentially parallel execution.
-   **Robust Dependency Management:** The implementation includes a topological sort to correctly order factor calculations and a crucial check to detect and report circular dependencies, preventing runtime failures.
-   **High-Performance Data Handling:** The choice of Polars for managing in-memory `memory_bar` data is ideal for this type of financial application, offering significant performance advantages over pandas for many operations.
-   **Comprehensive Error Handling:** The `execute_calculation` method features robust error handling, including a "circuit breaker" pattern (`consecutive_errors`) that stops the engine after a threshold of repeated failures, preventing system instability.

## 3. Critical Areas for Improvement

### 3.1. Dask Scheduler Configuration (Performance)

-   **Issue:** The code explicitly forces Dask to run in `single-threaded` mode, negating its primary benefit of parallel computation. The comment `this would fix the interpreter shut down issue` points to a critical thread-safety problem.
-   **Analysis:** The crash likely originates from non-thread-safe operations within a factor's `calculate` method, especially when accessing shared resources like `FactorMemory` instances. While `FactorMemory` has a write lock, concurrent reads during a multi-threaded computation could still lead to instability in underlying C extensions (e.g., in file access or NumPy/Polars operations).
-   **Recommendations:**
    1.  **Adopt a Stateless Calculation Model:** The most robust solution is to make factor calculations pure and stateless. Instead of passing the stateful `FactorMemory` object to Dask workers, pre-fetch the required historical data in the main thread and pass it as an immutable Polars DataFrame.
        -   **Action:** Modify `create_task` to read the necessary lookback data from `FactorMemory` and pass this static DataFrame to the `dask.delayed` call. This isolates workers and eliminates the root cause of thread-safety issues.
    2.  **Isolate and Diagnose:** As a first step, re-enable the `threads` scheduler (`scheduler='threads'`) and wrap the `dask.compute` call in a `try...except` block to identify which specific factor is causing the crash.
    3.  **Enforce Read-Only Contract:** Ensure all factor implementations use the `memory` object in a strictly read-only fashion.

### 3.2. Historical Data Filling (Reliability)

-   **Issue:** The `process_factor_filling_event` method uses `time.sleep(1)` to wait for historical data, creating a race condition. This is unreliable and can lead to failures or calculations on incomplete data.
-   **Analysis:** The engine already has the necessary event handlers (`process_load_bar_response_event`) but does not use them to synchronize the filling process. The logic should be event-driven, not time-based.
-   **Recommendations:**
    1.  **Implement a Two-Stage, Event-Driven Workflow:**
        -   **Stage 1 (Request):** `process_factor_filling_event` should only store the request parameters (start/end dates) in an instance variable (e.g., `self._active_filling_request`) and emit the `EVENT_HISTORY_DATA_REQUEST`.
        -   **Stage 2 (Execute):** Modify `process_load_bar_response_event` to check for an active filling request. Once the historical bar data is loaded, it should trigger a new, dedicated method (e.g., `execute_historical_fill`) that performs the actual calculations.
    2.  **Refactor for Clarity:** Extract the complex iteration logic from the current filling method into the new `execute_historical_fill` method and remove all duplicated and commented-out code. This will make the process deterministic and easier to maintain.

## 4. New Recommendation: Handling Delayed Bar Data (Responsiveness)

-   **Issue:** The current engine waits for bars from **all** tracked symbols before starting a calculation (`if all(self.receiving_status.values())`). A single slow or dead data feed can halt all factor computations, which is a significant vulnerability in a live trading environment.
-   **Analysis:** This all-or-nothing approach prioritizes completeness over timeliness. For live trading, it's often better to calculate on partially complete data than to wait indefinitely.
-   **Recommendations:**
    1.  **Implement a Batch Timeout Mechanism:**
        -   Introduce a configurable `bar_batch_timeout` (e.g., 0.5 seconds).
        -   When the first bar of a new batch arrives, start a timer (using `threading.Timer` or a delayed event if the `EventEngine` supports it).
        -   If the timer expires before all bars have arrived, proceed with the `on_bars` calculation using the data received so far. If all bars arrive before the timeout, cancel the timer and proceed immediately.
    2.  **Enforce Null-Resistant Factors:** This change requires that all factor `calculate` methods are robust to missing data (i.e., can handle `null` values in the `memory_bar` DataFrames). This is a critical prerequisite for ensuring stability.

## 5. General Recommendations & Refactoring

### 5.1. Idiomatic Data Pivoting

-   **Issue:** The `process_database_bar_data` method uses a manual, iterative approach to pivot data from a long to a wide format.
-   **Recommendation:** Replace the manual loop with Polars' built-in `pivot` method. It is more idiomatic, readable, and performant.
    ```python
    # Suggested replacement
    pivoted = df.pivot(index="datetime", columns="vt_symbol", values=col).sort("datetime")
    ```

### 5.2. Explicit Lookback Period Contract （SOLVED）

-   **Issue:** The engine relies on a hardcoded list of parameter names (`lookback_attrs`) to infer a factor's required lookback period. This is brittle.
-   **Recommendation:** Formalize this contract by adding a dedicated method to the `FactorTemplate` base class.
    ```python
    # In vnpy/factor/template.py
    class FactorTemplate:
        # ...
        def get_lookback_period(self) -> int:
            """Returns the number of bars required for the calculation."""
            # Default implementation can use old logic for backward compatibility
            # New factors should override this for explicit declaration.
            return 60
    ```
    Update `init_all_factors` to call this method, creating a clear and stable API for all factors.

## 6. Conclusion

The `FactorEngine` is a strong piece of engineering. By focusing on the recommendations above, it can be elevated to a truly high-performance and resilient system. The highest priorities are resolving the Dask parallelism issue to unlock multi-core processing and implementing a deterministic, event-driven workflow for historical data filling. The addition of a bar batch timeout will significantly improve its robustness for live trading.