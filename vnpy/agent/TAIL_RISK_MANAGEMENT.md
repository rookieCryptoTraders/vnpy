
# Tail Risk Management Mechanism for AgentEngine

## 1. Overview

This document outlines a new mechanism to handle extreme tail risks detected by the `AgentEngine`. The current `AgentEngine` analyzes text from sources like Telegram but does not act on the insights. This proposal extends its functionality to translate critical risk alerts into automated, system-wide defensive actions, such as halting new trades, canceling open orders, and optionally liquidating positions.

The design emphasizes decoupling and leverages vn.py's event-driven architecture to ensure the new components are modular and do not disrupt existing logic.

## 2. Core Architectural Changes

We will introduce two new primary components and enhance existing ones to create a robust risk management workflow.

### 2.1. `AgentEngine` Enhancement

The `AgentEngine` will be upgraded from a passive information processor to an active risk detector.

-   **Structured AI Analysis**: The `GeminiAgent`'s prompt will be modified to request a structured JSON output containing a clear risk assessment. This ensures reliable, machine-readable output.

    **Proposed Prompt:**
    ```
    "Analyze the following text for potential market-moving events or risks. Classify the risk level as 'NONE', 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'. If the risk is HIGH or CRITICAL, provide a summary and suggest actions from the list: [HALT_TRADING, CANCEL_ALL, LIQUIDATE_ALL]. Respond in JSON format: {\"risk_level\": \"...\", \"summary\": \"...\", \"suggested_actions\": [...]} "
    ```

-   **Risk Signal Emission**: Upon receiving a `CRITICAL` risk classification from the AI, the `AgentEngine` will parse the JSON and emit a new, system-wide event: `EVENT_RISK_SIGNAL`.

### 2.2. New Event: `EVENT_RISK_SIGNAL`

This new event will serve as the primary communication channel for broadcasting detected threats.

-   **Event Name**: `EVENT_RISK_SIGNAL`
-   **Data Object**: A `RiskSignal` dataclass will carry the following information:
    ```python
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class RiskSignal:
        source: str          # e.g., "AgentEngine"
        risk_level: str      # e.g., "CRITICAL"
        message: str         # Original text or AI summary
        actions: List[str]   # e.g., ["HALT_TRADING", "CANCEL_ALL"]
        scope: str           # Which symbols/strategies it applies to, e.g., "*" for all
    ```

### 2.3. New Engine: `SystemRiskEngine`

A new, dedicated engine is required to manage the system's response to `EVENT_RISK_SIGNAL`. Unlike the existing `RiskManager`, which checks outgoing orders (a reactive role), the `SystemRiskEngine` will be **proactive**, taking global actions based on incoming risk alerts.

**Responsibilities:**

1.  **Listen for `EVENT_RISK_SIGNAL`**: The engine will register a handler for this event.
2.  **Maintain System State**: It will manage a global state, such as a "trading halted" flag.
3.  **Orchestrate Actions**: Based on the `RiskSignal`, it will execute defensive actions by coordinating with other engines.

**Proposed Actions:**

1.  **Halt New Trading**:
    -   The `SystemRiskEngine` will set a global flag, e.g., `main_engine.trading_halted = True`.
    -   The existing `RiskManager`'s `intercept_send_order` method will be modified to check this flag. If `True`, it will reject all new orders. This is the most efficient and non-intrusive way to enforce a trading halt.

2.  **Cancel All Open Orders**:
    -   Upon receiving a signal with the `CANCEL_ALL` action, the `SystemRiskEngine` will fetch all active orders from the `MainEngine` (`main_engine.get_all_active_orders()`).
    -   It will then iterate through them and issue cancellation requests via `main_engine.cancel_order()`.

3.  **Liquidate All Positions (Optional & High-Risk)**:
    -   This feature must be explicitly enabled in the configuration.
    -   If signaled, the `SystemRiskEngine` will retrieve all positions from the `PortfolioEngine`.
    -   For each position, it will generate and send an immediate market order to close it. This is a highly sensitive operation and must be implemented with extreme care.

### 2.4. `RiskManager` Enhancement

The existing `RiskManager` will be slightly modified to respect the global trading halt.

-   **`intercept_send_order` Modification**:
    ```python
    # In RiskManager.intercept_send_order()
    if getattr(self.main_engine, 'trading_halted', False):
        self.write_log("Order rejected: Trading is currently halted by SystemRiskEngine.")
        return ""

    # ... existing risk checks ...
    ```

## 3. Workflow Diagram

The end-to-end process will be as follows:

```
[Telegram Message]
       |
       v
EventEngine --- (EVENT_TELEGRAM) ---> [AgentEngine]
                                          |
                                          v
                                    [GeminiAgent] (Analyzes text, returns JSON)
                                          |
                                          v
[AgentEngine] (Parses JSON, detects CRITICAL risk)
       |
       v
EventEngine --- (EVENT_RISK_SIGNAL) --> [SystemRiskEngine]
                                          |
                                          v
                                 (Orchestrates Actions)
                                /         |           \
                               /          |            \
                              v           v             v
      [RiskManager] <--- [MainEngine]   [MainEngine]   [PortfolioEngine]
(Blocks new orders)  (Cancels orders)  (Sends market    (Gets positions)
                                        orders to
                                        liquidate)
```

## 4. Configuration

All critical parameters of this mechanism will be configurable in a settings file (e.g., `system_risk_settings.json`) to ensure flexibility and safety.

-   `active`: Master switch to enable/disable the `SystemRiskEngine`.
-   `risk_threshold`: The `risk_level` that triggers action (e.g., "CRITICAL").
-   `enable_auto_liquidation`: A boolean to explicitly enable the high-risk liquidation feature. Default: `False`.
-   `liquidation_strategies`: A list of strategy names to apply liquidation to, or `["*"]` for all.

## 5. Conclusion

This design provides a powerful and modular framework for responding to tail risks. By creating a dedicated `SystemRiskEngine` and using a new `EVENT_RISK_SIGNAL`, we cleanly separate risk detection from risk response. This approach enhances the platform's resilience to sudden, adverse market events without complicating the core logic of the existing strategy and risk management engines.

