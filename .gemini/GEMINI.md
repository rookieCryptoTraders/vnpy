This is the GEMINI.md file for the `vnpy` (Core) submodule.

You have several modes of operation. You must only follow the instructions inside the protocol block for the current mode.

<PROTOCOL:EXPLAIN>
When asked to explain code in `vnpy`:
- **Core Focus**: The event-driven engine, gateway interfaces, and standard application logic.
- **Key Components**:
    - `event/`: The `EventEngine` and event types.
    - `trader/`: The `MainEngine` and standard object definitions (`BarData`, `OrderRequest`).
    - `gateway/`: Exchange adapters (Binance, OKX, etc.).
    - `app/`: Trading applications (`PortfolioManager`, `RiskManager`).
- **Philosophy**: Emphasize the decoupled nature of components communicating via events.
</PROTOCOL:EXPLAIN>

<PROTOCOL:PLAN>
When creating a plan for `vnpy`:
1.  **Scope**: Determine if the change is in the core engine, a specific gateway, or an app.
2.  **Compatibility**: Ensure changes do not break the standard `BaseGateway` or `BaseApp` interfaces.
3.  **Tooling**: Specific files to modify.
4.  **Testing**:
    - Use `pytest vnpy/tests/` for unit tests.
    - Integration tests often require the full system (root `tests/`).
5.  **Approval**: Ask for user approval before proceeding.
</PROTOCOL:PLAN>

<PROTOCOL:IMPLEMENT>
Only enter this mode after a plan has been approved.
1.  **Execute**: Apply changes sequentially.
2.  **Verify**:
    - Run `pytest vnpy/tests/`.
    - Check for circular imports, as `vnpy` has many inter-dependencies.
3.  **Style**: Strictly adhere to the existing coding style (PEP 8).
</PROTOCOL:IMPLEMENT>