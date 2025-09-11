# vn.py Crypto Trading Framework

**vn.py** is a comprehensive, modular, event-driven quantitative trading framework written in Python. It provides a complete suite of tools for connecting to various exchanges, receiving market data, and managing trades. The framework is designed with a decoupled architecture, allowing for high extensibility and easy integration of custom modules.

## Features

*   **Modular**: A decoupled architecture allows for high extensibility and easy integration of custom modules.
*   **Event-Driven**: Ensures that different components can communicate asynchronously without being tightly linked.
*   **Extensible**: Easily integrate custom modules and gateways.
*   **Cross-Platform**: Runs on macOS, and Linux.
*   **Multi-Asset**: Supports trading in cryptocurrencies.
*   **Advanced Risk Management**: Includes a sophisticated, AI-powered tail risk management system.

## Getting Started

### Installation

It is recommended to use a virtual environment to install `vn.py`.

```bash
# Clone the repository
git clone https://github.com/rookieCryptoTraders/vnpy.git
cd vnpy

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Running a Simple Example

The following example demonstrates how to initialize the framework, connect to a gateway, and subscribe to market data.

```python
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.gateway.binance import BinanceSpotGateway
from vnpy.trader.object import SubscribeRequest, Exchange

# 1. Initialize the engines
event_engine = EventEngine()
main_engine = MainEngine(event_engine)

# 2. Add the Binance Spot gateway
main_engine.add_gateway(BinanceSpotGateway)

# 3. Connect to the gateway
#    (Replace with your actual API keys for live trading)
settings = {
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET",
    "server": "TESTNET"  # Use TESTNET for safe testing
}
main_engine.connect(settings, "BINANCE_SPOT")

# 4. Subscribe to market data
req = SubscribeRequest(symbol="BTCUSDT", exchange=Exchange.BINANCE)
main_engine.subscribe(req, "BINANCE_SPOT")

print("Framework initialized. Waiting for market data...")

# Keep the script running to receive data
# In a real application, you would have a loop here to process events
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Exiting...")
    main_engine.close()
```

## Core Engines

The heart of vn.py is its event-driven architecture, which ensures that different components can communicate asynchronously without being tightly linked. This promotes modularity and scalability.

*   **`MainEngine`**: The central orchestrator of the framework. It is responsible for initializing and managing all other engines, loading gateways, and providing a unified API for core trading actions.
*   **`EventEngine`**: The central message bus of the entire platform. All modules communicate by putting `Event` objects into the `EventEngine` and listening for events they are interested in.
*   **`LogEngine`**: A dedicated engine for logging. It captures log events from all other parts of the system and outputs them to the console and/or log files.
*   **`EmailEngine`**: Provides functionality for sending emails, which can be used for notifications, alerts, or reports.

## Gateways

Gateways are the bridges to external exchanges (e.g., Binance, OKX) or data sources (e.g., Telegram, Twitter). Each gateway implements a standardized interface for:

*   Connecting to the external API.
*   Subscribing to real-time data.
*   Sending, updating, and canceling orders.
*   Querying account balances, positions, and historical data.

The project includes several pre-built gateways, and you can easily create your own.

## Application Modules

vn.py includes several pre-built application modules that provide essential trading functionalities:

*   **`PortfolioManager`**: Tracks trading performance, calculating PnL, positions, and key statistics.
*   **`DataRecorder`**: Captures and stores market data for historical analysis and backtesting.
*   **`RiskManager`**: Provides pre-trade risk control by intercepting outgoing orders and enforcing rules.

## AI-Powered Tail Risk Management

`vn.py` features an advanced mechanism to handle extreme tail risks. The `AgentEngine` can analyze text from sources like Telegram using an AI model (e.g., Gemini) to detect critical market-moving events.

When a critical risk is identified, the `AgentEngine` emits a `EVENT_RISK_SIGNAL`. A dedicated `SystemRiskEngine` listens for this signal and can trigger defensive actions, such as:

*   **Halting new trades** across the system.
*   **Canceling all open orders**.
*   **(Optionally) Liquidating all positions**.

This provides a powerful, automated layer of defense against sudden market shocks.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
