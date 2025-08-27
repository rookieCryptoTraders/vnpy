# GEMINI.md: AI Agent Developer Guide

This document outlines the architecture, coding standards, and core principles of this quantitative trading framework. It serves as a guide for AI agents to generate code that is compliant and well-integrated.

## 1. Core Principles & Coding Standards

Adherence to these standards is mandatory for all generated code.

### Language
All code, comments, documentation, and commit messages must be in English.

### Style
The code must be object-oriented (OOP). Logic should be encapsulated within classes.

### Architecture
The framework must be decoupled. Modules should communicate via the central EventEngine and should not have direct dependencies on each other, except where explicitly designed (e.g., a StrategyTemplate using the StrategyEngine interface).

### Commands
AI prompts and commands should be specific and clear. Vague requests will be rejected.

## 2. Architectural Overview

The framework is a modular, event-driven system designed for running factor-based, model-driven trading strategies. The core components are separated by function.

### 2.1. Core Infrastructure

**EventEngine**: The central nervous system of the application. It acts as a message bus, allowing all other modules (Gateways, Engines, etc.) to communicate with each other asynchronously without being directly linked. This is the foundation of the decoupled architecture.

**Gateway**: The bridge to the outside world. Each gateway is responsible for managing the connection to a specific exchange (e.g., Binance, Bybit). It handles sending orders, receiving market data (ticks, bars), and managing the websocket connection.

**Database**: The persistence layer. The framework is designed to use ClickHouse for storing and retrieving large volumes of historical market data (K-lines) and factor data efficiently.

**MainEngine**: The central orchestrator. It initializes all other engines, provides a unified interface for interacting with gateways, and manages the overall application lifecycle.

### 2.2. The Factor Module

The factor module is responsible for all data processing and feature engineering.

**FactorEngine**: The heart of the calculation layer. It manages the lifecycle of all factor instances, using Dask for computation and Polars for data manipulation. It listens for raw market data (EVENT_BAR) and broadcasts the calculated factor values via an EVENT_FACTOR.

**FactorTemplate**: The abstract base class for all factors. Any new factor must inherit from this class and implement the calculate and get_output_schema methods. Its .to_setting() method is used to serialize its configuration to JSON.

### 2.3. The Strategy Module

The strategy module is the decision-making layer.

**StrategyEngine**: Manages the lifecycle of all strategy instances. It loads strategy configurations from a JSON file, feeds them factor data from EVENT_FACTOR events, and sends their generated orders to the MainEngine.

**StrategyTemplate**: The abstract base class for all model-driven strategies. It enforces a standard pipeline for prediction and provides helper methods for interacting with the engine.

**StrategyParameters**: A helper class that holds a strategy's configuration. All tunable settings must be managed through this object.

### 2.4. The Portfolio Manager Module

The portfolio manager is the performance tracking and risk management layer.

**PortfolioEngine**: A passive, event-driven engine. It listens for EVENT_TRADE and EVENT_ORDER events from the main event bus. It uses the reference tag on these events to automatically calculate and track the PnL, positions, and statistics for each individual strategy. It is fully decoupled from the StrategyEngine.

## 3. Configuration Workflow

The entire framework is parameter-driven via JSON files. Hardcoding parameters is strictly forbidden.

**Factor Configuration (factor_defination_setting.json)**: This file contains a list of all available factors and their default parameters. It is populated by instantiating FactorTemplate subclasses and calling their .to_setting() method.

**Strategy Configuration (strategy_definitions.json)**: This is the primary file for defining which strategies to run. Each entry must specify:

- `"class_name"`: The Python class of the strategy.
- `"strategy_name"`: A unique name for this specific instance.
- `"params"`: A dictionary holding general parameters like vt_symbols, model_path, and trading_config.
- `"model_params"`: A dictionary holding parameters specific to the predictive model, including:
  - `"model_name"`: The name of the model (e.g., "LogisticRegression").
  - `"features"`: A dictionary mapping feature aliases to their factor_key.
  - `"sklearn_params"`: A dictionary of hyperparameters for scikit-learn models.

## 4. Data Flow

The end-to-end data flow is critical to understanding the system.

1. The Gateway receives raw bar data and sends it to the EventEngine.
2. The FactorEngine consumes the bar data from the EventEngine.
3. It calculates all configured factors and broadcasts the results in an EVENT_FACTOR.
4. The StrategyEngine receives the EVENT_FACTOR.
5. It passes the factor data to the on_factor_update() method of each active strategy.
6. The StrategyTemplate subclass executes its prediction pipeline and returns a list of OrderRequest objects.
7. The StrategyEngine receives these requests, sets the .reference field to the strategy's name, and sends them to the MainEngine.
8. The MainEngine sends the order through the appropriate Gateway.
9. Upon execution, the Gateway sends trade data back, which is broadcast as an EVENT_TRADE.
10. The PortfolioEngine catches the EVENT_TRADE, reads the .reference, and updates the PnL for the corresponding strategy portfolio.

## 5. Specific Commands & Instructions for AI

- **To create a new factor**: Create a new Python file. The class must inherit from FactorTemplate and implement calculate() and get_output_schema(). For common technical indicators, utility scripts like factor_generator.py can be used to automate this process.

- **To create a new strategy**: Create a new Python file. The class must inherit from StrategyTemplate and implement the four abstract methods (_transform_latest_factors, predict_from_model, generate_signals_from_prediction, prepare_training_data).

- **When generating a strategy**: The logic must be generic. All specific details (features to use, trade sizes, thresholds) must be read from self.params or self.model_params, which are populated from the JSON file.

- **For portfolio integration**: When creating an OrderRequest inside a strategy, you must set the reference field to self.strategy_name. This is the critical link to the PortfolioEngine.

- **For data manipulation**: Always use the Polars library for its performance benefits.