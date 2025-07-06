# Factor Analysis and Calculation Engine

This module contains a comprehensive framework for factor creation, backtesting, analysis, and optimization.

## Directory Structure

- **`backtesting/`**: Contains the core logic for backtesting factors, including the `BacktestEngine`, `FactorAnalyser`, `FactorCalculator`, and `FactorInitializer`.
- **`docs/`**: Documentation related to the factor framework, including explanations of `factor_key` vs. `factor_name`, parameter management, and the optimization workflow.
- **`factors/`**: Directory for user-defined and auto-generated factor implementations.
- **`utils/`**: Utility scripts for factor management, including `factor_generator.py` for creating new factors from templates and `setting_populator.py` for managing factor configurations.
- **`__init__.py`**: Initializes the factor application.
- **`base.py`**: Defines the base enumerations and constants for the factor module.
- **`engine.py`**: The main `FactorEngine` for live factor calculation.
- **`exceptions.py`**: Custom exceptions for the factor module.
- **`factor_definition_setting.json`**: Default settings and definitions for factors.
- **`factor_settings.json`**: General settings for the factor module.
- **`memory.py`**: `FactorMemory` class for managing historical factor data.
- **`setting.py`**: Manages settings for the factor module.
- **`template.py`**: The `FactorTemplate` abstract base class that all factors should inherit from.
