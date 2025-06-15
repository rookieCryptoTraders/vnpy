"""
Manages settings for the StrategyEngine and individual strategy instances.

This module establishes a clear priority for loading configurations:
1. Hardcoded defaults within this file.
2. Values from JSON files (e.g., 'strategy_engine_config.json') located in the
   main .vntrader directory.
3. Overrides from the global VNPY_GLOBAL_SETTINGS dictionary.
"""

import json
from pathlib import Path
from typing import Any

# Caching flag to prevent re-initialization
_STRATEGY_SETTINGS_INITIALIZED = False

# Attempt to import global SETTINGS from vnpy.trader for root_path and overrides
try:
    from vnpy.trader.setting import SETTINGS as VNPY_GLOBAL_SETTINGS
    from vnpy.trader.utility import load_json as vnpy_load_json_utility
except ImportError:
    print(
        "Warning: [vnpy.strategy.setting] Failed to import from vnpy.trader. Using fallbacks."
    )
    VNPY_GLOBAL_SETTINGS: dict[str, Any] = {}

    # Basic fallback for load_json if the main utility is not available
    def vnpy_load_json_utility(filename: str) -> dict:
        # This fallback assumes the file is in the current working directory,
        # unlike the real utility which knows about the .vntrader folder.
        filepath = Path(filename)
        if filepath.exists() and filepath.is_file():
            with open(filepath, encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: JSONDecodeError in {filename}")
                    return {}
        return {}


# --- Base Paths ---
# Root path for all strategy-related data (models, persistent data, etc.)
# This can be overridden by the global SETTINGS["strategy.root_path"]
ROOT_PATH = Path(
    VNPY_GLOBAL_SETTINGS.get("strategy.root_path", Path.cwd() / ".vnpy" / "strategy")
)
MODEL_PATH = ROOT_PATH / "models"
DATA_PATH = ROOT_PATH / "data"
CACHE_PATH = ROOT_PATH / "cache"

# Ensure all data directories exist
for path in [ROOT_PATH, MODEL_PATH, DATA_PATH, CACHE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# --- Module-level Globals for Strategy Settings ---
STRATEGY_ENGINE_OPERATIONAL_PARAMS: dict[str, Any] = {}
STRATEGY_INSTANCES_FILENAME: str | None = None

if not _STRATEGY_SETTINGS_INITIALIZED:
    # --- Default Parameters for the Strategy Engine and Config Files ---
    DEFAULT_OPERATIONAL_PARAMS = {
        "strategy_code_module_path": "strategies",
        "default_execution_gateway": "DEFAULT_GW",
        "init_max_workers": 1,
        "default_retrain_interval_days": 30,
    }

    # --- Configuration Filenames ---
    # These files are expected to be in the main .vntrader directory.
    DEFAULT_OPERATIONAL_PARAMS_FILENAME = "strategy_engine_config.json"
    DEFAULT_STRATEGY_DEFINITIONS_FILENAME = "strategy_definitions.json"

    OPERATIONAL_PARAMS_FILENAME = VNPY_GLOBAL_SETTINGS.get(
        "strategy.engine_config_file", DEFAULT_OPERATIONAL_PARAMS_FILENAME
    )
    STRATEGY_INSTANCES_FILENAME = VNPY_GLOBAL_SETTINGS.get(
        "strategy.definitions_file", DEFAULT_STRATEGY_DEFINITIONS_FILENAME
    )

    # --- Load and Merge Operational Parameters ---
    # 1. Start with hardcoded defaults
    STRATEGY_ENGINE_OPERATIONAL_PARAMS = DEFAULT_OPERATIONAL_PARAMS.copy()

    # 2. Load from the operational parameters JSON file (overrides defaults)
    loaded_op_params = vnpy_load_json_utility(OPERATIONAL_PARAMS_FILENAME)
    if isinstance(loaded_op_params, dict):
        STRATEGY_ENGINE_OPERATIONAL_PARAMS.update(loaded_op_params)
        print(
            f"INFO: [vnpy.strategy.setting] Loaded operational params from '{OPERATIONAL_PARAMS_FILENAME}'."
        )

    # 3. Override with values from global VNPY_GLOBAL_SETTINGS (highest precedence)
    global_overrides_map = {
        "strategy.code_module_path": "strategy_code_module_path",
        "strategy.default_gateway": "default_execution_gateway",
        "strategy.init_workers": "init_max_workers",
    }
    for global_key, local_key in global_overrides_map.items():
        if (global_value := VNPY_GLOBAL_SETTINGS.get(global_key)) is not None:
            STRATEGY_ENGINE_OPERATIONAL_PARAMS[local_key] = global_value

    _STRATEGY_SETTINGS_INITIALIZED = True


def get_strategy_engine_operational_param(key: str, default: Any | None = None) -> Any:
    """Gets an operational parameter for the StrategyEngine."""
    return STRATEGY_ENGINE_OPERATIONAL_PARAMS.get(key, default)


def get_strategy_instance_definitions_filename() -> str:
    """
    Gets the filename for the strategy instance definitions JSON file.
    This file lists the strategies to be loaded with their specific settings.
    """
    if STRATEGY_INSTANCES_FILENAME is None:
        print(
            "CRITICAL: [vnpy.strategy.setting] STRATEGY_INSTANCES_FILENAME is not set."
        )
        return DEFAULT_STRATEGY_DEFINITIONS_FILENAME
    return STRATEGY_INSTANCES_FILENAME


# Expose for use by the StrategyEngine and other components
__all__ = [
    "STRATEGY_ENGINE_OPERATIONAL_PARAMS",
    "get_strategy_engine_operational_param",
    "STRATEGY_INSTANCES_FILENAME",
    "get_strategy_instance_definitions_filename",
    "ROOT_PATH",
    "MODEL_PATH",
    "DATA_PATH",
    "CACHE_PATH",
]
