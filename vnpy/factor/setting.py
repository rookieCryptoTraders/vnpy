import json
from pathlib import Path
from typing import Any

# Caching variables
_CACHED_FACTOR_MODULE_SETTINGS: dict[str, Any] = None
_SETTINGS_INITIALIZED: bool = False

if not _SETTINGS_INITIALIZED:
    try:
        from vnpy.trader.setting import SETTINGS
        from vnpy.trader.utility import load_json as load_json_main
    except ImportError:
        # Fallback for standalone use if core vnpy parts are not available
        SETTINGS: dict[str, Any] = {}
        def load_json_fallback(filename: str) -> dict:
            filepath = Path(filename)
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: JSONDecodeError in {filename}")
                        return {}
            return {}
        load_json_main = load_json_fallback

    # Default filenames used if not specified in global SETTINGS
    DEFAULT_FACTOR_SETTINGS_FILENAME = "factor_settings.json"
    DEFAULT_FACTOR_DEFINITIONS_FILENAME = "factor_defination_setting.json"

    # --- MODIFICATION START ---
    # Both settings and definitions files are now treated as filenames only,
    # expected to be located in the main .vntrader directory. The core
    # load_json utility is responsible for finding and loading them.

    # Determine the factor definitions filename.
    FACTOR_DEFINITIONS_FILENAME = SETTINGS.get(
        "factor.definitions_file_path", DEFAULT_FACTOR_DEFINITIONS_FILENAME
    )

    # Determine the factor settings filename.
    FACTOR_SETTINGS_FILENAME = SETTINGS.get(
        "factor.settings_file_path", DEFAULT_FACTOR_SETTINGS_FILENAME
    )

    # Load Factor Module Settings from its JSON file.
    _temp_factor_module_settings = load_json_main(FACTOR_SETTINGS_FILENAME)

    # Validate that the loaded settings file contains a dictionary.
    if not isinstance(_temp_factor_module_settings, dict):
        print(f"[vnpy.factor.setting] Warning: Content of {FACTOR_SETTINGS_FILENAME} is not a valid JSON object. Using empty settings.")
        _temp_factor_module_settings = {}

    # --- MODIFICATION END ---

    # Override with values from global SETTINGS if they exist
    _keys_to_override = [
        "module_name", "datetime_col", "max_memory_length_bar",
        "max_memory_length_factor", "error_threshold"
    ]
    for key in _keys_to_override:
        if (override_value := SETTINGS.get(f"factor.{key}")) is not None:
            _temp_factor_module_settings[key] = override_value

    _CACHED_FACTOR_MODULE_SETTINGS = _temp_factor_module_settings
    _SETTINGS_INITIALIZED = True

    FACTOR_MODULE_SETTINGS: dict[str, Any] = _CACHED_FACTOR_MODULE_SETTINGS

    # Base paths for factor data, cache, etc., relative to the main .vnpy folder
    ROOT_PATH = Path(SETTINGS.get("factor.root_path", Path.cwd() / ".vnpy" / "factor"))
    DATA_PATH = ROOT_PATH / "data"
    REPORT_PATH = ROOT_PATH / "reports"
    CACHE_PATH = ROOT_PATH / "cache"

    # Ensure base directories exist
    for path in [ROOT_PATH, DATA_PATH, CACHE_PATH, REPORT_PATH]:
        path.mkdir(parents=True, exist_ok=True)

    FACTOR_SETTINGS = FACTOR_MODULE_SETTINGS

def get_factor_setting(key: str) -> Any:
    """Gets a specific factor-related setting by key."""
    return FACTOR_MODULE_SETTINGS[key]

# --- MODIFICATION START ---
def get_factor_definitions_filename() -> str:
    """Gets the filename for the factor definitions JSON file."""
    return FACTOR_DEFINITIONS_FILENAME
# --- MODIFICATION END ---

def get_factor_data_cache_path() -> Path:
    """Gets the path to the factor data cache directory."""
    return CACHE_PATH / "factor_data_cache"

def get_backtest_data_cache_path() -> Path:
    """Gets the path to the backtest factor data cache directory."""
    return CACHE_PATH / "backtest_factor_data_cache"

def get_backtest_report_path() -> Path:
    """Gets the path to the backtest reports directory."""
    return REPORT_PATH / "backtest_reports"

# Make the filename and its getter function available for import
__all__ = [
    "FACTOR_MODULE_SETTINGS",
    "FACTOR_SETTINGS",
    "get_factor_setting",
    "FACTOR_DEFINITIONS_FILENAME",         # <-- MODIFIED
    "get_factor_definitions_filename",     # <-- MODIFIED
    "ROOT_PATH",
    "DATA_PATH",
    "CACHE_PATH",
    "REPORT_PATH",
    "get_factor_data_cache_path",
    "get_backtest_data_cache_path",
    "get_backtest_report_path"
]
