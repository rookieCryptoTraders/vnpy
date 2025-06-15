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
    # The logic for resolving the factor definitions file path has been simplified.
    # We now only store the filename, assuming it will be located in the
    # main .vntrader directory and loaded by a utility function that is
    # aware of that directory.

    # Determine the factor definitions *filename*.
    FACTOR_DEFINITIONS_FILENAME = SETTINGS.get(
        "factor.definitions_file_path", DEFAULT_FACTOR_DEFINITIONS_FILENAME
    )

    # --- MODIFICATION END ---

    # Root path of this module, used for resolving factor settings file if relative
    MODULE_ROOT_PATH = Path(__file__).parent

    # Determine the factor settings file path (This file can still be local to the module)
    _factor_settings_file_path_str = SETTINGS.get("factor.settings_file_path", DEFAULT_FACTOR_SETTINGS_FILENAME)
    _factor_settings_filepath = Path(_factor_settings_file_path_str)
    if not _factor_settings_filepath.is_absolute():
        _factor_settings_filepath = MODULE_ROOT_PATH / _factor_settings_filepath

    # Load Factor Module Settings from its JSON file
    _temp_factor_module_settings: dict[str, Any] = {}
    if _factor_settings_filepath.exists():
        _temp_factor_module_settings = load_json_main(str(_factor_settings_filepath))
        if not isinstance(_temp_factor_module_settings, dict):
            print(f"[vnpy.factor.setting] Warning: Content of {_factor_settings_filepath} is not a valid JSON object.")
            _temp_factor_module_settings = {}
    else:
        print(f"[vnpy.factor.setting] Warning: Factor settings file not found at {_factor_settings_filepath}.")

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
