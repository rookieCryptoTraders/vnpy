import json
from pathlib import Path
from typing import Any, Dict

# Caching variables
_CACHED_FACTOR_MODULE_SETTINGS: Dict[str, Any] = None
_SETTINGS_INITIALIZED: bool = False

if not _SETTINGS_INITIALIZED:
    try:
        from vnpy.trader.setting import SETTINGS
        from vnpy.trader.utility import load_json as load_json_main
    except ImportError:
        SETTINGS: Dict[str, Any] = {}
        # Basic load_json fallback for standalone if vnpy.trader.utility.load_json is not available
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

    # Default filenames used if not specified in global SETTINGS or for standalone mode
    DEFAULT_FACTOR_SETTINGS_FILENAME = "factor_settings.json"
    DEFAULT_FACTOR_DEFINITIONS_FILENAME = "factor_maker_setting.json"

    # Root path of this module, used for resolving relative paths in standalone mode
    MODULE_ROOT_PATH = Path(__file__).parent

    # Determine the factor settings file path
    _factor_settings_file_path_str = SETTINGS.get("factor.settings_file_path", DEFAULT_FACTOR_SETTINGS_FILENAME)
    _factor_settings_filepath = Path(_factor_settings_file_path_str)
    if not _factor_settings_filepath.is_absolute():
        _factor_settings_filepath = MODULE_ROOT_PATH / _factor_settings_filepath

    # Determine the factor definitions file path
    _factor_definitions_file_path_str = SETTINGS.get("factor.definitions_file_path", DEFAULT_FACTOR_DEFINITIONS_FILENAME)
    FACTOR_DEFINITIONS_FILEPATH = Path(_factor_definitions_file_path_str)
    if not FACTOR_DEFINITIONS_FILEPATH.is_absolute():
        FACTOR_DEFINITIONS_FILEPATH = MODULE_ROOT_PATH / FACTOR_DEFINITIONS_FILEPATH

    # Load Factor Module Settings from the JSON file
    _temp_factor_module_settings: Dict[str, Any] = {}
    if _factor_settings_filepath.exists():
        _temp_factor_module_settings = load_json_main(str(_factor_settings_filepath))
        if not isinstance(_temp_factor_module_settings, dict): # Ensure it's a dict if file was empty or malformed
            print(f"[vnpy.factor.setting] Warning: Content of factor settings file {_factor_settings_filepath} is not a valid JSON object. Initializing with empty settings.")
            _temp_factor_module_settings = {}
    else:
        print(f"[vnpy.factor.setting] Warning: Factor settings file not found at {_factor_settings_filepath}. Initializing with empty settings.")

    # Override _temp_factor_module_settings with values from global SETTINGS if they exist
    _keys_to_override = [
        "module_name",
        "datetime_col",
        "max_memory_length_bar",
        "max_memory_length_factor",
        "error_threshold"
    ]
    for key in _keys_to_override:
        override_value = SETTINGS.get(f"factor.{key}")
        if override_value is not None:
            _temp_factor_module_settings[key] = override_value
    
    _CACHED_FACTOR_MODULE_SETTINGS = _temp_factor_module_settings
    _SETTINGS_INITIALIZED = True

    FACTOR_MODULE_SETTINGS: Dict[str, Any] = _CACHED_FACTOR_MODULE_SETTINGS

    # Base paths for factor data, cache, etc.
    ROOT_PATH = Path(SETTINGS.get("factor.root_path", Path.cwd() / ".vnpy" / "factor"))
    DATA_PATH = ROOT_PATH / "data"
    CACHE_PATH = ROOT_PATH / "cache"

    # Ensure base directories exist
    for path in [ROOT_PATH, DATA_PATH, CACHE_PATH]:
        path.mkdir(parents=True, exist_ok=True)

    # For easier access if needed, though direct use of FACTOR_MODULE_SETTINGS is common
    FACTOR_SETTINGS = FACTOR_MODULE_SETTINGS

def get_factor_setting(key: str) -> Any:
    """
    Get a factor-related setting by key.
    Checks FACTOR_MODULE_SETTINGS.
    """
    value = FACTOR_MODULE_SETTINGS.get(key)
    if value is None and key not in FACTOR_MODULE_SETTINGS: # Check key existence for explicit None values
        raise KeyError(f"Setting key '{key}' not found in factor module settings.")
    return value

def get_factor_definitions_filepath() -> Path:
    """Get the absolute path to the factor definitions JSON file."""
    return FACTOR_DEFINITIONS_FILEPATH

def get_factor_data_cache_path() -> Path:
    """Get the path to the factor data cache directory."""
    return CACHE_PATH / "factor_data_cache"

def get_backtest_data_cache_path() -> Path:
    """Get the path to the backtest factor data cache directory."""
    return CACHE_PATH / "backtest_factor_data_cache"

# Make FACTOR_DEFINITIONS_FILEPATH available for import if needed elsewhere for clarity,
# though get_factor_definitions_filepath() is the preferred accessor.
__all__ = [
    "FACTOR_MODULE_SETTINGS",
    "FACTOR_SETTINGS",  # Alias for FACTOR_MODULE_SETTINGS
    "get_factor_setting",
    "FACTOR_DEFINITIONS_FILEPATH",
    "get_factor_definitions_filepath",
    "ROOT_PATH",
    "DATA_PATH",
    "CACHE_PATH",
    "get_factor_data_cache_path",
    "get_backtest_data_cache_path"
]
