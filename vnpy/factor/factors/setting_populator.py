import importlib
import inspect
import json
from pathlib import Path
import sys

# The library to inspect for default parameters
import polars_talib as plta

from vnpy.trader.constant import Interval

# --- Configuration ---

# Add the parent directory of your generated package to the Python path.
# This ensures that we can import the 'ta_lib' module successfully.
GENERATED_FACTORS_ROOT = Path(__file__).parent
if str(GENERATED_FACTORS_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATED_FACTORS_ROOT))

# The path for the JSON file that will store the generated factor settings.
SETTINGS_JSON_PATH = GENERATED_FACTORS_ROOT / "factor_defination_setting.json"
PACKAGE_NAME = "ta_lib"


def get_original_func_name(class_name: str) -> str:
    """
    Parses a generated class name to find the original ta-lib function name.
    Example: 'BBANDS_UPPERBANDFactor' -> 'bbands'
    Example: 'SMAFactor' -> 'sma'
    """

    # Take the part before the first underscore for multi-output indicators
    return class_name.split('_')[0].lower()


def populate_settings_file():
    """
    Finds all generated factor classes by importing the generated package,
    instantiates them with default parameters, and saves their configurations
    to a single JSON file.
    """
    print("--- Starting Factor Setting Populator ---")

    try:
        # Dynamically import the top-level __init__.py of the package
        factor_package = importlib.import_module(PACKAGE_NAME)
        print(f"Successfully imported generated package: '{PACKAGE_NAME}'")
    except ImportError:
        print(f"ERROR: Could not import the '{PACKAGE_NAME}' package.")
        print(
            f"Please ensure the generator script has been run and the '{PACKAGE_NAME}' directory exists at: {GENERATED_FACTORS_ROOT}"
        )
        return

    # Get the list of all exported class names
    if not hasattr(factor_package, "__all__"):
        print(
            "ERROR: The generated package's __init__.py is missing the '__all__' variable."
        )
        return

    all_class_names = factor_package.__all__
    all_settings = []

    # Iterate through the official list of classes from __all__
    for class_name in sorted(all_class_names):
        try:
            FactorClass = getattr(factor_package, class_name)

            # Deduce the original ta-lib function name from the class name
            original_func_name = get_original_func_name(class_name)

            # Inspect the original function to get its default parameters
            func_obj = getattr(plta, original_func_name)
            sig = inspect.signature(func_obj)

            config_params = [
                p
                for p, v in sig.parameters.items()
                if p not in {"real", "open", "high", "low", "close", "volume", "price"}
                and v.default is not inspect.Parameter.empty
            ]
            default_params = {
                p_name: p_obj.default
                for p_name, p_obj in sig.parameters.items()
                if p_name in config_params
            }

            # Create a setting dictionary and instantiate the class
            setting_dict = {"params": default_params}
            instance = FactorClass(setting=setting_dict)

            instance.freq = Interval.MINUTE  # Set a default frequency

            # Get the full setting from the instance and add it to our list
            setting_to_save = instance.to_setting()
            all_settings.append(setting_to_save)
            print(f"  -> Generated setting for: {class_name}")

        except Exception as e:
            print(f"  -> FAILED to process class {class_name}: {e}")

    # Write all collected settings to the JSON file, overwriting the old one
    with open(SETTINGS_JSON_PATH, "w", encoding="utf-8") as f:
        print(all_settings)
        json.dump(all_settings, f, indent=4)

    print("\n--- Population Complete ---")
    print(f"Generated settings for {len(all_settings)} factors.")
    print(f"Settings saved to '{SETTINGS_JSON_PATH}'.")


if __name__ == "__main__":
    populate_settings_file()
