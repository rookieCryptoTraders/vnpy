import os
os.environ['POLARS_TIME_ZONE'] = 'UTC'

import inspect

import polars as pl

# The library to inspect and generate wrappers for
import polars_talib as plta

from vnpy.factor.memory import FactorMemory

# Import the base class that our dynamic classes will inherit from
from vnpy.factor.template import FactorTemplate

# --- Constants and Mappings ---
DEFAULT_DATETIME_COL = "datetime"

# This dictionary now defines how to split multi-output functions
# into individual factor components.
MULTI_OUTPUT_FUNCTIONS = {
    # Overlap Studies
    "bbands": ["upperband", "middleband", "lowerband"],
    "mama": ["mama", "fama"],
    # Momentum Indicators
    "macd": ["macd", "macdsignal", "macdhist"],
    "macdext": ["macd", "macdsignal", "macdhist"],
    "macdfix": ["macd", "macdsignal", "macdhist"],
    "stoch": ["slowk", "slowd"],
    "stochf": ["fastk", "fastd"],
    "aroon": ["aroondown", "aroonup"],
    # Cycle Indicators
    "ht_phasor": ["inphase", "quadrature"],
    "ht_sine": ["sine", "leadsine"],
}

# Mapping of TA-Lib parameter names to required input DataFrame columns
INPUT_COLUMN_MAP = {
    "real": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

# A cache to store the generated classes so we don't regenerate them on every call
_FACTOR_CLASS_CACHE: dict[str, type[FactorTemplate]] = {}


def create_factor_class(
    func_name: str, func_obj: callable, output_component_name: str | None = None
) -> type[FactorTemplate]:
    """
    Dynamically creates a FactorTemplate subclass for a given TA-Lib function.
    If output_component_name is provided, it creates a factor for just that
    specific output of a multi-output indicator.
    """
    # --- 1. Analyze the function signature to determine parameters ---
    sig = inspect.signature(func_obj)
    params = sig.parameters

    required_input_cols = {INPUT_COLUMN_MAP[p] for p in params if p in INPUT_COLUMN_MAP}
    config_params = [
        p
        for p, v in params.items()
        if p not in INPUT_COLUMN_MAP and v.default is not inspect.Parameter.empty
    ]

    # --- 2. Define class name and factor name based on single/multi output ---
    if output_component_name:
        base_name = f"{func_name.upper()}_{output_component_name.upper()}"
    else:
        base_name = func_name.upper()

    class_name = f"{base_name}Factor"
    factor_name_upper = base_name

    # --- 3. Define the methods for our new class dynamically ---
    def __init__(
        self, setting: dict | None = None, vt_symbols: list[str] | None = None, **kwargs
    ):
        """Initializes the dynamically created factor class."""
        super(self.__class__, self).__init__(setting, **kwargs)
        self.vt_symbols: list[str] = vt_symbols if vt_symbols else []

        for p in config_params:
            if not hasattr(self.params, p):
                raise ValueError(f"{self.factor_key} requires a '{p}' parameter.")
            try:
                setattr(self, p, float(getattr(self.params, p)))
            except (ValueError, TypeError):
                setattr(self, p, int(getattr(self.params, p)))

    def calculate(
        self, input_data: dict[str, pl.DataFrame], memory: FactorMemory
    ) -> pl.DataFrame:
        """Calculates the indicator and extracts the relevant component if necessary."""
        for col in required_input_cols:
            if input_data.get(col) is None or input_data[col].is_empty():
                return pl.DataFrame(data={}, schema=self.get_output_schema())

        df_base = input_data.get("close") or next(iter(input_data.values()))

        symbol_columns = [col for col in df_base.columns if col != DEFAULT_DATETIME_COL]
        if self.vt_symbols:
            symbol_columns = [col for col in symbol_columns if col in self.vt_symbols]

        if not symbol_columns:
            return pl.DataFrame(data={}, schema=self.get_output_schema())

        kwargs = {p: getattr(self, p) for p in config_params}
        expressions_to_run = []

        for symbol in symbol_columns:
            symbol_inputs = {
                mapped_name: pl.col(symbol)
                for param_name, mapped_name in INPUT_COLUMN_MAP.items()
                if param_name in params
            }

            result_expr = func_obj(**symbol_inputs, **kwargs)

            # If this is a component of a multi-output function, extract the field.
            if output_component_name:
                result_expr = result_expr.struct.field(output_component_name)

            expressions_to_run.append(result_expr.alias(symbol))

        df_exec_base = df_base
        for col_name, df_wide in input_data.items():
            if df_wide is not df_base:
                df_exec_base = df_exec_base.join(
                    df_wide.select(pl.all().exclude(DEFAULT_DATETIME_COL)),
                    on=DEFAULT_DATETIME_COL,
                    how="left",
                )

        result_df = df_exec_base.select(
            pl.col(DEFAULT_DATETIME_COL), *expressions_to_run
        )
        return result_df

    # --- 4. Create the class object dynamically ---
    NewFactorClass = type(
        class_name,
        (FactorTemplate,),
        {
            "__init__": __init__,
            "calculate": calculate,
            "author": "Auto-Generated by DynamicLoader",
            "factor_name": factor_name_upper,
        },
    )
    return NewFactorClass


def get_factor_class_map() -> dict[str, type[FactorTemplate]]:
    """
    Inspects polars_talib and returns a dictionary mapping factor names
    to dynamically generated FactorTemplate classes. Multi-output indicators
    are split into individual classes.
    """
    if _FACTOR_CLASS_CACHE:
        return _FACTOR_CLASS_CACHE

    print("--- Dynamically Generating Factor Classes from polars_talib ---")
    factor_map = {}
    for func_name, func_obj in inspect.getmembers(plta, inspect.isfunction):
        if func_name.startswith("_"):
            continue

        try:
            # If the function is in our multi-output map, create a class for each component
            if func_name in MULTI_OUTPUT_FUNCTIONS:
                for component in MULTI_OUTPUT_FUNCTIONS[func_name]:
                    factor_class = create_factor_class(
                        func_name, func_obj, output_component_name=component
                    )
                    class_key = f"{func_name.upper()}_{component.upper()}"
                    factor_map[class_key] = factor_class
                    print(f"  -> Generated class for: {class_key}")
            # Otherwise, create a single class for the function
            else:
                factor_class = create_factor_class(func_name, func_obj)
                factor_map[func_name.upper()] = factor_class
                print(f"  -> Generated class for: {func_name.upper()}")

        except Exception as e:
            print(f"  -> Failed to generate class for '{func_name}': {e}")

    _FACTOR_CLASS_CACHE.update(factor_map)
    print(f"--- Generation Complete: {len(factor_map)} classes created ---")
    return factor_map


if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Get the dictionary of all available factor classes
    all_factors = get_factor_class_map()
    print(f"\nFound {len(all_factors)} available factor classes.")

    # 2. Example: Get one part of the BBANDS indicator
    BBANDS_UPPERBAND_Factor = all_factors.get("BBANDS_UPPERBAND")

    if BBANDS_UPPERBAND_Factor:
        bbands_settings = {
            "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0}
        }
        # Note: All BBANDS components share the same parameters
        bbands_instance = BBANDS_UPPERBAND_Factor(
            setting=bbands_settings, vt_symbols=["BTCUSDT", "ETHUSDT"]
        )

        print("\n--- Split Multi-Output Example ---")
        print(f"Instance Created: {bbands_instance.factor_key}")
        print(f"Factor Name: {bbands_instance.factor_name}")
        print(f"Output Schema: {bbands_instance.get_output_schema()}")

    # 3. Example: Get a standard single-output indicator
    SMAFactor = all_factors.get("SMA")
    if SMAFactor:
        sma_settings = {"params": {"timeperiod": 10}}
        sma_instance = SMAFactor(setting=sma_settings, vt_symbols=["BTCUSDT"])
        print("\n--- Single Output Example ---")
        print(f"Instance Created: {sma_instance.factor_key}")
        print(f"Output Schema: {sma_instance.get_output_schema()}")
