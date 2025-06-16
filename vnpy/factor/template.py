from __future__ import annotations  # For List['FactorTemplate'] type hint
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, TypeVar
import polars as pl
from dask.delayed import Delayed

from vnpy.factor.utils.factor_utils import init_factors

# Assuming FactorMemory is in a sibling file 'factor_memory.py'
# If FactorMemory is defined elsewhere, adjust the import path accordingly.
from vnpy.factor.memory import FactorMemory


from vnpy.factor.base import FactorMode
from vnpy.trader.constant import Exchange, Interval
from vnpy.config import VTSYMBOL_FACTOR  # General vnpy config/constant

DEFAULT_DATETIME_COL = "datetime"


class FactorParameters(object):  # noqa: UP004
    """
    A flexible container for storing and managing factor parameters.
    Parameters can be accessed as attributes.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the FactorParameters object.

        Parameters:
            params (Optional[Dict[str, Any]]): A dictionary of parameter names and values.
        """
        if params is not None:
            for key, value in params.items():
                setattr(self, key, value)

    def __str__(self) -> str:
        """
        Convert the parameters to a string representation, typically for keys or filenames.
        Example: "param1_value1-param2_value2"

        Returns:
            str: A string representation of the parameters.
        """
        if not self.__dict__:
            return "noparams"
        return "-".join(
            [f"{k}_{v}" for k, v in sorted(self.__dict__.items())]
        )  # Sort for consistency

    def __iter__(self):
        """
        Iterate over the parameters (name, value) pairs.
        """
        return iter(self.__dict__.items())

    def __contains__(self, item: str) -> bool:
        """
        Check if a parameter name exists.

        Parameters:
            item (str): The parameter name.

        Returns:
            bool: True if the parameter exists, False otherwise.
        """
        return hasattr(self, item)

    def set_parameters(self, params: dict[str, Any]) -> None:
        """
        Set or update parameters.

        Parameters:
            params (Dict[str, Any]): A dictionary of parameter names and values.
        """
        for key, value in params.items():
            # Optional: Add a log or print if a parameter is updated vs. newly set
            # if hasattr(self, key):
            #     print(f"Parameter '{key}' updated: {getattr(self, key)} -> {value}")
            # else:
            #     print(f"Parameter '{key}' set: {value}")
            setattr(self, key, value)

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get the value of a parameter.

        Parameters:
            key (str): The parameter name.
            default (Any): The default value to return if the parameter is not found.

        Returns:
            Any: The value of the parameter or the default value.
        """
        return getattr(self, key, default)

    def get_all_parameters(self) -> dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary of all parameters.
        """
        return self.__dict__.copy()  # Return a copy

    def del_parameters(self, key_or_keys: str | list[str]) -> None:
        """
        Delete one or more parameters.

        Parameters:
            key_or_keys (Union[str, List[str]]): The parameter name(s) to delete.
        """
        keys_to_delete = [key_or_keys] if isinstance(key_or_keys, str) else key_or_keys
        for k in keys_to_delete:
            if hasattr(self, k):
                delattr(self, k)

    def to_str(self, with_value: bool = True) -> str:
        """
        Convert the parameters to a string, optionally excluding values.

        Parameters:
            with_value (bool): Whether to include the values of the parameters.

        Returns:
            str: A string representation of the parameters.
        """
        if not self.__dict__:
            return "noparams"

        if with_value:
            return self.__str__()  # Uses the sorted version from __str__
        else:
            return "-".join(sorted(self.__dict__.keys()))  # Sort keys for consistency

    def items(self):
        """
        Get all parameter items as a view object.

        Returns:
            ItemsView: A view of the parameter (name, value) items.
        """
        return self.__dict__.items()

    def update(self, other_params: dict[str, Any] | FactorParameters) -> None:
        """
        Update parameters from another dictionary or FactorParameters instance.
        """
        if isinstance(other_params, FactorParameters):
            self.set_parameters(other_params.get_all_parameters())
        elif isinstance(other_params, dict):
            self.set_parameters(other_params)
        else:
            raise TypeError(
                "Input must be a dict or FactorParameters instance to update."
            )


class FactorTemplate(ABC):
    """
    Abstract Base Class for all factor implementations.

    Subclasses must implement `get_output_schema` and `calculate`.
    """

    author: str = "Unknown"  # Author of the factor
    module_factors_lookup = None  # Lookup for module factors, set by FactorEngine

    factor_name: str = (
        ""  # Unique name for the factor, set by subclass or from settings
    )
    freq: Interval | None = None  # Calculation frequency/interval of the factor

    # These are typically not used directly by individual factors if they are symbol/exchange agnostic
    # but can be part of the context if a factor is specific.
    _vt_symbols: list[str] = []
    exchange: Exchange = Exchange.TEST

    # Configurations for dependencies, will be resolved to FactorTemplate instances
    dependencies_factor_config: list[dict[str, Any]] = {}
    dependencies_freq_config: list[Interval] = []  # Or strings
    dependencies_symbol_config: list[str] = []
    dependencies_exchange_config: list[Exchange] = []  # Or strings

    # Actual instances of dependency factors
    dependencies_factor: list["FactorTemplate"] = []  # noqa: UP037

    _factor_mode: FactorMode = FactorMode.LIVE  # Default mode (LIVE, BACKTEST)
    VTSYMBOL_TEMPLATE: str = (
        VTSYMBOL_FACTOR  # Template string for generating factor keys
    )

    def get_output_schema(self) -> dict[str, pl.DataType]:
        """
        Defines the output schema: a datetime column and one Float64 column for each symbol.
        Column names for symbols are the vt_symbol strings themselves.
        """
        schema = {
            DEFAULT_DATETIME_COL: pl.Datetime(
                time_unit="us"
            )
        }
        # Allows schema with only datetime if vt_symbols is empty.
        # FactorMemory might require at least one value column depending on its own logic,
        # but this factor will produce an empty value set for such a schema.
        for symbol in self.vt_symbols:
            schema[symbol] = pl.Float64
        return schema

    @property
    def factor_key(self) -> str:
        """
        Generates a unique key for this factor instance based on its name,
        frequency, and parameters.
        """

        interval_value = self.freq.value if self.freq else "UNKNOWN"  # Handle None freq

        # Using VTSYMBOL_TEMPLATE which expects 'interval' and 'factorname'
        # Example: VTSYMBOL_FACTOR = "factor.{interval}.{factorname}" from vnpy.config
        # If your VTSYMBOL_TEMPLATE is different, adjust accordingly.
        base_key_part = self.VTSYMBOL_TEMPLATE.format(
            interval=interval_value,
            factorname=self.__class__.__name__.lower(),  # Ensure factor name is lowercase
        )
        param_str = self.params.to_str(with_value=True)
        return f"{base_key_part}@{param_str}"

    @property
    def vt_symbols(self) -> list[str]:
        return self._vt_symbols

    @property
    def factor_mode(self) -> FactorMode:
        """
        Returns the current factor mode (LIVE, BACKTEST, etc.).
        """
        return self._factor_mode

    @vt_symbols.setter
    def vt_symbols(self, value: list[str]) -> None:
        if not isinstance(value, list):
            raise TypeError(
                f"vt_symbols must be a list, got {type(value)} for factor {self.factor_key}"
            )

        # Update own _vt_symbols
        self._vt_symbols = value

        # Propagate to dependencies if they exist
        if hasattr(self, "dependencies_factor") and self.dependencies_factor:
            for dep_factor in self.dependencies_factor:
                if isinstance(dep_factor, FactorTemplate):
                    dep_factor.vt_symbols = (
                        value  # This will call the setter on the dependency
                    )
    @factor_mode.setter
    def factor_mode(self, value: FactorMode) -> None:
        """
        Setter for factor_mode. Ensures the value is a valid FactorMode.
        """
        if not isinstance(value, FactorMode):
            raise TypeError(
                f"factor_mode must be a FactorMode instance, got {type(value)}"
            )
        self._factor_mode = value

        # Propagate to dependencies if they exist
        if hasattr(self, "dependencies_factor") and self.dependencies_factor:
            for dep_factor in self.dependencies_factor:
                if isinstance(dep_factor, FactorTemplate):
                    dep_factor.factor_mode = value

    def _init_dependency_instances(self):
        """
        Initializes actual FactorTemplate instances for dependencies based on
        their stored configurations (self.dependencies_factor_config).
        Requires self.module_factors_lookup to be set by the FactorEngine.
        """
        if not self.dependencies_factor_config:
            return

        if not self._dependencies_module_lookup:
            if (
                self.dependencies_factor_config
            ):  # Only an issue if there are dependencies
                raise ValueError(
                    f"Factor '{self.factor_name}' has dependencies but "
                    "dependencies_module_lookup was not provided during initialization."
                )
            return  # No dependencies to init, so it's fine

        self.dependencies_factor = init_factors(
            self._dependencies_module_lookup,
            self.dependencies_factor_config,
            self._dependencies_module_lookup,  # Pass the same module lookup for dependencies
        )

    def __init__(
        self,
        setting: dict | None = None,
        dependencies_module_lookup: object | None = None,
        **kwargs,
    ):
        """
        Initializes the FactorTemplate.

        Parameters:
            setting (Optional[dict]): Settings dictionary for the factor.
                                      Can be the factor's own settings block.
            **kwargs: Additional parameters to set, typically overriding those in `setting`.
        """
        self.params: FactorParameters = FactorParameters()

        # Subclasses should define self.factor_name, or it can be loaded from settings.
        # If self.factor_name is not set by the class, from_setting will try to load it.

        # Load from settings dictionary first
        if setting:
            self.from_setting(setting)  # This loads params, name, freq, dep_configs

        # Then, override with any kwargs provided (e.g., by FactorEngine or direct instantiation)
        # This allows dynamic parameter adjustments.
        if kwargs:
            self.set_params(kwargs)

        self._dependencies_module_lookup = dependencies_module_lookup  # Store it

        # Initialize dependency instances from their configurations.
        # This requires self.module_factors_lookup to be set, usually by FactorEngine.
        self._init_dependency_instances()

        # Internal state
        self.inited: bool = False  # True after on_init is successfully called
        self.trading: bool = False  # True if the factor is active and calculating

        self.on_init()  # Call user-defined initialization logic

        self.run_datetime: str = self.update_datetime()  # Timestamp of last run/update

    def update_datetime(self) -> str:
        """Updates and returns the current datetime as a string."""
        self.run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.run_datetime

    def add_params(
        self, param_names: str | list[str], auto_property: bool = True
    ) -> None:
        """
        Adds parameters to this factor instance. If auto_property is True,
        it attempts to create getter/setter properties on the class for these params.
        """
        if isinstance(param_names, str):
            param_names = [param_names]

        # Add parameters to the params object with a default of None
        current_param_values = {name: None for name in param_names}
        self.params.set_parameters(current_param_values)

        if auto_property:
            for attr_name in param_names:
                # Check if a property with this name already exists on the class
                if not hasattr(self.__class__, attr_name) or not isinstance(
                    getattr(self.__class__, attr_name, None), property
                ):
                    # Define getter and setter dynamically
                    def getter(
                        self_instance, name=attr_name
                    ):  # Use self_instance to avoid cls/self confusion
                        return self_instance.params.get_parameter(name)

                    def setter(self_instance, value, name=attr_name):
                        self_instance.params.set_parameters({name: value})

                    # Attach the property dynamically to the class
                    setattr(self.__class__, attr_name, property(getter, setter))

    def set_params(self, params_dict: dict[str, Any]) -> None:
        """
        Sets multiple parameters for the factor.
        If a parameter is not already known (via add_params or class attribute),
        it will be added dynamically.
        """
        if not params_dict:
            return

        for key, value in params_dict.items():
            # Ensure the parameter is known to self.params and has a property if auto_property was used
            if not hasattr(self.params, key):
                self.add_params(
                    key, auto_property=True
                )  # Default auto_property behavior

            # Set the parameter value. This will use the property setter if one exists on self,
            # otherwise it directly sets on self.params (though add_params should create the property).
            if hasattr(self, key) and isinstance(
                getattr(self.__class__, key, None), property
            ):
                setattr(self, key, value)
            else:  # Fallback if property wasn't created or direct access is intended
                self.params.set_parameters({key: value})

    def get_params(self) -> dict[str, Any]:
        """Returns all current parameters as a dictionary."""
        return self.params.get_all_parameters()

    def get_param(self, key: str, default: Any = None) -> Any:
        """Gets a single parameter's value."""
        return self.params.get_parameter(key, default)

    def on_init(self) -> None:
        """
        Callback when the factor is initialized.
        Subclasses can override this for custom initialization logic.
        Called at the end of FactorTemplate.__init__.
        """
        # Example: self.write_log("Factor initialized.")
        self.inited = True

    def on_start(self) -> None:
        """
        Callback when the factor's calculation process starts.
        Called by FactorEngine. Subclasses can override.
        """
        # Example: self.write_log("Factor calculation started.")
        self.trading = True
        self.update_datetime()

    def on_stop(self) -> None:
        """
        Callback when the factor's calculation process stops.
        Called by FactorEngine. Subclasses can override.
        """
        # Example: self.write_log("Factor calculation stopped.")
        self.trading = False

    @abstractmethod
    def calculate(
        self,
        input_data: dict[str, pl.DataFrame | Delayed],
        memory: FactorMemory,  # Expects a FactorMemory instance
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Abstract method for calculating factor values.

        Parameters:
        ----------
        input_data : Dict[str, Union[pl.DataFrame, Delayed]]
            Input data for calculation.
            - For leaf factors (no factor dependencies): Typically contains DataFrames
              of OHLCV data (e.g., input_data["close"] is a DataFrame).
            - For factors with dependencies: Contains DataFrames which are the
              outputs of its dependency factors (e.g., input_data["dependency_factor_key_1"]).
            These DataFrames are usually the direct result of Dask computations if dependencies exist,
            or cloned DataFrames from FactorEngine's memory_bar.
        memory : FactorMemory
            An instance of FactorMemory associated with this factor.
            - Use `memory.get_data()` to retrieve the current full historical data for this factor.
            - This method should return the NEW complete historical DataFrame for the factor.

        Returns:
        -------
        pl.DataFrame
            The NEW complete historical data for this factor. This DataFrame should
            conform to the schema defined by `get_output_schema()`. It represents
            the full state of the factor's history after this calculation cycle.
            The FactorEngine will then call `memory.update_data()` with this returned DataFrame,
            which handles merging, sorting, de-duplication, truncation to max_rows, and persistence.
        """
        pass

    def from_setting(self, setting: dict | None = None) -> None:
        """
        Loads factor configuration from a settings dictionary.
        This method is called during __init__.
        """
        if setting is None:
            return

        # If setting is {factor_key: actual_settings_dict}, extract actual_settings_dict
        actual_setting_dict = setting
        if (
            len(setting) == 1
            and isinstance(list(setting.values())[0], dict)
            and "class_name" in list(setting.values())[0]
        ):
            actual_setting_dict = list(setting.values())[0]

        # Set factor_name from settings if available and not already set by subclass
        # Subclass definition of self.factor_name takes precedence.
        if "factor_name" in actual_setting_dict:
            self.factor_name = actual_setting_dict["factor_name"]
        elif not self.factor_name and "class_name" in actual_setting_dict:
            # Fallback: use class_name for factor_name if factor_name is not in settings
            self.factor_name = actual_setting_dict["class_name"]
        # If still no factor_name, it must be set by the subclass directly.

        freq_val = actual_setting_dict.get("freq")
        if isinstance(freq_val, str):
            try:
                self.freq = Interval(freq_val)
            except ValueError:  # Handle invalid interval string
                print(
                    f"Warning: Invalid interval string '{freq_val}' in settings for {self.factor_name}. Using UNKNOWN."
                )
                self.freq = Interval.UNKNOWN
        elif isinstance(freq_val, Interval):
            self.freq = freq_val
        # If freq_val is None, self.freq retains its class-defined or previous value.

        # Load parameters from settings; these can be overridden by __init__ kwargs later.
        if "params" in actual_setting_dict:
            self.params.set_parameters(actual_setting_dict["params"])

        # Store dependency *configurations*. Instance creation happens in _init_dependency_instances.
        self.dependencies_factor_config = actual_setting_dict.get(
            "dependencies_factor", []
        )

        # Store other dependency info (freq, symbol, exchange) as configurations
        self.dependencies_freq_config = [
            Interval(f_str) if isinstance(f_str, str) else f_str
            for f_str in actual_setting_dict.get("dependencies_freq", [])
        ]
        self.dependencies_symbol_config = actual_setting_dict.get(
            "dependencies_symbol", []
        )
        self.dependencies_exchange_config = [
            Exchange(e_str) if isinstance(e_str, str) else e_str
            for e_str in actual_setting_dict.get("dependencies_exchange", [])
        ]

        self.run_datetime = actual_setting_dict.get(
            "last_run_datetime", self.update_datetime()
        )

        mode_str = actual_setting_dict.get("factor_mode")
        if isinstance(mode_str, str):
            try:
                self.factor_mode = FactorMode[mode_str.upper()]
            except KeyError:
                print(
                    f"Warning: Invalid factor_mode '{mode_str}' in settings for {self.factor_name}. Using default."
                )
                # self.factor_mode remains its current value (e.g., class default)
        elif isinstance(mode_str, FactorMode):
            self.factor_mode = mode_str

    def to_setting(self) -> dict:
        dep_factor_settings_list = []
        for dep_instance in (
            self.dependencies_factor
        ):  # self.dependencies_factor holds FactorTemplate instances
            if isinstance(dep_instance, FactorTemplate):
                # Get the full settings of the dependency instance
                dep_factor_settings_list.append(dep_instance.to_setting())
            elif isinstance(dep_instance, dict):
                # This case implies dep_instance is already a settings dict (e.g. from config)
                # This might occur if dependencies aren't fully resolved to instances yet
                # or if the stored dependency was just a config.
                # For saving, we prefer actual instance settings if available.
                # If it's a dict, assume it's already a compliant settings dict.
                dep_factor_settings_list.append(dep_instance)

        return {
            "class_name": self.__class__.__name__,
            "factor_name": self.factor_name,
            "factor_key": self.factor_key,
            "freq": str(self.freq.value) if self.freq else Interval.UNKNOWN.value,
            "params": self.params.get_all_parameters(),
            "dependencies_factor": dep_factor_settings_list,
            "dependencies_freq": [
                str(f.value) if isinstance(f, Interval) else f
                for f in self.dependencies_freq_config
            ],
            "dependencies_symbol": self.dependencies_symbol_config,
            "dependencies_exchange": [
                str(e.value) if isinstance(e, Exchange) else e
                for e in self.dependencies_exchange_config
            ],
            "last_run_datetime": self.run_datetime,
            "factor_mode": self.factor_mode.name
            if self.factor_mode
            else FactorMode.LIVE.name,
        }

    def to_dict(self) -> dict:
        """
        Converts the factor to a dictionary keyed by its factor_key,
        with its settings as the value. Standard format for saving multiple factors.
        """
        return {self.factor_key: self.to_setting()}

    def validate_inputs(
        self, input_data: dict[str, Any], memory_instance: FactorMemory
    ) -> bool:
        """
        Optional method to validate inputs before calculation.
        FactorEngine does not call this by default; factor implementations can use it internally.

        Parameters:
        ----------
        input_data : Dict[str, Any]
            The input data that will be passed to calculate().
        memory_instance : FactorMemory
            The FactorMemory instance.

        Returns:
        -------
        bool
            True if inputs are valid, False otherwise.

        Raises:
        ------
        ValueError: If input data is critically invalid.
        """
        if (
            not input_data and not self.dependencies_factor
        ):  # Leaf node should have bar data
            print(f"Warning: Input data is empty for leaf factor {self.factor_key}.")
            # Depending on factor logic, this might be an error or acceptable.
            # raise ValueError("Input data cannot be empty for a leaf factor.")

        # Example: Check if required columns exist in memory.get_data()
        # historical_df = memory_instance.get_data()
        # expected_historical_cols = set(self.get_output_schema().keys())
        # if not historical_df.is_empty() and not expected_historical_cols.issubset(historical_df.columns):
        #     raise ValueError(f"Historical data for {self.factor_key} missing required columns. "
        #                      f"Expected: {expected_historical_cols}, Found: {historical_df.columns}")
        return True

    # --- Optimizer Parameter Path Methods (Modified for simpler paths) ---
    def get_nested_params_for_optimizer(
        self, current_path_prefix: str = ""
    ) -> dict[str, Any]:
        """
        Recursively collects all tunable parameters, using simplified dot-separated
        factor_name (nickname) paths for dependencies.
        Example paths: "param_A", "short_ema.period", "my_macd.fast_ema_of_macd.period"
        """
        nested_params: dict[str, Any] = {}
        own_params = self.get_params()
        for param_name, param_value in own_params.items():
            # For the root factor's params, prefix is empty, so key is just param_name
            key_path = f"{current_path_prefix}{param_name}"
            nested_params[key_path] = param_value

        if hasattr(self, "dependencies_factor") and self.dependencies_factor:
            dep_nicknames_count = {}  # To handle potential non-unique nicknames among siblings
            for dep_factor_instance in self.dependencies_factor:
                if isinstance(dep_factor_instance, FactorTemplate):
                    dep_nickname = dep_factor_instance.factor_name
                    if not dep_nickname:
                        print(
                            f"Warning: Dependency of '{self.factor_key}' (key: {dep_factor_instance.factor_key}) has empty factor_name. Using unique key for path."
                        )
                        dep_nickname = dep_factor_instance.factor_key  # Fallback

                    # Handle potentially non-unique nicknames by appending a counter for path uniqueness
                    base_nickname_for_count = (
                        dep_nickname  # Use the original nickname for counting
                    )
                    current_count = dep_nicknames_count.get(base_nickname_for_count, 0)
                    effective_nickname_for_path = (
                        f"{dep_nickname}_{current_count}"
                        if current_count > 0
                        else dep_nickname
                    )
                    dep_nicknames_count[base_nickname_for_count] = current_count + 1

                    if current_count > 0:  # Log if de-duplication happened
                        print(
                            f"Warning: Duplicate nickname '{dep_nickname}' for dependencies of '{self.factor_key}'. Using '{effective_nickname_for_path}' in optimizer path."
                        )

                    # Construct new prefix: if current_path_prefix is empty, new_prefix is "nickname.",
                    # otherwise it's "parent_prefix.nickname."
                    new_prefix_for_child = (
                        f"{current_path_prefix}{effective_nickname_for_path}."
                    )

                    child_params = dep_factor_instance.get_nested_params_for_optimizer(
                        current_path_prefix=new_prefix_for_child
                    )
                    nested_params.update(child_params)
        return nested_params

    def set_nested_params_for_optimizer(
        self, nested_params_dict: dict[str, Any]
    ) -> None:
        """
        Sets parameters using simplified dot-separated factor_name (nickname) paths.
        Example paths: "param_A", "short_ema.period", "my_macd.fast_ema_of_macd.period"
        """
        own_params_to_set: dict[str, Any] = {}
        # Group parameters for each uniquely identified direct dependency
        deps_params_to_set_grouped: dict[
            str, dict[str, Any]
        ] = {}  # Keyed by effective_nickname_for_path

        for path_key, value in nested_params_dict.items():
            path_parts = path_key.split(".", 1)  # Split only on the first dot

            if (
                len(path_parts) == 1
            ):  # No dot, so it's a parameter for the current (self) factor
                own_params_to_set[path_key] = value
            else:  # Contains at least one dot, meaning it's for a dependency
                # direct_dep_effective_nickname is the first segment of the path
                direct_dep_effective_nickname = path_parts[0]
                remainder_path_for_child = path_parts[
                    1
                ]  # This is the rest of the path for the child

                if direct_dep_effective_nickname not in deps_params_to_set_grouped:
                    deps_params_to_set_grouped[direct_dep_effective_nickname] = {}
                deps_params_to_set_grouped[direct_dep_effective_nickname][
                    remainder_path_for_child
                ] = value

        # Apply parameters to the current factor
        if own_params_to_set:
            self.set_params(own_params_to_set)

        # Apply parameters to direct dependencies
        if hasattr(self, "dependencies_factor") and self.dependencies_factor:
            dep_nicknames_count = {}  # To reconstruct the effective_nickname_for_path used in get_nested_params
            for dep_instance in self.dependencies_factor:
                if isinstance(dep_instance, FactorTemplate):
                    base_nickname = dep_instance.factor_name
                    if not base_nickname:
                        base_nickname = dep_instance.factor_key  # Fallback

                    current_count = dep_nicknames_count.get(base_nickname, 0)
                    effective_nickname_for_path = (
                        f"{base_nickname}_{current_count}"
                        if current_count > 0
                        else base_nickname
                    )
                    dep_nicknames_count[base_nickname] = current_count + 1

                    if effective_nickname_for_path in deps_params_to_set_grouped:
                        params_for_this_dep = deps_params_to_set_grouped[
                            effective_nickname_for_path
                        ]
                        # Recursively call set_nested_params_for_optimizer on the dependency
                        dep_instance.set_nested_params_for_optimizer(
                            params_for_this_dep
                        )
                    # Else: No parameters in nested_params_dict were targeted for this specific dependency instance


# Type variable for FactorTemplate subclasses
TV_FactorTemplate = TypeVar("TV_FactorTemplate", bound=FactorTemplate)
