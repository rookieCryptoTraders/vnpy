import importlib
from logging import INFO, DEBUG, WARNING, ERROR
from typing import Any

from vnpy.factor.template import FactorTemplate
from vnpy.factor.base import (
    APP_NAME,
    FactorMode,
)
from vnpy.factor.utils.factor_utils import (
    init_factors,
    load_factor_setting,
)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(APP_NAME + "_FactorInitializer")


class FactorInitializer:
    """
    Handles the initialization of a factor and its dependency tree.
    It can initialize a factor from a FactorTemplate instance, a dictionary configuration,
    or a factor key string.
    """
    engine_name = APP_NAME + "FactorInitializer"

    def __init__(
        self,
        factor_module_name: str,
        factor_json_conf_path: str | None = None,
    ):
        self.factor_module_name: str = factor_module_name
        self.factor_json_conf_path: str | None = factor_json_conf_path
        self.module_factors: Any | None = None
        try:
            self.module_factors = importlib.import_module(self.factor_module_name)
            self._write_log(
                f"Successfully imported factor module: '{self.factor_module_name}'",
                level=DEBUG,
            )
        except ImportError as e:
            self._write_log(
                f"Could not import factor module '{self.factor_module_name}': {e}. Factor initialization will fail.",
                level=ERROR,
            )
            raise

    def init_and_flatten(
        self,
        factor_definition: FactorTemplate | dict | str,
        vt_symbols_for_factor: list[str],
    ) -> tuple[FactorTemplate | None, dict[str, FactorTemplate] | None]:
        """
        Initializes the target factor and flattens its dependency tree.
        Returns the target factor instance and the dictionary of all flattened factors.
        """
        self._write_log(
            f"Initializing factor graph for symbols: {vt_symbols_for_factor}",
            level=INFO,
        )
        if not self.module_factors:
            self._write_log(
                "Factor module not loaded. Cannot initialize factor.", level=ERROR
            )
            return None, None

        target_factor_instance: FactorTemplate | None = None

        # Logic to get a single FactorTemplate instance based on factor_definition
        if isinstance(factor_definition, FactorTemplate):
            target_factor_instance = factor_definition
            target_factor_instance.factor_mode = FactorMode.BACKTEST
            # Ensure vt_symbols are aligned if provided
            if target_factor_instance.vt_symbols != vt_symbols_for_factor:
                target_factor_instance.vt_symbols = vt_symbols_for_factor
                target_factor_instance._init_dependency_instances()  # Re-initialize dependencies with new symbols
        elif isinstance(factor_definition, dict):
            setting_copy = factor_definition.copy()
            setting_copy["factor_mode"] = FactorMode.BACKTEST.name
            if "params" not in setting_copy:
                setting_copy["params"] = {}
            # setting_copy["params"]["vt_symbols"] = vt_symbols_for_factor
            try:
                inited_list = init_factors(
                    self.module_factors,
                    [setting_copy],
                    self.module_factors,
                    vt_symbols_for_factor,
                )
                if inited_list:
                    target_factor_instance = inited_list[0]
            except Exception as e:
                self._write_log(
                    f"Error initializing factor from dict: {e}", level=ERROR
                )
                return None, None
        elif isinstance(factor_definition, str):  # factor_key
            if not self.factor_json_conf_path:
                self._write_log(
                    "factor_json_conf_path needed for factor_key definition.",
                    level=ERROR,
                )
                return None, None
            try:
                all_settings = load_factor_setting(str(self.factor_json_conf_path))
                found_setting = None
                for setting_item_outer in all_settings:
                    setting_item = setting_item_outer
                    if (
                        len(setting_item_outer) == 1
                        and isinstance(list(setting_item_outer.values())[0], dict)
                        and "class_name" in list(setting_item_outer.values())[0]
                    ):
                        setting_item = list(setting_item_outer.values())[0]

                    temp_setting = setting_item.copy()
                    temp_setting["factor_mode"] = FactorMode.BACKTEST.name
                    if "params" not in temp_setting:
                        temp_setting["params"] = {}

                    TempFactorClass: FactorTemplate = getattr(
                        self.module_factors, temp_setting["class_name"]
                    )
                    temp_instance = TempFactorClass(
                        setting=temp_setting,
                        dependencies_module_lookup=self.module_factors,
                    )
                    temp_instance.vt_symbols = (
                        vt_symbols_for_factor  # Ensure symbols are set
                    )
                    if temp_instance.factor_key == factor_definition:
                        found_setting = setting_item.copy()
                        break
                if found_setting:
                    final_setting = found_setting
                    final_setting["factor_mode"] = FactorMode.BACKTEST.name
                    if "params" not in final_setting:
                        final_setting["params"] = {}
                    inited_list = init_factors(
                        self.module_factors,
                        [final_setting],
                        self.module_factors,
                        vt_symbols_for_factor,
                    )
                    if inited_list:
                        target_factor_instance = inited_list[0]
                else:
                    self._write_log(
                        f"Factor key '{factor_definition}' not found in '{self.factor_json_conf_path}'.",
                        level=ERROR,
                    )
                    return None, None
            except Exception as e:
                self._write_log(
                    f"Error initializing factor from key '{factor_definition}': {e}",
                    level=ERROR,
                )
                return None, None
        else:
            self._write_log(
                f"Invalid factor_definition type: {type(factor_definition)}",
                level=ERROR,
            )
            return None, None

        if not target_factor_instance:
            self._write_log("Failed to create target_factor_instance.", level=ERROR)
            return None, None

        self._write_log(
            f"Target factor instance created: {target_factor_instance.factor_key}",
            level=DEBUG,
        )

        # Flatten the dependency tree
        stacked_factors = {target_factor_instance.factor_key: target_factor_instance}
        flattened_factors = self._complete_factor_tree(stacked_factors)

        if not flattened_factors:
            self._write_log("Failed to flatten factor tree.", level=ERROR)
            return None, None

        self._write_log(
            f"Factor tree flattened. Total factors in graph: {len(flattened_factors)}",
            level=DEBUG,
        )
        return target_factor_instance, flattened_factors

    def _complete_factor_tree(
        self, root_factors: dict[str, FactorTemplate]
    ) -> dict[str, FactorTemplate]:
        """
        Helper method to recursively traverse dependencies and build a flat dictionary
        of all unique FactorTemplate instances.
        """
        resolved_factors: dict[str, FactorTemplate] = {}

        def traverse(factor: FactorTemplate):
            if factor.factor_key in resolved_factors:
                return
            for dep_instance in (
                factor.dependencies_factor
            ):  # These are already FactorTemplate instances
                if not isinstance(dep_instance, FactorTemplate):
                    # This case should ideally not be hit if init_factors correctly resolves dependencies.
                    self._write_log(
                        f"Warning: Dependency for {factor.factor_key} is not a FactorTemplate instance: {type(dep_instance)}. This might indicate an issue in how FactorTemplate initializes its dependencies_factor list.",
                        level=WARNING,
                    )
                    continue
                traverse(dep_instance)
            resolved_factors[factor.factor_key] = factor

        for _, factor_instance in root_factors.items():
            traverse(factor_instance)
        return resolved_factors

    def _write_log(self, msg: str, level: int = INFO) -> None:
        level_map = {
            DEBUG: logger.debug,
            INFO: logger.info,
            WARNING: logger.warning,
            ERROR: logger.error,
        }
        log_func = level_map.get(level, logger.info)
        log_func(msg, gateway_name=self.engine_name)
