# Utility functions for factor management

import copy
import re
import typing  # Added import
from types import ModuleType
from typing import Any

if typing.TYPE_CHECKING:
    from vnpy.factor.template import (
        FactorTemplate,  # Import for type hinting, made conditional
    )

from vnpy.trader.utility import load_json, save_json


def get_factor_class(module_to_search: ModuleType, class_name: str) -> type:
    """
    Retrieves a factor class by its name from a specified module object.

    Parameters:
        module_to_search (ModuleType): The Python module object in which to find the class.
        class_name (str): Name of the class to load.

    Returns:
        The factor class object (Type).

    Raises:
        AttributeError: If the class_name is not found in the module_to_search.
    """
    try:
        return getattr(module_to_search, class_name)
    except AttributeError as e:
        error_msg = (
            f"Factor class '{class_name}' not found in module "
            f"'{module_to_search.__name__}'."
        )
        # You might log this error or let the caller handle it.
        # Re-raising with more context is often good.
        raise AttributeError(error_msg) from e


def save_factor_setting(
    settings_list_to_save: list[dict[str, Any]], setting_filename: str
) -> None:
    """
    Saves a list of factor settings dictionaries to a JSON file.
    """
    save_json(setting_filename, settings_list_to_save)


def load_factor_setting(setting_file_name: str) -> list[dict[str, Any]]:
    """
    Loads a list of factor settings dictionaries from a JSON file.
    """
    loaded_data = load_json(setting_file_name)
    if not isinstance(loaded_data, list):
        # Optional: Add handling for old dictionary format for backward compatibility
        # For now, enforce the new list format.
        raise TypeError(
            f"Factor settings file '{setting_file_name}' is not in the expected list format. "
            f"Expected a list of factor settings, got {type(loaded_data)}."
        )
    # Further validation: check if all items in the list are dicts
    if not all(isinstance(item, dict) for item in loaded_data):
        raise ValueError(
            f"Factor settings file '{setting_file_name}' contains non-dictionary items in the list."
        )
    return loaded_data


def init_factors(
    module_for_primary_classes: ModuleType,
    settings_data: list[dict[str, Any]],  # THIS IS NOW A LIST OF ACTUAL SETTINGS DICTS
    dependencies_module_lookup_for_instances: ModuleType,
    vt_symbols: list[str] | str = None,
) -> list["FactorTemplate"]:  # Updated return type hint to string literal
    initialized_factors: list[
        FactorTemplate
    ] = []  # Explicitly type initialized_factors to string literal

    if not isinstance(settings_data, list):  # Should be caught by load_factor_setting
        raise TypeError(
            f"init_factors expected settings_data to be a list, got {type(settings_data)}"
        )

    for actual_factor_settings in settings_data:  # Iterate directly over settings dicts
        if not isinstance(actual_factor_settings, dict):
            print(
                f"[FactorUtils] Warning: Expected a dict for factor settings, got {type(actual_factor_settings)}. Skipping."
            )
            continue

        class_name = actual_factor_settings.get("class_name")
        if not class_name:
            print(
                f"[FactorUtils] Warning: 'class_name' not found in factor settings: {actual_factor_settings}. Skipping."
            )
            continue

        try:
            FactorClass = get_factor_class(module_for_primary_classes, class_name)
        except AttributeError as e:
            print(f"[FactorUtils] Error initializing factor: {str(e)} Skipping.")
            continue

        instance = FactorClass(
            setting=actual_factor_settings,
            dependencies_module_lookup=dependencies_module_lookup_for_instances,
        )
        if vt_symbols is not None:
            if isinstance(vt_symbols, str):
                vt_symbols = [vt_symbols]
            instance.vt_symbols = vt_symbols
        initialized_factors.append(instance)
    return initialized_factors


def apply_params_to_definition_dict(
    definition_dict: dict[str, Any], params_with_paths: dict[str, Any]
) -> dict[str, Any]:
    """
    Applies a flat dictionary of path-based parameters to a factor definition dictionary.
    Modifies and returns a deep copy of the original definition_dict.
    Path keys are like "param_name" for root, "dependencies_factor[0].param_name" for a
    direct dependency's param, or "dependencies_factor[0].dependencies_factor[1].param_name" for nested.

    Args:
        definition_dict: The factor definition dictionary (JSON-like structure).
        params_with_paths: Flat dictionary with path-based keys and values to set.

    Returns:
        A new definition dictionary with updated parameters.
    """
    if not params_with_paths:
        return copy.deepcopy(definition_dict)

    new_def_dict = copy.deepcopy(definition_dict)

    for path_key, value_to_set in params_with_paths.items():
        path_parts = path_key.split(".")
        param_name_for_target_level = path_parts[-1]
        traversal_path_segments = path_parts[:-1]

        current_target_for_traversal = (
            new_def_dict  # Correctly re-initialize for each path
        )
        valid_path_so_far = True

        for segment in traversal_path_segments:
            dep_match = re.fullmatch(r"dependencies_factor\[(\d+)\]", segment)
            if dep_match:
                dep_index = int(dep_match.group(1))
                if (
                    "dependencies_factor" in current_target_for_traversal
                    and isinstance(
                        current_target_for_traversal["dependencies_factor"], list
                    )
                    and 0
                    <= dep_index
                    < len(current_target_for_traversal["dependencies_factor"])
                    and isinstance(
                        current_target_for_traversal["dependencies_factor"][dep_index],
                        dict,
                    )
                ):
                    current_target_for_traversal = current_target_for_traversal[
                        "dependencies_factor"
                    ][dep_index]
                else:
                    # print(f"Warning: Invalid path segment '{segment}' in '{path_key}' during traversal.") # Optional: Add logging
                    valid_path_so_far = False
                    break
            else:
                # print(f"Warning: Unrecognized segment format '{segment}' in '{path_key}'.") # Optional: Add logging
                valid_path_so_far = False
                break

        if valid_path_so_far:
            if "params" not in current_target_for_traversal or not isinstance(
                current_target_for_traversal["params"], dict
            ):
                current_target_for_traversal["params"] = {}
            current_target_for_traversal["params"][param_name_for_target_level] = (
                value_to_set
            )
        # else: print(f"Warning: Could not apply parameter for path '{path_key}' due to invalid path.") # Optional: Add logging

    return new_def_dict


def apply_params_to_definition_dict_nickname_paths(
    definition_dict: dict[str, Any], params_with_paths: dict[str, Any]
) -> dict[str, Any]:
    """
    Applies a flat dictionary of path-based parameters to a factor definition dictionary.
    This version uses dot-separated 'factor_name' (nickname) paths.
    Example paths: "param_A", "short_ema.period", "my_macd.fast_ema_of_macd.period"
    Modifies and returns a deep copy of the original definition_dict.

    Args:
        definition_dict: The factor definition dictionary (JSON-like structure).
        params_with_paths: Flat dictionary with nickname-based path keys and values to set.

    Returns:
        A new definition dictionary with updated parameters.
    """
    if not params_with_paths:
        return copy.deepcopy(definition_dict)

    new_def_dict = copy.deepcopy(definition_dict)

    for path_key, value_to_set in params_with_paths.items():
        path_parts = path_key.split(".")
        param_name_to_set = path_parts[-1]
        nickname_segments_in_path = path_parts[
            :-1
        ]  # List of nicknames forming the path to the target factor dict

        current_level_dict_being_updated = (
            new_def_dict  # Start traversal from the root definition
        )
        valid_path_for_this_key = True

        # This tracks occurrences of base nicknames at the current dependency level being searched
        # to match paths like "nickname_0", "nickname_1" if get_nested_params added them.

        for path_segment_nickname_with_suffix in nickname_segments_in_path:
            # path_segment_nickname_with_suffix could be "baseNickname" or "baseNickname_count"
            # (e.g., "short_ema" or "short_ema_0" if de-duplication was applied by get_nested_params)

            target_base_nickname = path_segment_nickname_with_suffix
            target_occurrence_index = (
                0  # If no suffix, we are looking for the first (0-th) occurrence
            )

            suffix_match = re.fullmatch(
                r"(.+)_(\d+)", path_segment_nickname_with_suffix
            )
            if suffix_match:
                target_base_nickname = suffix_match.group(1)
                target_occurrence_index = int(suffix_match.group(2))

            found_next_level_dict = None
            current_occurrence_count_for_base_nickname = (
                0  # Counter for actual deps with this base nickname
            )

            if (
                "dependencies_factor" not in current_level_dict_being_updated
                or not isinstance(
                    current_level_dict_being_updated["dependencies_factor"], list
                )
            ):
                print(
                    f"Warning (apply_params): Path key '{path_key}' - 'dependencies_factor' "
                    f"not found or not a list at segment '{path_segment_nickname_with_suffix}'."
                )
                valid_path_for_this_key = False
                break

            for dependency_config_dict in current_level_dict_being_updated[
                "dependencies_factor"
            ]:
                if not isinstance(dependency_config_dict, dict):
                    continue

                # The 'factor_name' in the config dict is the original nickname.
                original_dependency_nickname = dependency_config_dict.get("factor_name")
                if not original_dependency_nickname:
                    # Fallback if dependency config has no 'factor_name'.
                    # This needs to align with how get_nested_params_for_optimizer handles empty nicknames
                    # (e.g., if it used factor_key as a fallback nickname in path generation).
                    # If get_nested_params used factor_key, then target_base_nickname here would be a factor_key.
                    original_dependency_nickname = dependency_config_dict.get(
                        "factor_key", ""
                    )  # Example fallback

                if original_dependency_nickname == target_base_nickname:
                    if (
                        current_occurrence_count_for_base_nickname
                        == target_occurrence_index
                    ):
                        found_next_level_dict = dependency_config_dict
                        break
                    current_occurrence_count_for_base_nickname += 1

            if found_next_level_dict:
                current_level_dict_being_updated = found_next_level_dict
            else:
                print(
                    f"Warning (apply_params): Path key '{path_key}' - could not find dependency matching "
                    f"path segment '{path_segment_nickname_with_suffix}' (parsed base: '{target_base_nickname}', target occurrence: {target_occurrence_index})."
                )
                valid_path_for_this_key = False
                break

        if valid_path_for_this_key:
            # Ensure the 'params' dictionary exists at the target level
            if "params" not in current_level_dict_being_updated or not isinstance(
                current_level_dict_being_updated["params"], dict
            ):
                current_level_dict_being_updated["params"] = {}
            current_level_dict_being_updated["params"][param_name_to_set] = value_to_set
        # else: parameter for this path_key was not set due to invalid path or mismatch.

    return new_def_dict
