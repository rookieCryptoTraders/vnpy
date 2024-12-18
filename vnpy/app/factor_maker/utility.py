import json

from vnpy.trader.utility import load_json, save_json
import factors

from typing import Dict, Any
import dask


def get_factor_class(class_name: str) -> None:
    """Load a factor class from a specified module."""
    return getattr(factors, class_name)


def save_factor_setting(self) -> None:
    """������������"""
    factor_setting: dict = {}
    for name, factor in self.factors.items():
        factor_setting[name] = factor.to_dict()
    save_json(self.setting_filename, factor_setting)


# mycode
def load_factor_setting(setting_path: str):
    """load_factor_setting from a json file, which is usually saved in `.vntrader/factor_maker_setting.json`

    Parameters
    ----------
    setting_path :

    Returns
    -------

    """
    with open(setting_path, 'r') as fr:
        js = json.load(fr)
    return js


def init_factors(f_setting: Dict):
    """initiate factors from the factor setting with a for loop, but the factor will init dependency factors recursively

    Parameters
    ----------
    f_setting : Dict
        factor settings saved in json file

    Returns
    -------

    """
    factors_list = []
    for module_name, module_setting in f_setting.items():
        f_class = getattr(factors, module_setting["class_name"])
        f_class = f_class({module_name: module_setting}, **module_setting["params"])  # recursion
        factors_list.append(f_class)
    return factors_list


def build_computational_graph(factors: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, dask.delayed]:
    """
    Build a Dask computational graph for factors with dependencies.

    Parameters:
        factors (Dict[str, Any]): Dictionary of factors keyed by their unique `factor_key`.
        data (Dict[str, pl.DataFrame]): Historical OHLCV or other input data.

    Returns:
        Dict[str, dask.delayed]: Dask tasks for each factor keyed by `factor_key`.
    """
    # Step 1: Complete the dependency tree
    full_factor_tree = complete_factor_tree(factors)

    # Step 2: Create tasks for all factors
    tasks = {}
    for factor_key in full_factor_tree.keys():
        create_task(factor_key, full_factor_tree, data, tasks)

    return tasks


def complete_factor_tree(factors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively complete the dependency tree for all factors.

    Parameters:
        factors (Dict[str, Any]): Dictionary of factors keyed by factor_key.

    Returns:
        Dict[str, Any]: Complete dependency tree including all dependent factors.
    """

    def traverse_dependencies(factor: Any, resolved_factors: Dict[str, Any]) -> None:
        """
        Recursively traverse and add dependencies of the given factor.

        Parameters:
            factor (Any): Current factor to traverse.
            resolved_factors (Dict[str, Any]): Dictionary to store all resolved dependencies.
        """
        if factor.factor_key in resolved_factors:
            return  # Skip if already resolved

        # Add the current factor
        resolved_factors[factor.factor_key] = factor

        # Traverse dependencies recursively
        for dependency in factor.dependencies_factor:
            traverse_dependencies(dependency, resolved_factors)

    resolved_factors = {}
    for factor_key, factor in factors.items():
        traverse_dependencies(factor, resolved_factors)

    return resolved_factors


def create_task(
        factor_key: str,
        factors: Dict[str, Any],
        data: Dict[str, Any],
        tasks: Dict[str, dask.delayed]
) -> dask.delayed:
    """
    Recursively create a Dask task for a given factor and its dependencies.

    Parameters:
        factor_key (str): The key of the factor to create a task for.
        factors (Dict[str, Any]): Dictionary of all factors keyed by factor_key.
        data (Dict[str, Any]): Historical OHLCV or other input data.
        tasks (Dict[str, dask.delayed]): Dictionary to store the created tasks.

    Returns:
        dask.delayed: The Dask task for the factor calculation.
    """
    if factor_key in tasks:
        return tasks[factor_key]  # Return cached task if already created

    factor = factors[factor_key]

    # Handle dependencies
    if not factor.dependencies_factor:  # No dependencies
        memory_dict = {key: dask.delayed(lambda df=df: df.clone())() for key, df in data.items()}
        tasks[factor_key] = dask.delayed(factor.calculate)(input_data=memory_dict)
    else:  # Resolve dependencies recursively
        dep_tasks = {
            dep_factor.factor_key: create_task(dep_factor.factor_key, factors, data, tasks)
            for dep_factor in factor.dependencies_factor
        }
        tasks[factor_key] = dask.delayed(factor.calculate)(input_data=dep_tasks)

    return tasks[factor_key]


if __name__ == '__main__':
    setting = load_factor_setting(
        '/Users/hongyifan/Desktop/work/crypto/20240720/examples/no_ui/.vntrader/factor_maker_setting.json')
    factors_list = init_factors(setting)
    print(factors_list)
