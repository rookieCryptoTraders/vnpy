from collections.abc import Callable
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from random import random, choice
from time import perf_counter
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.managers import DictProxy
from _collections_abc import dict_keys, dict_values, Iterable

from tqdm import tqdm                                   # type: ignore
from deap import creator, base, tools, algorithms       # type: ignore

# Define type hints for reusable function signatures
OUTPUT_FUNC = Callable[[str], None]
EVALUATE_FUNC = Callable[[dict], dict]
KEY_FUNC = Callable[[tuple], float]


# Create base classes for the genetic algorithm using DEAP's creator.
# "FitnessMax" aims to maximize a single objective fitness value.
# "Individual" is a list-based representation of a solution with a fitness attribute.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Holds the settings for running an optimization task, including
    the parameters to be tested and the target metric to optimize.
    """

    def __init__(self) -> None:
        """Initializes the OptimizationSetting object."""
        self.params: dict[str, list] = {}
        self.target_name: str = ""

    def add_parameter(
        self,
        name: str,
        start: float,
        end: float | None = None,
        step: float | None = None
    ) -> tuple[bool, str]:
        """
        Adds a parameter to the optimization settings. It can be a fixed
        value or a range of values with a defined step.

        Parameters:
        - name: The name of the parameter.
        - start: The starting value or the fixed value of the parameter.
        - end: The end value for a parameter range.
        - step: The increment step for a parameter range.

        Returns:
        - A tuple containing a success flag and a descriptive message.
        """
        # Handle fixed-value parameters
        if end is None or step is None:
            self.params[name] = [start]
            return True, "Fixed parameter added successfully"

        # Validate range parameters
        if start >= end:
            return False, "Parameter start value must be less than its end value"

        if step <= 0:
            return False, "Parameter step must be greater than 0"

        # Generate the list of values for the parameter range
        value: float = start
        value_list: list[float] = []
        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

        return True, f"Range parameter added successfully, count: {len(value_list)}"

    def set_target(self, target_name: str) -> None:
        """
        Sets the name of the target metric to be optimized.

        Parameters:
        - target_name: The name of the key in the evaluation result dict
                       that represents the optimization target.
        """
        self.target_name = target_name

    def generate_settings(self) -> list[dict]:
        """
        Generates a list of all possible parameter combinations (settings)
        from the configured parameters.

        Returns:
        - A list of dictionaries, where each dictionary is a unique
          parameter combination.
        """
        if not self.params:
            return []

        keys: dict_keys = self.params.keys()
        values: dict_values = self.params.values()

        # Create the Cartesian product of all parameter values
        products: list = list(product(*values))

        settings: list = []
        for p in products:
            setting: dict = dict(zip(keys, p, strict=False))
            settings.append(setting)

        return settings


def check_optimization_setting(
    optimization_setting: OptimizationSetting,
    output: OUTPUT_FUNC = print
) -> bool:
    """
    Validates the optimization setting to ensure it's runnable.

    Parameters:
    - optimization_setting: The settings object to validate.
    - output: A function to print feedback messages.

    Returns:
    - True if the settings are valid, False otherwise.
    """
    if not optimization_setting.generate_settings():
        output("Parameter combination is empty, please check settings")
        return False

    if not optimization_setting.target_name:
        output("Optimization target not set, please check settings")
        return False

    return True


def run_bf_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
    output: OUTPUT_FUNC = print
) -> list[tuple]:
    """
    Runs a brute-force optimization by evaluating every possible
    parameter combination in parallel.

    Parameters:
    - evaluate_func: The function that evaluates a single parameter setting.
    - optimization_setting: The object containing the optimization parameters.
    - key_func: A function to extract the target value from the evaluation result.
    - max_workers: The maximum number of processes to use.
    - output: A function to print progress and results.

    Returns:
    - A sorted list of results.
    """
    settings: list[dict] = optimization_setting.generate_settings()

    output("Starting brute-force optimization")
    output(f"Parameter optimization space: {len(settings)}")

    start: float = perf_counter()

    with ProcessPoolExecutor(max_workers, mp_context=get_context("spawn")) as executor:
        # Use tqdm to display a progress bar
        iterator: Iterable = tqdm(
            executor.map(evaluate_func, settings),
            total=len(settings)
        )
        results: list[tuple] = list(iterator)
        results.sort(reverse=True, key=key_func)

        end: float = perf_counter()
        cost: int = int(end - start)
        output(f"Brute-force optimization finished, cost: {cost} seconds")

        return results


def run_ga_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
    population_size: int = 100,
    ngen_size: int = 30,
    output: OUTPUT_FUNC = print
) -> list[tuple]:
    """
    Runs a genetic algorithm optimization.

    Parameters:
    - evaluate_func: The function that evaluates a single parameter setting.
    - optimization_setting: The object containing the optimization parameters.
    - key_func: A function to extract the target value from the evaluation result.
    - max_workers: The maximum number of processes to use.
    - population_size: The number of individuals in each generation.
    - ngen_size: The number of generations to run.
    - output: A function to print progress and results.

    Returns:
    - A sorted list of results from the cache.
    """
    # Generate all possible settings to be used for initial population and mutations
    all_settings_list: list[dict] = optimization_setting.generate_settings()
    settings: list[list[tuple]] = [list(d.items()) for d in all_settings_list]

    def generate_parameter() -> list:
        """Randomly selects one full parameter set."""
        return choice(settings)

    def mutate_individual(individual: list, indpb: float) -> tuple:
        """
        Mutation function that replaces genes with values from a new,
        randomly chosen parameter set.
        """
        size: int = len(individual)
        paramlist: list = generate_parameter()
        for i in range(size):
            if random() < indpb:
                individual[i] = paramlist[i]
        return individual,

    # Set up multiprocessing Pool and Manager
    ctx: BaseContext = get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(max_workers) as pool:
        # Create a shared dict for caching results to avoid re-evaluating
        # the same parameter set.
        cache: DictProxy[tuple, tuple] = manager.dict()

        # Set up the DEAP toolbox with registered genetic operators
        toolbox: base.Toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=1.0)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("map", pool.map)
        toolbox.register(
            "evaluate",
            ga_evaluate,
            cache,
            evaluate_func,
            key_func
        )

        total_size: int = len(settings)
        pop_size: int = population_size
        lambda_: int = pop_size
        mu: int = int(pop_size * 0.8)

        cxpb: float = 0.95
        mutpb: float = 1.0 - cxpb
        ngen: int = ngen_size

        pop: list = toolbox.population(n_=pop_size)

        # Print optimization settings
        output("Starting genetic algorithm optimization")
        output(f"Parameter optimization space: {total_size}")
        output(f"Population size per generation: {pop_size}")
        output(f"Number of individuals to select for next generation: {mu}")
        output(f"Number of generations: {ngen}")
        output(f"Crossover probability: {cxpb:.0%}")
        output(f"Mutation probability: {mutpb:.0%}")

        start: float = perf_counter()

        # Run the genetic algorithm
        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            verbose=True
        )

        end: float = perf_counter()
        cost: int = int(end - start)

        output(f"Genetic algorithm optimization finished, cost: {cost} seconds")

        # Sort and return all evaluated results from the cache
        results: list = list(cache.values())
        results.sort(reverse=True, key=key_func)
        return results


def ga_evaluate(
    cache: dict,
    evaluate_func: Callable,
    key_func: Callable,
    parameters: list
) -> tuple[float, ]:
    """
    A wrapper for the evaluation function used within the genetic algorithm.
    It checks a shared cache for existing results before running the
    evaluation to save computation time.
    """
    # Convert parameter list to a tuple to use it as a dictionary key
    param_tuple: tuple = tuple(parameters)

    if param_tuple in cache:
        result: dict = cache[param_tuple]
    else:
        setting: dict = dict(parameters)
        result = evaluate_func(setting)
        cache[param_tuple] = result

    # Extract the fitness value using the provided key function
    value: float = key_func(result)
    return (value,)
