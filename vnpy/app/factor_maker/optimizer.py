import itertools
from typing import Dict, List, Any

import dask
import numpy as np
import pandas as pd
import polars as pl

from vnpy.app.factor_maker.backtesting import FactorBacktester
from vnpy.app.factor_maker.template import FactorTemplate


class FactorOptimizer:
    """
    A class to optimize factor parameters using backtesting.
    """

    def __init__(self, backtester: FactorBacktester, data: Dict[str, pd.DataFrame]):
        """
        Initialize the optimizer.

        Parameters:
            backtester: The backtester class to use for evaluation.
            data: Historical data dictionary with keys ('open', 'high', 'low', 'close', 'volume').
        """
        self.dependency_factor = {}
        self.factor = None

        self.backtester = backtester
        self.data = data

    def add_factor(self, factor: type[FactorTemplate]):
        self.factor = factor
        self.dependency_factor = {f.factor_key: f for f in factor.dependencies_factor}

        self.build_computational_graph()

        self.prepare_data()

    def build_computational_graph(self):
        def complete_factor_tree():
            dependency_factor = self.dependency_factor.copy()
            for f_key, f in self.dependency_factor.items():
                for dep_f in f.dependencies_factor:
                    if dep_f.factor_key in dependency_factor:
                        continue
                    dependency_factor[dep_f.factor_key] = dep_f
            self.dependency_factor = dependency_factor

        complete_factor_tree()

        self.tasks = {}

        # Function to create a task for a factor
        def create_task(factor_key: str) -> dask.delayed:
            """
            Create a Dask task for a given factor dynamically.

            Parameters:
                factor_key (str): The name of the factor to create the task for.

            Returns:
                dask.delayed: The Dask task for the factor calculation.
            """
            # Check if the task has already been created
            if factor_key in self.tasks:
                return self.tasks[factor_key]

            # Retrieve the factor instance
            factor = self.dependency_factor[factor_key]

            # Resolve dependencies recursively
            dep_tasks = {}
            if not factor.dependencies_factor:
                # Create memory dict with delayed tasks for each key
                memory_dict = {key: dask.delayed(lambda df=df: df.clone())() for key, df in self.data.items()}
                # Pass the memory_dict as input to factor.calculate
                self.tasks[factor_key] = dask.delayed(factor.calculate)(input_data=memory_dict)
            else:
                # Resolve dependencies recursively
                for f in factor.dependencies_factor:
                    dep = f.factor_key
                    dep_tasks[dep] = create_task(dep)

                # Create the task for the current factor using memory and resolved dependencies
                self.tasks[factor_key] = dask.delayed(factor.calculate)(input_data=dep_tasks)

            return self.tasks[factor_key]

        # Build tasks for all factors
        for factor_key in self.dependency_factor.keys():
            create_task(factor_key)

    def prepare_data(self):
        """
        Compute the results of the Dask DAGs and extract the outputs into a structured format.
        """
        if not hasattr(self, "tasks"):
            raise ValueError("Computation graph has not been built. Please run build_computational_graph first.")

        # Execute the Dask computation graph
        computed_results = dask.compute(*self.tasks.values())

        # Map computed results back to their corresponding factors
        self.factor_data = {
            factor_key: result for factor_key, result in zip(self.tasks.keys(), computed_results)
        }

        return self.factor_data

    def optimize(self, param_grid: Dict[str, List[Any]], metric: str = "Sharpe Ratio") -> Dict[str, Any]:
        """
        Optimize the params of the factor using grid search and Dask.

        Parameters:
            param_grid (Dict[str, List[Any]]): Dictionary of parameter names and their grid values.
            metric (str): The performance metric to optimize. Default is "Sharpe Ratio".

        Returns:
            Dict[str, Any]: Best params and their corresponding performance metrics.
        """

        # Generate all combinations of params
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        # Define evaluate_params outside the loop
        @dask.delayed
        def evaluate_params(params):
            # Set params for the factor
            self.factor.set_params(params)

            # Calculate factor values
            factor_values = self.factor.calculate(self.factor_data)

            # Run backtesting
            metrics = self.backtester.run_backtesting_pandas(factor_values, if_plot=False)

            # Return the chosen metric and params
            return metrics.get(metric, -float("inf")), params

        # Create tasks for all parameter combinations
        tasks = [evaluate_params(dict(zip(param_names, param_values))) for param_values in param_combinations]

        # Compute all tasks
        results = dask.compute(*tasks)

        # Find the best parameter set
        best_score = -float("inf")
        best_params = None
        for score, params in results:
            if score > best_score:
                best_score = score
                best_params = params

        self.factor.set_params(best_params)
        self.factor.to_dict()

        return {"best_params": best_params, "best_score": best_score}


if __name__ == "__main__":
    # Load historical data
    def load_data() -> Dict[str, pd.DataFrame]:
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="B")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        close_prices = pd.DataFrame(
            data=np.random.uniform(100, 200, size=(len(dates), len(symbols))),
            index=dates,
            columns=symbols,
        )
        return {"close": close_prices}


    data = load_data()

    class MACD(FactorTemplate):
        factor_name = 'macd'
        dependencies_factor = []

        def __init__(self, setting, **kwargs):
            self.add_params('window')
            self.add_params('freq')
            super().__init__(setting, **kwargs)

        def __init_dependencies__(self, ma10_setting, ma20_setting):
            self.ma_fast = MA(ma10_setting)
            


        def calculate_polars(self, input_data: pl.DataFrame, *args, **kwargs) -> Any:
            ma_fast = input_data[self.ma_fast.factor_key]

    macd = MACD({}, window=10, freq='1min')

    input_data = {
        'ma_window_10', df1,
        'ma_window_20', df2
    }

    ma10_setting = {}
    ma20_setting = {}
    ma10 = MA(ma10_setting)
    ma20 = MA(ma20_setting)

    macd.dependencies_factor = [ma20, ma10]

    macd.to_dict()

