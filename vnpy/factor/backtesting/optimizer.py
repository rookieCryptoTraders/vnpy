import os
os.environ['POLARS_TIME_ZONE'] = 'UTC'

import copy
import itertools
from datetime import datetime
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
from typing import Any
from collections.abc import Callable
from collections.abc import Iterator

import polars as pl
from bayes_opt import BayesianOptimization

from loguru import logger

from vnpy.factor.backtesting.backtesting import BacktestEngine
from vnpy.factor.backtesting.factor_analyzer import (
    FactorAnalyser,
    get_annualization_factor,
)
from vnpy.factor.template import FactorTemplate
from vnpy.factor.utils.factor_utils import apply_params_to_definition_dict
from vnpy.trader.constant import Interval

APP_NAME = "FactorOptimizer"
DEFAULT_DATETIME_COL = "datetime"


class FactorOptimizer:
    """
    Optimizes factor parameters using grid search or Bayesian optimization.

    This optimizer loads a full dataset, splits it into training and testing
    sets, and then uses the specified optimization method to find the best
    parameter combination on the training data. The best parameters are then
    validated on the out-of-sample test data.
    """

    engine_name = APP_NAME

    def __init__(self, backtest_engine: BacktestEngine):
        """Initializes the FactorOptimizer with a BacktestEngine instance."""
        self.backtest_engine = backtest_engine
        self._write_log(f"{self.engine_name} initialized.", level=INFO)

    def _prepare_data(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        data_interval: Interval,
        vt_symbols: list[str],
        test_size_ratio: float,
    ) -> tuple[dict[str, pl.DataFrame] | None, dict[str, pl.DataFrame] | None]:
        """Loads and splits data into training and testing sets."""
        if not self.backtest_engine._load_bar_data_engine(
            start_datetime, end_datetime, data_interval, vt_symbols
        ):
            self._write_log("Failed to load data, aborting optimization.", ERROR)
            return None, None

        try:
            train_data, test_data = self._split_data(test_size_ratio)
            train_rows = train_data["close"].height
            test_rows = test_data["close"].height
            self._write_log(
                f"Data split into {train_rows} train rows and {test_rows} test rows.",
                level=DEBUG,
            )
            return train_data, test_data
        except ValueError as e:
            self._write_log(f"Error splitting data: {e}", ERROR)
            return None, None

    def _run_optimization(
        self,
        factor_definition_template: FactorTemplate | dict,
        start_datetime: datetime,
        end_datetime: datetime,
        vt_symbols: list[str],
        data_interval: Interval,
        factor_json_conf_path: str | None,
        test_size_ratio: float,
        num_quantiles: int,
        long_short_percentile: float,
        report_filename_prefix: str,
        optimizer_callable: Callable[
            [dict[str, pl.DataFrame]], tuple[dict | None, dict | None]
        ],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path | None]:
        """Orchestrates the optimization workflow."""
        train_data, test_data = self._prepare_data(
            start_datetime, end_datetime, data_interval, vt_symbols, test_size_ratio
        )
        if not train_data or not test_data:
            return None, None, None

        best_params, search_results = optimizer_callable(train_data)

        if not best_params:
            self._write_log("Optimization failed to find any valid parameters.", ERROR)
            return None, search_results, None

        self._write_log(
            f"Best parameters found with score {search_results['best_score']:.4f}",
            data=best_params,
            level=INFO,
        )

        self._write_log("Running final analysis on the out-of-sample test set.", INFO)
        final_factor = self._create_final_factor(
            factor_definition_template, best_params
        )

        report_path = self._evaluate_on_test_set(
            final_factor=final_factor,
            test_data=test_data,
            vt_symbols=vt_symbols,
            factor_json_conf_path=factor_json_conf_path,
            num_quantiles=num_quantiles,
            long_short_percentile=long_short_percentile,
            report_filename_prefix=report_filename_prefix,
        )

        return best_params, search_results, report_path

    def optimize_factor(
        self,
        factor_definition_template: FactorTemplate | dict,
        parameter_grid: dict[str, list[Any]],
        start_datetime: datetime,
        end_datetime: datetime,
        vt_symbols: list[str],
        data_interval: Interval,
        factor_json_conf_path: str | None = None,
        test_size_ratio: float = 0.3,
        num_quantiles: int = 5,
        long_short_percentile: float = 0.5,
        report_filename_prefix: str = "optimized_factor",
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path | None]:
        """Performs grid search optimization."""
        self._write_log("Starting factor parameter optimization with Grid Search.", INFO)

        def grid_search_optimizer(
            train_data: dict[str, pl.DataFrame]
        ) -> tuple[dict | None, dict | None]:
            total_combinations = 1
            if parameter_grid:
                for p_list in parameter_grid.values():
                    total_combinations *= len(p_list)
            self._write_log(
                f"Grid search space: {total_combinations} combinations.", INFO
            )

            param_combinations = self._generate_param_combinations(parameter_grid)
            return self._run_grid_search(
                param_combinations=param_combinations,
                total_combinations=total_combinations,
                factor_definition_template=factor_definition_template,
                vt_symbols=vt_symbols,
                factor_json_conf_path=factor_json_conf_path,
                train_data=train_data,
                num_quantiles=num_quantiles,
                long_short_percentile=long_short_percentile,
            )

        return self._run_optimization(
            factor_definition_template=factor_definition_template,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            vt_symbols=vt_symbols,
            data_interval=data_interval,
            factor_json_conf_path=factor_json_conf_path,
            test_size_ratio=test_size_ratio,
            num_quantiles=num_quantiles,
            long_short_percentile=long_short_percentile,
            report_filename_prefix=report_filename_prefix,
            optimizer_callable=grid_search_optimizer,
        )

    def optimize_factor_bayes(
        self,
        factor_definition_template: FactorTemplate | dict,
        parameter_bounds: dict[str, tuple[float, float]],
        start_datetime: datetime,
        end_datetime: datetime,
        vt_symbols: list[str],
        data_interval: Interval,
        factor_json_conf_path: str | None = None,
        test_size_ratio: float = 0.3,
        num_quantiles: int = 5,
        long_short_percentile: float = 0.5,
        report_filename_prefix: str = "optimized_factor_bayes",
        n_iter: int = 25,
        init_points: int = 5,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path | None]:
        """Performs Bayesian optimization."""
        self._write_log("Starting factor parameter optimization with Bayesian method.", INFO)

        # Auto-detect parameter types (int or float) from bounds
        param_types = {
            p_name: "int"
            if isinstance(bounds[0], int) and isinstance(bounds[1], int)
            else "float"
            for p_name, bounds in parameter_bounds.items()
        }

        def bayesian_optimizer(
            train_data: dict[str, pl.DataFrame]
        ) -> tuple[dict | None, dict | None]:
            def black_box_function(**params):
                # Round and cast parameters to their detected types
                typed_params = {}
                for p_name, p_value in params.items():
                    if param_types.get(p_name) == "int":
                        typed_params[p_name] = int(round(p_value))
                    else:
                        typed_params[p_name] = p_value

                return self._calculate_factor_score(
                    base_factor_definition=factor_definition_template,
                    params_to_set=typed_params,
                    vt_symbols=vt_symbols,
                    factor_json_conf_path=factor_json_conf_path,
                    data_to_use=train_data,
                    num_quantiles=num_quantiles,
                    long_short_percentile=long_short_percentile,
                )

            optimizer = BayesianOptimization(
                f=black_box_function, pbounds=parameter_bounds, random_state=1
            )
            optimizer.maximize(init_points=init_points, n_iter=n_iter)

            best_params = optimizer.max["params"]
            # Ensure final best parameters are correctly typed
            for p_name, p_value in best_params.items():
                if param_types.get(p_name) == "int":
                    best_params[p_name] = int(round(p_value))

            search_results = {
                "best_score": optimizer.max["target"],
                "all_results": optimizer.res,
            }
            return best_params, search_results

        return self._run_optimization(
            factor_definition_template=factor_definition_template,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            vt_symbols=vt_symbols,
            data_interval=data_interval,
            factor_json_conf_path=factor_json_conf_path,
            test_size_ratio=test_size_ratio,
            num_quantiles=num_quantiles,
            long_short_percentile=long_short_percentile,
            report_filename_prefix=report_filename_prefix,
            optimizer_callable=bayesian_optimizer,
        )

    def _run_grid_search(
        self,
        param_combinations: Iterator[dict[str, Any]],
        total_combinations: int,
        factor_definition_template: FactorTemplate | dict,
        vt_symbols: list[str],
        factor_json_conf_path: str | None,
        train_data: dict[str, pl.DataFrame],
        num_quantiles: int,
        long_short_percentile: float,
    ) -> tuple[dict[str, Any] | None, dict]:
        """Iterates through parameter combinations and finds the best score."""
        best_score = -float("inf")
        best_params = None
        search_results = {"params": [], "scores": []}

        for i, params in enumerate(param_combinations):
            self._write_log(
                f"Evaluating combo {i + 1}/{total_combinations}",
                data=params,
                level=DEBUG,
            )
            score = self._calculate_factor_score(
                base_factor_definition=factor_definition_template,
                params_to_set=params,
                vt_symbols=vt_symbols,
                factor_json_conf_path=factor_json_conf_path,
                data_to_use=train_data,
                num_quantiles=num_quantiles,
                long_short_percentile=long_short_percentile,
            )

            search_results["params"].append(params)
            search_results["scores"].append(score)

            if score > best_score:
                best_score = score
                best_params = params

        search_results["best_score"] = best_score
        return best_params, search_results

    def _calculate_factor_score(
        self,
        base_factor_definition: FactorTemplate | dict,
        params_to_set: dict[str, Any],
        vt_symbols: list[str],
        factor_json_conf_path: str | None,
        data_to_use: dict[str, pl.DataFrame],
        num_quantiles: int,
        long_short_percentile: float,
    ) -> float:
        """
        Calculates the performance score (Sharpe Ratio) for a single set of parameters.
        """
        current_factor_def = self._create_final_factor(
            base_factor_definition, params_to_set
        )

        target_factor, flat_factors = self.backtest_engine._init_and_flatten_factor(
            factor_definition=current_factor_def,
            vt_symbols_for_factor=vt_symbols,
            factor_json_conf_path=factor_json_conf_path,
        )
        if not target_factor or not flat_factors:
            self._write_log(
                f"Failed to init/flatten factor for params: {params_to_set}", WARNING
            )
            return -float("inf")

        calculator = self.backtest_engine._create_calculator()
        factor_df = calculator.compute_factor_values(
            target_factor_instance_input=target_factor,
            flattened_factors_input=flat_factors,
            memory_bar_input=data_to_use,
            num_data_rows_input=data_to_use["close"].height,
            vt_symbols_for_run=vt_symbols,
        )
        calculator.close()
        if factor_df.is_empty():
            self._write_log(
                f"Factor calculation yielded no data for params: {params_to_set}",
                WARNING,
            )
            return -float("inf")

        analyser = FactorAnalyser()
        analyser.config.num_quantiles = num_quantiles
        analyser.config.long_short_percentile = long_short_percentile

        market_close = data_to_use["close"]
        symbol_returns_df = analyser._prepare_symbol_returns(market_close, vt_symbols)
        analysis_data = analyser._prepare_analysis_data(factor_df, symbol_returns_df)

        analyser.annualization_factor = get_annualization_factor(
            datetimes=market_close[DEFAULT_DATETIME_COL]
        )

        if analysis_data is None or analysis_data.is_empty():
            self._write_log(f"Analysis data empty for params: {params_to_set}", WARNING)
            analyser.close()
            return -float("inf")

        analyser.perform_long_short_analysis(analysis_data)

        score = -float("inf")
        if (
            analyser.long_short_stats
            and analyser.long_short_stats.sharpe_ratio is not None
        ):
            score = analyser.long_short_stats.sharpe_ratio

        analyser.close()
        self._write_log(f"Score:{score}", data=params_to_set, level=DEBUG)
        return score

    def _evaluate_on_test_set(
        self,
        final_factor: FactorTemplate,
        test_data: dict[str, pl.DataFrame],
        vt_symbols: list[str],
        factor_json_conf_path: str,
        num_quantiles: int,
        long_short_percentile: float,
        report_filename_prefix: str,
    ) -> Path | None:
        """Runs a full analysis on the hold-out test set and generates a report."""
        target_factor, flat_factors = self.backtest_engine._init_and_flatten_factor(
            factor_definition=final_factor,
            vt_symbols_for_factor=vt_symbols,
            factor_json_conf_path=factor_json_conf_path,
        )
        if not target_factor or not flat_factors:
            self._write_log(
                "Failed to init final factor for test set evaluation.", ERROR
            )
            return None

        calculator = self.backtest_engine._create_calculator()
        factor_df = calculator.compute_factor_values(
            target_factor_instance_input=target_factor,
            flattened_factors_input=flat_factors,
            memory_bar_input=test_data,
            num_data_rows_input=test_data["close"].height,
            vt_symbols_for_run=vt_symbols,
        )
        calculator.close()
        if factor_df is None or factor_df.is_empty():
            self._write_log("Factor calculation failed on the test set.", ERROR)
            return None

        analyser = FactorAnalyser(
            output_data_dir_for_reports=self.backtest_engine.output_data_dir_for_analyser_reports,
        )

        analyser.annualization_factor = get_annualization_factor(
            datetimes=test_data["close"][DEFAULT_DATETIME_COL]
        )

        start_dt = test_data["close"][DEFAULT_DATETIME_COL].min()
        end_dt = test_data["close"][DEFAULT_DATETIME_COL].max()

        report_path = analyser.run_analysis_and_report(
            factor_data_df=factor_df,
            market_close_prices_df=test_data["close"],
            factor_instance=target_factor,
            analysis_start_dt=start_dt,
            analysis_end_dt=end_dt,
            num_quantiles=num_quantiles,
            long_short_percentile=long_short_percentile,
            report_filename_prefix=report_filename_prefix + "_test_set",
        )
        analyser.close()
        return report_path

    def _create_final_factor(
        self, template: FactorTemplate | dict, params: dict
    ) -> FactorTemplate | dict:
        """Creates the final factor instance with the best parameters."""
        final_factor = copy.deepcopy(template)
        if isinstance(final_factor, FactorTemplate):
            final_factor.set_nested_params_for_optimizer(params)
        else:
            final_factor = apply_params_to_definition_dict(final_factor, params)
        return final_factor

    def _split_data(
        self, test_size_ratio: float
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        """Splits the data loaded in the backtest engine into train and test sets."""
        if self.backtest_engine.memory_bar.get("close").is_empty():
            raise ValueError("Memory bar is not loaded in the backtest engine.")

        total_rows = self.backtest_engine.memory_bar["close"].height
        if total_rows == 0:
            raise ValueError("No data available to split.")

        train_size = int(total_rows * (1 - test_size_ratio))
        train_data, test_data = {}, {}

        for key, df in self.backtest_engine.memory_bar.items():
            if isinstance(df, pl.DataFrame) and not df.is_empty():
                train_data[key] = df.slice(0, train_size)
                test_data[key] = df.slice(train_size)
            else:
                train_data[key] = df
                test_data[key] = df

        return train_data, test_data

    def _generate_param_combinations(
        self, parameter_grid: dict[str, list[Any]]
    ) -> Iterator[dict[str, Any]]:
        """Generates parameter combinations from a grid using itertools."""
        param_names = list(parameter_grid.keys())
        param_values = parameter_grid.values()

        for combo in itertools.product(*param_values):
            yield dict(zip(param_names, combo, strict=False))

    def _write_log(self, msg: str, data=None, level: int = INFO) -> None:
        level_map = {
            DEBUG: logger.debug,
            INFO: logger.info,
            WARNING: logger.warning,
            ERROR: logger.error,
        }
        log_func = level_map.get(level, logger.info)
        log_func(msg, data=data, gateway_name=self.engine_name)
