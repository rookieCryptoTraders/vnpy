import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from logging import INFO, DEBUG, WARNING, ERROR
from pathlib import Path
import re
from typing import Any
import webbrowser

# Third-party imports
import numpy as np
import polars as pl
import pandas as pd
import quantstats as qs

# VnTrader imports
from vnpy.factor.template import FactorTemplate
from vnpy.factor.base import APP_NAME
from vnpy.factor.setting import get_backtest_report_path

# Default datetime column name
DEFAULT_DATETIME_COL = "datetime"

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(APP_NAME + "_FactorAnalyser")


# --- Data Structures for Configuration and Results ---


@dataclass
class AnalysisConfig:
    """Configuration for factor analysis parameters."""

    num_quantiles: int = 5
    long_short_percentile: float = 0.3
    risk_free_rate: float = 0.05


@dataclass
class QuantileResults:
    """Container for quantile analysis results."""

    by_time: pl.DataFrame
    overall_average: pl.DataFrame


@dataclass
class LongShortStats:
    """Container for script-calculated long-short portfolio statistics."""

    mean_return: float | None = None
    std_return: float | None = None
    sharpe_ratio: float | None = None
    t_stat: float | None = None


@dataclass
class PreparedAnalysisData:
    """Holds all data required for running core factor analyses."""

    analysis_df: pl.DataFrame
    symbols_of_interest: list[str]
    actual_start_dt: datetime
    actual_end_dt: datetime


def get_annualization_factor(datetimes: pl.Series) -> float:
    """
    Calculates the annualization factor based on the median time difference
    in a datetime series.
    """
    if datetimes.len() < 2:
        return 365.0  # Default to daily if not enough data

    median_delta_seconds = datetimes.diff().dt.total_seconds().median()
    if median_delta_seconds is None:
        return 365.0

    one_day_seconds = timedelta(days=1).total_seconds()
    if median_delta_seconds > 0:
        periods_per_day = one_day_seconds / median_delta_seconds
    else:
        return 365.0

    return 365.0 if periods_per_day < 2 else 365.0 * periods_per_day


class FactorReportGenerator:
    """Handles the generation of JSON and HTML reports for factor analysis."""

    engine_name = APP_NAME + "FactorReportGenerator"

    def __init__(
        self,
        output_data_dir: Path,
        factor_datetime_col: str,
        annualization_factor: float,
        config: AnalysisConfig,
    ):
        self.output_data_dir = output_data_dir
        self.factor_datetime_col = factor_datetime_col
        self.annualization_factor = annualization_factor
        self.config = config

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Sanitizes a string to be used as a valid filename."""
        return re.sub(r"[^\w\.\-@]", "_", name)

    def generate_report_data(
        self,
        factor_instance: FactorTemplate,
        analysis_start_dt: datetime,
        analysis_end_dt: datetime,
        tested_vt_symbols: list[str],
        quantile_results: QuantileResults | None,
        long_short_stats: LongShortStats | None,
        performance_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compiles all analysis results into a single dictionary for reporting."""
        self._write_log("Generating consolidated report data...", level=DEBUG)
        report_data = {
            "factor_key": factor_instance.factor_key,
            "factor_class_name": factor_instance.__class__.__name__,
            "factor_parameters": factor_instance.get_nested_params_for_optimizer(),
            "analysis_run_datetime": datetime.now().isoformat(),
            "vt_symbols_tested": tested_vt_symbols,
            "data_period_analyzed": {
                "start_datetime": analysis_start_dt.isoformat(),
                "end_datetime": analysis_end_dt.isoformat(),
            },
            "analysis_config": asdict(self.config),
            "analysis_results": {},
        }

        if quantile_results:
            report_data["analysis_results"]["quantile_analysis"] = {
                "overall_average": quantile_results.overall_average.to_dicts()
            }
        if long_short_stats:
            report_data["analysis_results"][
                "long_short_portfolio_statistics_script"
            ] = asdict(long_short_stats)
        if performance_metrics:
            report_data["analysis_results"]["performance_metrics_quantstats"] = (
                performance_metrics
            )
        return report_data

    def save_json_report(
        self, report_data: dict[str, Any], report_filename_prefix: str
    ) -> Path | None:
        """Saves the report data dictionary as a JSON file."""
        if not report_data:
            self._write_log("Report data is empty, skipping JSON report.", level=WARNING)
            return None

        factor_key_safe = self._safe_filename(report_data.get("factor_key", "unknown_factor"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_filename_prefix}_{factor_key_safe}_{timestamp}.json"
        filepath = self.output_data_dir / "json_reports" / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            self._write_log(f"JSON report saved successfully: {filepath}", level=DEBUG)
            return filepath
        except Exception as e:
            self._write_log(f"Error saving JSON report: {e}", level=ERROR)
            return None

    def generate_html_report(
        self,
        factor_key: str,
        returns_pd: pd.Series,
        benchmark_pd: pd.Series | None = None,
    ) -> Path | None:
        """Generates and saves a comprehensive HTML report using QuantStats."""
        self._write_log("Generating interactive HTML report...", level=DEBUG)
        factor_key_safe = self._safe_filename(factor_key)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{factor_key_safe}_{timestamp}.html"
        filepath = self.output_data_dir / "html_reports" / filename

        try:
            qs.reports.html(
                returns=returns_pd,
                benchmark=benchmark_pd,
                rf=self.config.risk_free_rate,
                periods_per_year=self.annualization_factor,
                title=f"Factor Analysis Report: {factor_key}",
                output=str(filepath),
                download_filename=filename,
            )
            self._write_log(f"HTML report saved successfully: {filepath}", level=DEBUG)
            webbrowser.open(f"file://{filepath.resolve()}")
            return filepath
        except Exception as e:
            self._write_log(f"Error generating HTML report with QuantStats: {e}", level=ERROR)
            return None

    def generate_and_save_reports(
        self,
        prepared_data: PreparedAnalysisData,
        factor_instance: FactorTemplate,
        analysis_start_dt: datetime,
        analysis_end_dt: datetime,
        report_filename_prefix: str,
        quantile_results: QuantileResults | None,
        long_short_stats: LongShortStats | None,
        performance_metrics: dict[str, Any] | None,
        returns_pd: pd.Series,
        benchmark_pd: pd.Series | None,
    ) -> Path | None:
        """Generates and saves JSON and HTML reports."""
        self._write_log("--- Generating and Saving Reports ---", level=DEBUG)
        report_content = self.generate_report_data(
            factor_instance=factor_instance,
            analysis_start_dt=prepared_data.actual_start_dt or analysis_start_dt,
            analysis_end_dt=prepared_data.actual_end_dt or analysis_end_dt,
            tested_vt_symbols=prepared_data.symbols_of_interest,
            quantile_results=quantile_results,
            long_short_stats=long_short_stats,
            performance_metrics=performance_metrics,
        )

        json_report_path = self.save_json_report(report_content, report_filename_prefix)
        self.generate_html_report(
            factor_instance.factor_key, returns_pd, benchmark_pd
        )
        return json_report_path

    def _write_log(self, msg: str, level: int = INFO) -> None:
        level_map = {
            DEBUG: logger.debug, INFO: logger.info,
            WARNING: logger.warning, ERROR: logger.error,
        }
        log_func = level_map.get(level, logger.info)
        log_func(msg, gateway_name=self.engine_name)


class FactorAnalyser:
    """
    An advanced factor performance analyser.
    This class performs analysis and uses a report generator to output results.
    """

    engine_name = APP_NAME + "FactorAnalyser"

    def __init__(
        self,
        output_data_dir_for_reports: str | None = None,
        config: AnalysisConfig | None = None,
    ):
        self.output_data_dir: Path = (
            Path(output_data_dir_for_reports)
            if output_data_dir_for_reports
            else get_backtest_report_path()
        )
        self.factor_datetime_col: str = DEFAULT_DATETIME_COL
        self.config = config or AnalysisConfig()
        self.annualization_factor: float = 252.0

        self.quantile_analysis_results: QuantileResults | None = None
        self.long_short_portfolio_returns_df: pl.DataFrame | None = None
        self.long_short_stats: LongShortStats | None = None
        self.performance_metrics: dict[str, Any] | None = None

        self._prepare_output_directory()
        self._write_log(
            f"FactorAnalyser initialized. Reports will be saved in: {self.output_data_dir}",
            level=INFO,
        )

    def _prepare_output_directory(self) -> None:
        """Ensures the output directory and its subdirectories exist."""
        try:
            (self.output_data_dir / "json_reports").mkdir(parents=True, exist_ok=True)
            (self.output_data_dir / "html_reports").mkdir(parents=True, exist_ok=True)
            (self.output_data_dir / "plots").mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._write_log(f"Error creating report directories: {e}", level=ERROR)
            raise

    def _prepare_symbol_returns(
        self,
        market_close_prices_df: pl.DataFrame,
        symbols_of_interest: list[str]
    ) -> pl.DataFrame:
        """Calculate 1-period forward returns for symbols."""
        if not symbols_of_interest:
            self._write_log("No symbols provided for returns", level=WARNING)
            return pl.DataFrame()

        forward_returns = [
            (pl.col(symbol).shift(-1) - pl.col(symbol))
            .truediv(pl.col(symbol))
            .fill_nan(None)
            .alias(symbol)
            for symbol in symbols_of_interest
        ]
        returns_df = market_close_prices_df.select(
            [pl.col(self.factor_datetime_col)] + forward_returns
        ).drop_nulls()

        if returns_df.is_empty():
            self._write_log("Calculated returns DataFrame is empty", level=WARNING)
        return returns_df

    def _prepare_analysis_data(
        self, aligned_factor_df: pl.DataFrame, aligned_symbol_returns_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        """Merges factor and return data into a tidy 'long' format."""
        symbols = [col for col in aligned_factor_df.columns if col != self.factor_datetime_col]
        factor_long = aligned_factor_df.melt(
            id_vars=[self.factor_datetime_col], value_vars=symbols,
            variable_name="symbol", value_name="factor_value",
        )
        returns_long = aligned_symbol_returns_df.melt(
            id_vars=[self.factor_datetime_col], value_vars=symbols,
            variable_name="symbol", value_name="forward_return",
        )
        analysis_data = factor_long.join(
            returns_long, on=[self.factor_datetime_col, "symbol"], how="inner"
        ).filter(
            pl.col("factor_value").is_not_null() & pl.col("forward_return").is_not_null()
        )

        if analysis_data.is_empty():
            self._write_log("Analysis data empty after joining factors and returns.", level=ERROR)
            return None
        return analysis_data

    def perform_quantile_analysis(self, analysis_data: pl.DataFrame) -> bool:
        """Performs quantile analysis on the prepared data."""
        if analysis_data.is_empty():
            self._write_log("Analysis data is empty for quantile analysis.", level=ERROR)
            return False

        data_with_quantiles = analysis_data.with_columns(
            pl.col("factor_value").qcut(
                self.config.num_quantiles,
                labels=[f"Q{i+1}" for i in range(self.config.num_quantiles)],
                allow_duplicates=True
            ).alias("quantile")
        )
        quantile_returns_by_time = (
            data_with_quantiles.group_by([self.factor_datetime_col, "quantile"])
            .agg(pl.col("forward_return").mean().alias("mean_quantile_return"))
            .sort([self.factor_datetime_col, "quantile"])
        )
        average_quantile_returns_overall = (
            quantile_returns_by_time.group_by("quantile")
            .agg(
                pl.col("mean_quantile_return").mean().alias("avg_return"),
                pl.col("mean_quantile_return").std().alias("std_dev"),
            )
            .sort("quantile")
        )
        self.quantile_analysis_results = QuantileResults(
            by_time=quantile_returns_by_time,
            overall_average=average_quantile_returns_overall,
        )
        return True

    def perform_long_short_analysis(self, analysis_data: pl.DataFrame) -> bool:
        """Performs long-short portfolio analysis."""
        percentile = self.config.long_short_percentile
        if analysis_data.is_empty():
            self._write_log("Analysis data is empty for L/S analysis.", level=ERROR)
            return False

        data_with_legs = analysis_data.with_columns(
            pl.when(pl.col("factor_value") >= pl.col("factor_value").quantile(1 - percentile))
            .then(pl.lit("long"))
            .when(pl.col("factor_value") <= pl.col("factor_value").quantile(percentile))
            .then(pl.lit("short"))
            .otherwise(None)
            .alias("leg")
        ).filter(pl.col("leg").is_not_null())

        leg_returns = data_with_legs.group_by([self.factor_datetime_col, "leg"]).agg(
            pl.col("forward_return").mean().alias("leg_return")
        )
        ls_pivot = leg_returns.pivot(
            index=self.factor_datetime_col, columns="leg", values="leg_return"
        ).fill_null(0.0)

        ls_returns_df = ls_pivot.with_columns(
            (pl.col("long") - pl.col("short")).alias("ls_return")
        ).select([self.factor_datetime_col, "ls_return"]).sort(self.factor_datetime_col)

        self.long_short_portfolio_returns_df = ls_returns_df

        if ls_returns_df.height < 2:
            self.long_short_stats = LongShortStats()
            return True

        ls_return_col = ls_returns_df.get_column("ls_return")
        mean_ret = ls_return_col.mean()
        std_ret = ls_return_col.std()

        sharpe, t_stat = None, None
        if std_ret is not None and std_ret > 1e-6:
            sharpe = (mean_ret / std_ret) * (self.annualization_factor ** 0.5)
            t_stat = (mean_ret / (std_ret / (ls_returns_df.height ** 0.5)))

        self.long_short_stats = LongShortStats(
            mean_return=mean_ret, std_return=std_ret, sharpe_ratio=sharpe, t_stat=t_stat
        )
        return True

    def calculate_performance_metrics(
        self,
        benchmark_prices_df: pl.DataFrame | None = None,
        benchmark_symbol: str | None = None,
    ) -> bool:
        """Calculates performance metrics using QuantStats."""
        if self.long_short_portfolio_returns_df is None or self.long_short_portfolio_returns_df.is_empty():
            self._write_log("L/S portfolio returns not available for metrics.", level=ERROR)
            return False

        returns_pd = self.long_short_portfolio_returns_df.to_pandas().set_index(self.factor_datetime_col)["ls_return"].fillna(0.0)
        benchmark_pd = None
        if benchmark_prices_df is not None and benchmark_symbol:
            benchmark_returns_df = self._prepare_symbol_returns(benchmark_prices_df, [benchmark_symbol])
            if benchmark_returns_df is not None:
                benchmark_pd = benchmark_returns_df.to_pandas().set_index(self.factor_datetime_col)[benchmark_symbol].fillna(0.0)
                returns_pd, benchmark_pd = returns_pd.align(benchmark_pd, join="left", axis=0)
                benchmark_pd = benchmark_pd.fillna(0.0)

        try:
            metrics = qs.reports.metrics(
                returns=returns_pd, benchmark=benchmark_pd,
                rf=self.config.risk_free_rate, display=False, mode="full",
            )
            self.performance_metrics = {
                k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
                for k, v in metrics.to_dict().items()
            }
            return True
        except Exception as e:
            self._write_log(f"Error during QuantStats metrics calculation: {e}", level=ERROR)
            self.performance_metrics = None
            return False

    def _prepare_data_for_analysis(
        self,
        factor_data_df: pl.DataFrame,
        market_close_prices_df: pl.DataFrame,
    ) -> PreparedAnalysisData | None:
        """Validates and prepares all data needed for the core analysis."""
        if factor_data_df.is_empty() or market_close_prices_df.is_empty():
            self._write_log("Input factor or market price data is empty.", level=ERROR)
            return None

        symbols = [col for col in factor_data_df.columns if col != self.factor_datetime_col]
        if not symbols:
            self._write_log("No symbols found in factor data.", level=ERROR)
            return None

        symbol_returns_df = self._prepare_symbol_returns(market_close_prices_df, symbols)
        if symbol_returns_df.is_empty():
            self._write_log("Symbol returns preparation failed.", level=ERROR)
            return None

        aligned_factor_df, aligned_returns_df = pl.align_frames(
            factor_data_df, symbol_returns_df, on=self.factor_datetime_col, how="inner"
        )
        if aligned_factor_df.is_empty():
            self._write_log("Data empty after aligning factors and returns.", level=WARNING)
            return None

        analysis_data = self._prepare_analysis_data(aligned_factor_df, aligned_returns_df)
        if analysis_data is None or analysis_data.is_empty():
            self._write_log("Analysis data preparation failed.", level=ERROR)
            return None

        return PreparedAnalysisData(
            analysis_df=analysis_data,
            symbols_of_interest=symbols,
            actual_start_dt=aligned_factor_df.get_column(self.factor_datetime_col).min(),
            actual_end_dt=aligned_factor_df.get_column(self.factor_datetime_col).max(),
        )

    def _run_core_analyses(
        self,
        prepared_data: PreparedAnalysisData,
        benchmark_prices_df: pl.DataFrame | None,
        benchmark_symbol: str | None,
    ) -> None:
        """Runs all core analysis calculations."""
        self._write_log("--- Running Core Analyses ---", level=DEBUG)
        self.perform_quantile_analysis(prepared_data.analysis_df)
        self.perform_long_short_analysis(prepared_data.analysis_df)
        self.calculate_performance_metrics(benchmark_prices_df, benchmark_symbol)

    def run_analysis_and_report(
        self,
        factor_data_df: pl.DataFrame,
        market_close_prices_df: pl.DataFrame,
        factor_instance: FactorTemplate,
        analysis_start_dt: datetime,
        analysis_end_dt: datetime,
        num_quantiles: int = 5,
        long_short_percentile: float = 0.3,
        report_filename_prefix: str = "factor_analysis_report",
        benchmark_prices_df: pl.DataFrame | None = None,
        benchmark_symbol: str | None = None,
    ) -> Path | None:
        """
        Runs the full analysis pipeline and generates JSON and HTML reports.
        """
        self._write_log(f"--- Starting Full Analysis for Factor: {factor_instance.factor_key} ---", level=INFO)

        self.config.num_quantiles = num_quantiles
        self.config.long_short_percentile = long_short_percentile
        if not factor_data_df.is_empty():
            self.annualization_factor = get_annualization_factor(factor_data_df[self.factor_datetime_col])
            self._write_log(f"Auto-detected annualization factor: {self.annualization_factor}", level=DEBUG)

        prepared_data = self._prepare_data_for_analysis(factor_data_df, market_close_prices_df)
        if not prepared_data:
            self._write_log(f"Data preparation failed for {factor_instance.factor_key}.", level=ERROR)
            return None

        self._run_core_analyses(prepared_data, benchmark_prices_df, benchmark_symbol)

        returns_pd = self.long_short_portfolio_returns_df.to_pandas().set_index(self.factor_datetime_col)["ls_return"].fillna(0.0)
        benchmark_pd = None
        if benchmark_prices_df is not None and benchmark_symbol:
            benchmark_returns = self._prepare_symbol_returns(benchmark_prices_df, [benchmark_symbol])
            if benchmark_returns is not None:
                benchmark_pd = benchmark_returns.to_pandas().set_index(self.factor_datetime_col)[benchmark_symbol].fillna(0.0)
                returns_pd, benchmark_pd = returns_pd.align(benchmark_pd, join="left", axis=0)
                benchmark_pd = benchmark_pd.fillna(0.0)

        reporter = FactorReportGenerator(
            output_data_dir=self.output_data_dir,
            factor_datetime_col=self.factor_datetime_col,
            annualization_factor=self.annualization_factor,
            config=self.config,
        )
        json_report_path = reporter.generate_and_save_reports(
            prepared_data=prepared_data,
            factor_instance=factor_instance,
            analysis_start_dt=analysis_start_dt,
            analysis_end_dt=analysis_end_dt,
            report_filename_prefix=report_filename_prefix,
            quantile_results=self.quantile_analysis_results,
            long_short_stats=self.long_short_stats,
            performance_metrics=self.performance_metrics,
            returns_pd=returns_pd,
            benchmark_pd=benchmark_pd,
        )

        self._write_log(f"--- Analysis for {factor_instance.factor_key} Complete. ---", level=INFO)
        return json_report_path

    def _write_log(self, msg: str, level: int = INFO) -> None:
        level_map = {
            DEBUG: logger.debug, INFO: logger.info,
            WARNING: logger.warning, ERROR: logger.error,
        }
        log_func = level_map.get(level, logger.info)
        log_func(msg, gateway_name=self.engine_name)

    def close(self) -> None:
        self._write_log("FactorAnalyser closed.", level=INFO)
