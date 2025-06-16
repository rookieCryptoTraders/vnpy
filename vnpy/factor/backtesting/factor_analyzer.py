import json
from dataclasses import dataclass, asdict
from datetime import datetime
from logging import INFO, DEBUG, WARNING, ERROR
from pathlib import Path
import re
from typing import Any
import webbrowser

# Third-party imports
import numpy as np
import polars as pl
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


def safe_filename(name: str) -> str:
    """Sanitizes a string to be used as a valid filename."""
    return re.sub(r"[^\w\.\-@]", "_", name)


# --- Data Structures for Configuration and Results ---


@dataclass
class AnalysisConfig:
    """Configuration for factor analysis parameters."""

    num_quantiles: int = 5
    long_short_percentile: float = 0.3
    annualization_factor: int = 24*365  # Assumes hours data
    risk_free_rate: float = 0.0


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


# --- Main Analyser Class ---


class FactorAnalyser:
    """
    An advanced factor performance analyser.

    This class performs quantile and long-short portfolio analysis,
    calculates extensive performance metrics using QuantStats, and
    generates comprehensive JSON and interactive HTML reports.
    """

    engine_name = APP_NAME + "FactorAnalyser"

    def __init__(
        self,
        output_data_dir_for_reports: str | None = None,
        config: AnalysisConfig | None = None,
    ):
        """
        Initializes the FactorAnalyser.

        Args:
            output_data_dir_for_reports (Optional[str]): Directory to save reports. Defaults to vnpy's backtest path.
            config (Optional[AnalysisConfig]): A configuration object for analysis parameters.
        """
        self.output_data_dir: Path = (
            Path(output_data_dir_for_reports)
            if output_data_dir_for_reports
            else get_backtest_report_path()
        )
        self.factor_datetime_col: str = DEFAULT_DATETIME_COL
        self.config = config or AnalysisConfig()

        # --- Result attributes ---
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
        """
        Calculate 1-period forward returns for symbols.

        Args:
            market_close_prices_df: DataFrame with datetime and symbol columns
            symbols_of_interest: List of symbol columns to calculate returns for

        Returns:
            DataFrame with forward returns
        """
        if not symbols_of_interest:
            self._write_log("No symbols provided for returns", level=WARNING)
            return pl.DataFrame()

        # Vectorized calculation of all forward returns
        forward_returns = [
            (pl.col(symbol).shift(-1) - pl.col(symbol))
            .truediv(pl.col(symbol))
            .fill_nan(None)
            .alias(symbol)
            for symbol in symbols_of_interest
        ]

        returns_df = market_close_prices_df.select(
            [pl.col(self.factor_datetime_col)] + forward_returns
        ).drop_nulls()  # Automatically drops last row with null returns

        if returns_df.is_empty():
            self._write_log("Calculated returns DataFrame is empty", level=WARNING)

        return returns_df

    def _prepare_analysis_data(
        self, aligned_factor_df: pl.DataFrame, aligned_symbol_returns_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        """Merges factor and return data into a tidy 'long' format for analysis."""
        self._write_log(
            "Preparing and aligning factor and returns data...", level=DEBUG
        )
        symbols = [
            col for col in aligned_factor_df.columns if col != self.factor_datetime_col
        ]

        factor_long = aligned_factor_df.melt(
            id_vars=[self.factor_datetime_col],
            value_vars=symbols,
            variable_name="symbol",
            value_name="factor_value",
        )
        returns_long = aligned_symbol_returns_df.melt(
            id_vars=[self.factor_datetime_col],
            value_vars=symbols,
            variable_name="symbol",
            value_name="forward_return",
        )

        # Join and filter out any rows with missing data needed for analysis
        analysis_data = factor_long.join(
            returns_long, on=[self.factor_datetime_col, "symbol"], how="inner"
        ).filter(
            pl.col("factor_value").is_not_null()
            & pl.col("forward_return").is_not_null()
        )

        if analysis_data.is_empty():
            self._write_log(
                "Analysis data is empty after joining factors and returns.", level=ERROR
            )
            return None
        return analysis_data

    def perform_quantile_analysis(self, analysis_data: pl.DataFrame) -> bool:
        """Performs quantile analysis on the prepared data."""
        self._write_log(
            f"Performing quantile analysis ({self.config.num_quantiles} quantiles)...",
            level=DEBUG,
        )
        if analysis_data.is_empty():
            self._write_log(
                "Analysis data is empty for quantile analysis.", level=ERROR
            )
            return False

        # Assign assets to quantiles using ntile
        data_with_quantiles = analysis_data.with_columns(
            pl.col("factor_value").qcut(
                self.config.num_quantiles, labels=[f"Q{i+1}" for i in range(self.config.num_quantiles)]
            ).alias("quantile")
        )

        # Calculate mean return for each quantile at each time period
        quantile_returns_by_time = (
            data_with_quantiles.group_by([self.factor_datetime_col, "quantile"])
            .agg(pl.col("forward_return").mean().alias("mean_quantile_return"))
            .sort([self.factor_datetime_col, "quantile"])
        )

        # Calculate the overall average return and std dev for each quantile
        average_quantile_returns_overall = (
            quantile_returns_by_time.group_by("quantile")
            .agg(
                pl.col("mean_quantile_return").mean().alias("avg_return"),
                pl.col("mean_quantile_return").std().alias("std_dev"),
                pl.col("mean_quantile_return").count().alias("count_periods"),
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
        self._write_log(
            f"Performing L/S analysis (top/bottom {percentile * 100:.0f}%)...",
            level=DEBUG,
        )
        if analysis_data.is_empty():
            self._write_log("Analysis data is empty for L/S analysis.", level=ERROR)
            return False

        # Identify long (top percentile) and short (bottom percentile) legs
        data_with_legs = analysis_data.with_columns(
            pl.col("factor_value").qcut(
                [percentile], labels=['is_long', 'is_short']
            ).alias("pos")
        )

        data_with_legs = data_with_legs.group_by(
            [self.factor_datetime_col, "pos"]
        ).agg(
            pl.col("forward_return").mean().alias("forward_return")
        ).filter(
            pl.col("pos").is_in(["is_long", "is_short"])
        )

        data_with_legs = data_with_legs.pivot(
            index=self.factor_datetime_col,
            columns="pos",
            values="forward_return"
        ).fill_null(0.0)

        # Calculate long-short returns
        ls_returns_df = (
            data_with_legs.with_columns(
                (pl.col("is_long") - pl.col("is_short")).alias("ls_return")
            )
            .select([self.factor_datetime_col, "ls_return"])
            .sort(self.factor_datetime_col)
        )

        self.long_short_portfolio_returns_df = ls_returns_df

        # Calculate basic portfolio statistics
        if ls_returns_df.height < 2:
            self._write_log(
                "L/S portfolio returns series is too short for meaningful stats.",
                level=WARNING,
            )
            self.long_short_stats = LongShortStats()
            return True

        ls_return_col = ls_returns_df.get_column("ls_return")
        mean_ret = ls_return_col.mean() - self.config.risk_free_rate
        std_ret = ls_return_col.std()

        # Avoid division by zero
        if std_ret is not None and std_ret > 0:
            sharpe = (
                (mean_ret / std_ret) * (self.config.annualization_factor**0.5)
                if mean_ret is not None
                else None
            )
            t_stat = (
                (mean_ret / (std_ret / (ls_returns_df.height**0.5)))
                if mean_ret is not None
                else None
            )
        else:
            sharpe = None
            t_stat = None

        self.long_short_stats = LongShortStats(
            mean_return=mean_ret, std_return=std_ret, sharpe_ratio=sharpe, t_stat=t_stat
        )
        self._write_log(f"L/S Portfolio Stats: {self.long_short_stats}", level=INFO)
        return True

    def calculate_performance_metrics(
        self,
        benchmark_prices_df: pl.DataFrame | None = None,
        benchmark_symbol: str | None = None,
    ) -> bool:
        """Calculates a comprehensive suite of performance metrics using QuantStats."""
        self._write_log(
            "Calculating full suite of performance metrics with QuantStats...",
            level=DEBUG,
        )
        if (
            self.long_short_portfolio_returns_df is None
            or self.long_short_portfolio_returns_df.is_empty()
        ):
            self._write_log(
                "L/S portfolio returns are not available. Cannot calculate metrics.",
                level=ERROR,
            )
            return False

        # Convert portfolio returns to a pandas Series, required by QuantStats
        returns_pd = (
            self.long_short_portfolio_returns_df.select(
                [self.factor_datetime_col, "ls_return"]
            )
            .to_pandas()
            .set_index(self.factor_datetime_col)["ls_return"]
            .fillna(0.0)
        )

        # Prepare benchmark returns if provided
        benchmark_pd = None
        if benchmark_prices_df is not None and benchmark_symbol:
            benchmark_returns_df = self._prepare_symbol_returns(
                benchmark_prices_df, [benchmark_symbol]
            )
            if benchmark_returns_df is not None:
                benchmark_pd = (
                    benchmark_returns_df.select(
                        [self.factor_datetime_col, benchmark_symbol]
                    )
                    .to_pandas()
                    .set_index(self.factor_datetime_col)[benchmark_symbol]
                    .fillna(0.0)
                )
                # Align benchmark and strategy returns
                returns_pd, benchmark_pd = returns_pd.align(
                    benchmark_pd, join="left", axis=0
                )
                benchmark_pd = benchmark_pd.fillna(0.0)

        try:
            # Generate a dictionary of all QuantStats metrics
            self.performance_metrics = qs.reports.metrics(
                returns=returns_pd,
                benchmark=benchmark_pd,
                rf=self.config.risk_free_rate,
                display=False,
                mode="full",
            )
            # Clean up NaN/inf values for JSON serialization
            self.performance_metrics = {
                k: (
                    None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v
                )
                for k, v in self.performance_metrics.to_dict().items()
            }
            return True
        except Exception as e:
            self._write_log(
                f"Error during QuantStats metrics calculation: {e}", level=ERROR
            )
            self.performance_metrics = None
            return False

    def generate_report_data(
        self,
        factor_instance: FactorTemplate,
        analysis_start_dt: datetime,
        analysis_end_dt: datetime,
        tested_vt_symbols: list[str],
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

        if self.quantile_analysis_results:
            report_data["analysis_results"]["quantile_analysis"] = {
                "overall_average": self.quantile_analysis_results.overall_average.to_dicts()
            }
        if self.long_short_stats:
            report_data["analysis_results"][
                "long_short_portfolio_statistics_script"
            ] = asdict(self.long_short_stats)
        if self.performance_metrics:
            report_data["analysis_results"]["performance_metrics_quantstats"] = (
                self.performance_metrics
            )

        return report_data

    def save_json_report(
        self, report_data: dict[str, Any], report_filename_prefix: str
    ) -> Path | None:
        """Saves the report data dictionary as a JSON file."""
        if not report_data:
            self._write_log(
                "Report data is empty, skipping JSON report.", level=WARNING
            )
            return None

        factor_key_safe = safe_filename(report_data.get("factor_key", "unknown_factor"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_filename_prefix}_{factor_key_safe}_{timestamp}.json"
        filepath = self.output_data_dir / "json_reports" / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            self._write_log(f"JSON report saved successfully: {filepath}", level=INFO)
            return filepath
        except Exception as e:
            self._write_log(f"Error saving JSON report: {e}", level=ERROR)
            return None

    def generate_html_report(
        self,
        factor_key: str,
        benchmark_prices_df: pl.DataFrame | None = None,
        benchmark_symbol: str | None = None,
    ) -> Path | None:
        """Generates and saves a comprehensive HTML report using QuantStats."""
        self._write_log("Generating interactive HTML report...", level=DEBUG)
        if self.long_short_portfolio_returns_df is None:
            self._write_log(
                "L/S portfolio returns not available for HTML report.", level=ERROR
            )
            return None

        returns_pd = (
            self.long_short_portfolio_returns_df.to_pandas()
            .set_index(self.factor_datetime_col)["ls_return"]
            .fillna(0.0)
        )

        benchmark_pd = None
        if benchmark_prices_df is not None and benchmark_symbol:
            benchmark_returns_df = self._prepare_symbol_returns(
                benchmark_prices_df, [benchmark_symbol]
            )
            if benchmark_returns_df is not None:
                benchmark_pd = (
                    benchmark_returns_df.to_pandas()
                    .set_index(self.factor_datetime_col)[benchmark_symbol]
                    .fillna(0.0)
                )

        factor_key_safe = safe_filename(factor_key)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{factor_key_safe}_{timestamp}.html"
        filepath = self.output_data_dir / "html_reports" / filename

        try:
            qs.reports.html(
                returns=returns_pd,
                benchmark=benchmark_pd,
                rf=self.config.risk_free_rate,
                periods_per_year=self.config.annualization_factor,
                title=f"Factor Analysis Report: {factor_key}",
                output=str(filepath),
                download_filename=filename,
            )
            # https://github.com/ranaroussi/quantstats/issues/381
            self._write_log(f"HTML report saved successfully: {filepath}", level=INFO)
            webbrowser.open(f"file://{filepath.resolve()}")
            return filepath
        except Exception as e:
            self._write_log(
                f"Error generating HTML report with QuantStats: {e}", level=ERROR
            )
            return None

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
        # New optional parameters for enhanced functionality
        benchmark_prices_df: pl.DataFrame | None = None,
        benchmark_symbol: str | None = None,
    ) -> Path | None:
        """
        Runs the full analysis pipeline and generates JSON and HTML reports.

        This method maintains the original API but adds optional parameters
        for benchmark analysis.

        Args:
            factor_data_df (pl.DataFrame): DataFrame with datetime index and factor values for symbols.
            market_close_prices_df (pl.DataFrame): DataFrame with market close prices.
            factor_instance (FactorTemplate): The factor instance for metadata.
            analysis_start_dt (datetime): Start datetime for the analysis period.
            analysis_end_dt (datetime): End datetime for the analysis period.
            num_quantiles (int): Number of quantiles for quantile analysis.
            long_short_percentile (float): Percentile for long/short portfolio construction.
            report_filename_prefix (str): Prefix for the saved report files.
            benchmark_prices_df (Optional[pl.DataFrame]): Optional DataFrame of benchmark prices.
            benchmark_symbol (Optional[str]): The column name of the benchmark in benchmark_prices_df.

        Returns:
            Optional[Path]: The path to the saved JSON report, or None if failed.
        """
        self._write_log(
            f"--- Starting Full Analysis for Factor: {factor_instance.factor_key} ---",
            level=INFO,
        )

        # Update config with parameters from this specific run
        self.config.num_quantiles = num_quantiles
        self.config.long_short_percentile = long_short_percentile

        # --- 1. Data Validation and Preparation ---
        if factor_data_df.is_empty() or market_close_prices_df.is_empty():
            self._write_log(
                "Input factor or market price data is empty. Aborting.", level=ERROR
            )
            return None

        symbols_of_interest = [
            col for col in factor_data_df.columns if col != self.factor_datetime_col
        ]
        if not symbols_of_interest:
            self._write_log("No symbols found in factor data. Aborting.", level=ERROR)
            return None

        # --- 2. Calculate Forward Returns ---
        symbol_returns_df = self._prepare_symbol_returns(
            market_close_prices_df, symbols_of_interest
        )
        if symbol_returns_df is None or symbol_returns_df.is_empty():
            self._write_log(
                "Symbol returns preparation failed. Aborting analysis.", level=ERROR
            )
            return None

        # --- 3. Align DataFrames ---
        # Align factor data with forward returns to ensure correct time matching
        aligned_factor_df, aligned_returns_df = pl.align_frames(
            factor_data_df, symbol_returns_df, on=self.factor_datetime_col, how="inner"
        )

        if aligned_factor_df.is_empty():
            self._write_log(
                "Data is empty after aligning factors and returns. Check for datetime mismatches.",
                level=WARNING,
            )
            return None

        actual_start_dt = aligned_factor_df.get_column(self.factor_datetime_col).min()
        actual_end_dt = aligned_factor_df.get_column(self.factor_datetime_col).max()

        # --- 4. Prepare Final Analysis DataFrame ---
        analysis_data = self._prepare_analysis_data(
            aligned_factor_df, aligned_returns_df
        )
        if analysis_data is None or analysis_data.is_empty():
            self._write_log("Analysis data preparation failed. Aborting.", level=ERROR)
            return None

        # --- 5. Perform Core Analyses ---
        self.perform_quantile_analysis(analysis_data)
        self.perform_long_short_analysis(analysis_data)
        self.calculate_performance_metrics(benchmark_prices_df, benchmark_symbol)

        # --- 6. Generate and Save Reports ---
        report_content = self.generate_report_data(
            factor_instance=factor_instance,
            analysis_start_dt=actual_start_dt or analysis_start_dt,
            analysis_end_dt=actual_end_dt or analysis_end_dt,
            tested_vt_symbols=symbols_of_interest,
        )

        json_report_path = self.save_json_report(report_content, report_filename_prefix)
        self.generate_html_report(
            factor_instance.factor_key, benchmark_prices_df, benchmark_symbol
        )

        self._write_log(
            f"--- Analysis for {factor_instance.factor_key} Complete. ---", level=INFO
        )

        # Return path to JSON report for backward compatibility
        return json_report_path

    def _write_log(self, msg: str, level: int = INFO) -> None:
        contextual_logger = logger.bind(gateway_name=self.engine_name)
        if level == DEBUG:
            contextual_logger.debug(msg)
        elif level == INFO:
            contextual_logger.info(msg)
        elif level == WARNING:
            contextual_logger.warning(msg)
        elif level == ERROR:
            contextual_logger.error(msg)
        else:
            contextual_logger.info(msg)

    def close(self) -> None:
        self._write_log("FactorAnalyser closed.", level=INFO)
