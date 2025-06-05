import json
from datetime import datetime
from logging import INFO, DEBUG, WARNING, ERROR
from typing import Any
from pathlib import Path
import re

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
    name = re.sub(r"[^\w\.\-@]", "_", name)
    return name

class FactorAnalyser:
    engine_name = APP_NAME + "FactorAnalyser"

    def __init__(self, output_data_dir_for_reports: str | None = None):
        self.output_data_dir: Path = (
            Path(output_data_dir_for_reports)
            if output_data_dir_for_reports
            else get_backtest_report_path()
        )
        self.factor_datetime_col: str = DEFAULT_DATETIME_COL
        self.quantile_analysis_results: dict[str, pl.DataFrame] | None = None
        self.long_short_portfolio_returns_df: pl.DataFrame | None = None
        self.long_short_stats: dict[str, float | None] | None = None
        self.performance_metrics: dict[str, Any] | None = None
        self._write_log(f"FactorAnalyser initialized. Report dir: {self.output_data_dir}", level=INFO)
        self._prepare_output_directory()

    def _prepare_output_directory(self) -> None:
        try:
            self.output_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._write_log(f"Error creating report directory {self.output_data_dir}: {e}.", level=ERROR)

    def _prepare_symbol_returns(self, market_close_prices_df: pl.DataFrame, symbols_of_interest: list[str]) -> pl.DataFrame | None:
        self._write_log("Preparing 1-period symbol forward returns data...", level=INFO)
        if market_close_prices_df.is_empty() or self.factor_datetime_col not in market_close_prices_df.columns:
            self._write_log("Invalid market close prices data.", level=ERROR)
            return None
        if not symbols_of_interest:
            self._write_log("No symbols provided for returns calculation.", level=WARNING)
            return None
        try:
            forward_returns_expressions = []
            for symbol in symbols_of_interest:
                fwd_ret = (pl.col(symbol).shift(-1) - pl.col(symbol)) / pl.col(symbol)
                fwd_ret = pl.when(fwd_ret.is_nan() | fwd_ret.is_infinite()).then(None).otherwise(fwd_ret)
                forward_returns_expressions.append(fwd_ret.alias(symbol))

            calculated_returns_df = market_close_prices_df.select(
                [pl.col(self.factor_datetime_col)] + forward_returns_expressions
            )

            # Truncate last row if it contains nulls due to shift
            if calculated_returns_df.select(pl.col(symbols_of_interest[0]).tail(1).is_null()).item():
                calculated_returns_df = calculated_returns_df[:-1]

            if calculated_returns_df.is_empty():
                self._write_log("Calculated returns DataFrame is empty.", level=WARNING)
                return None

            self._write_log(f"Symbol forward returns prepared for {len(symbols_of_interest)} symbols.", level=INFO)
            return calculated_returns_df
        except Exception as e:
            self._write_log(f"Error calculating symbol forward returns: {e}", level=ERROR)
            return None

    def _prepare_analysis_data(self, aligned_factor_df: pl.DataFrame, aligned_symbol_returns_df: pl.DataFrame) -> pl.DataFrame | None:
        self._write_log("Preparing analysis data...", level=INFO)
        symbols = [col for col in aligned_factor_df.columns if col != self.factor_datetime_col]
        if not symbols:
            self._write_log("No symbols available for analysis data preparation.", level=ERROR)
            return None
        factor_long = aligned_factor_df.select([self.factor_datetime_col] + symbols).melt(
            id_vars=[self.factor_datetime_col], value_vars=symbols, variable_name="symbol", value_name="factor_value"
        )
        returns_long = aligned_symbol_returns_df.select([self.factor_datetime_col] + symbols).melt(
            id_vars=[self.factor_datetime_col], value_vars=symbols, variable_name="symbol", value_name="forward_return"
        )
        data = factor_long.join(returns_long, on=["datetime", "symbol"], how="inner")
        data = data.filter(pl.col("factor_value").is_not_null() & pl.col("forward_return").is_not_null())
        if data.is_empty():
            self._write_log("Analysis data empty after preparation.", level=ERROR)
            return None
        self._write_log("Analysis data prepared successfully.", level=INFO)
        return data

    def perform_quantile_analysis(self, analysis_data: pl.DataFrame, num_quantiles: int = 5) -> bool:
        self._write_log(f"Performing quantile analysis ({num_quantiles} quantiles)...", level=INFO)
        if analysis_data.is_empty():
            self._write_log("Analysis data is empty for quantile analysis.", level=ERROR)
            return False
        data = analysis_data.sort(["datetime", "factor_value"])
        n_assets_df = data.group_by("datetime").agg(pl.count("symbol").alias("n_assets"))
        data = data.join(n_assets_df, on="datetime").with_columns(
            pl.arange(0, pl.count()).over("datetime").alias("row_num")
        ).with_columns(
            (((pl.col("row_num") + 1) * num_quantiles / pl.col("n_assets")).ceil().cast(pl.Int8)).alias("quantile")
        )
        quantile_returns_by_time_df = data.group_by(["datetime", "quantile"]).agg(
            pl.col("forward_return").mean().alias("mean_quantile_return")
        ).sort(["datetime", "quantile"])
        average_quantile_returns_overall_df = quantile_returns_by_time_df.group_by("quantile").agg(
            pl.col("mean_quantile_return").mean().alias("avg_return"),
            pl.col("mean_quantile_return").std().alias("std_return"),
            pl.col("mean_quantile_return").count().alias("count_periods")
        ).sort("quantile")
        self.quantile_analysis_results = {
            "by_time": quantile_returns_by_time_df,
            "overall_average": average_quantile_returns_overall_df
        }
        self._write_log("Quantile analysis completed.", level=INFO)
        return True

    def perform_long_short_analysis(self, analysis_data: pl.DataFrame, long_short_percentile: float = 0.3) -> bool:
        self._write_log(f"Performing L/S analysis (top/bottom {long_short_percentile * 100:.0f}%)...", level=INFO)
        if analysis_data.is_empty():
            self._write_log("Analysis data is empty for L/S analysis.", level=ERROR)
            return False
        data = analysis_data.sort(["datetime", "factor_value"])
        n_assets_df = data.group_by("datetime").agg(pl.count("symbol").alias("n_assets"))
        data = data.join(n_assets_df, on="datetime").with_columns(
            pl.arange(0, pl.count()).over("datetime").alias("row_num")
        ).with_columns(
            (pl.col("row_num") >= (pl.col("n_assets") * (1 - long_short_percentile))).alias("is_long"),
            (pl.col("row_num") < (pl.col("n_assets") * long_short_percentile)).alias("is_short")
        )
        ls_returns_df = data.group_by("datetime").agg(
            pl.col("forward_return").filter(pl.col("is_long")).mean().alias("long_return").fill_null(0.0),
            pl.col("forward_return").filter(pl.col("is_short")).mean().alias("short_return").fill_null(0.0)
        ).with_columns(
            (pl.col("long_return") - pl.col("short_return")).alias("ls_return")
        ).sort("datetime")
        self.long_short_portfolio_returns_df = ls_returns_df
        if ls_returns_df.is_empty() or ls_returns_df.height < 2:
            self._write_log("L/S portfolio returns series too short.", level=WARNING)
            self.long_short_stats = {"mean_return": None, "std_return": None, "sharpe_ratio": None, "t_stat": None}
            return True
        mean_ls_ret = ls_returns_df.get_column("ls_return").mean()
        std_ls_ret = ls_returns_df.get_column("ls_return").std()
        sharpe_ratio = (mean_ls_ret / std_ls_ret) * (252**0.5) if std_ls_ret else None
        t_stat = (mean_ls_ret / (std_ls_ret / (ls_returns_df.shape[0]**0.5))) if std_ls_ret else None
        self.long_short_stats = {"mean_return": mean_ls_ret, "std_return": std_ls_ret, "sharpe_ratio": sharpe_ratio, "t_stat": t_stat}
        self._write_log(f"L/S Portfolio Stats: {self.long_short_stats}", level=INFO)
        return True

    def calculate_performance_metrics(self) -> bool:
        self._write_log("Calculating L/S portfolio performance metrics...", level=INFO)
        if not self.long_short_portfolio_returns_df or self.long_short_portfolio_returns_df.is_empty():
            self._write_log("L/S portfolio returns empty.", level=ERROR)
            self.performance_metrics = None
            return False
        try:
            returns_series_pd = self.long_short_portfolio_returns_df.select(
                [self.factor_datetime_col, pl.col("ls_return").fill_null(0.0)]
            ).sort(self.factor_datetime_col).to_pandas().set_index(self.factor_datetime_col)["ls_return"].astype(float)
            returns_series_pd = returns_series_pd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            metrics = {
                "sharpe_qs": qs.stats.sharpe(returns_series_pd, smart=True),
                "max_drawdown": qs.stats.max_drawdown(returns_series_pd),
                "cumulative_returns_total": qs.stats.comp(returns_series_pd).iloc[-1] if not qs.stats.comp(returns_series_pd).empty else 0.0,
                "cagr": qs.stats.cagr(returns_series_pd),
                "volatility_annualized": qs.stats.volatility(returns_series_pd, annualize=True),
                "win_rate": qs.stats.win_rate(returns_series_pd),
                "avg_win_return": qs.stats.avg_win(returns_series_pd),
                "avg_loss_return": qs.stats.avg_loss(returns_series_pd),
                "sortino_qs": qs.stats.sortino(returns_series_pd, smart=True),
                "calmar": qs.stats.calmar(returns_series_pd)
            }
            self.performance_metrics = {k: None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for k, v in metrics.items()}
            self._write_log("Performance metrics calculated.", level=INFO)
            return True
        except Exception as e:
            self._write_log(f"Error during QuantStats calculation: {e}", level=ERROR)
            self.performance_metrics = None
            return False

    def generate_report_data(self, factor_key: str, factor_class_name: str, factor_parameters: dict[str, Any], analysis_start_dt: datetime, analysis_end_dt: datetime, tested_vt_symbols: list[str]) -> dict[str, Any]:
        self._write_log("Generating report data...", level=INFO)
        report_data = {
            "factor_key": factor_key,
            "factor_class_name": factor_class_name,
            "factor_parameters": factor_parameters,
            "backtest_run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vt_symbols_tested": tested_vt_symbols,
            "data_period_analyzed": {
                "start_datetime": analysis_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end_datetime": analysis_end_dt.strftime("%Y-%m-%d %H:%M:%S")
            },
            "analysis_results": {}
        }
        if self.quantile_analysis_results:
            report_data["analysis_results"]["quantile_analysis"] = {
                "overall_average_returns_per_quantile": self.quantile_analysis_results["overall_average"].to_dicts()
                if not self.quantile_analysis_results["overall_average"].is_empty() else None
            }
        if self.long_short_stats:
            report_data["analysis_results"]["long_short_portfolio_statistics_script"] = self.long_short_stats
        if self.performance_metrics:
            report_data["analysis_results"]["performance_metrics_quantstats"] = self.performance_metrics
        return report_data

    def save_report(self, report_data: dict[str, Any], report_filename_prefix: str = "factor_analysis_report") -> Path | None:
        if not report_data or not self.output_data_dir.exists():
            self._write_log("Report data empty or directory issue.", level=WARNING)
            return None
        factor_key_safe = safe_filename(report_data.get("factor_key", "unknown_factor"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_filename_prefix}_{factor_key_safe}_{timestamp}.json"
        filepath = self.output_data_dir.joinpath(filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False, default=lambda o: o.isoformat() if isinstance(o, datetime) else None)
            self._write_log(f"Report saved: {filepath}", level=INFO)
            return filepath
        except Exception as e:
            self._write_log(f"Error saving report: {e}", level=ERROR)
            return None

    def run_analysis_and_report(self, factor_data_df: pl.DataFrame, market_close_prices_df: pl.DataFrame, factor_instance: FactorTemplate, analysis_start_dt: datetime, analysis_end_dt: datetime, num_quantiles: int = 5, long_short_percentile: float = 0.3, report_filename_prefix: str = "factor_analysis_report") -> Path | None:
        self._write_log(f"Starting analysis for factor: {factor_instance.factor_key}", level=INFO)
        if factor_data_df.is_empty() or market_close_prices_df.is_empty() or self.factor_datetime_col not in factor_data_df.columns or self.factor_datetime_col not in market_close_prices_df.columns:
            self._write_log("Invalid input data.", level=ERROR)
            return None
        symbols_of_interest = [col for col in factor_data_df.columns if col != self.factor_datetime_col]
        if not symbols_of_interest:
            self._write_log("No symbols in factor data.", level=ERROR)
            return None
        local_symbol_returns_df = self._prepare_symbol_returns(market_close_prices_df, symbols_of_interest)
        if local_symbol_returns_df.is_empty():
            self._write_log("Symbol returns preparation failed.", level=ERROR)
            return None
        factor_data_df_truncated = factor_data_df[:-1] if not factor_data_df.is_empty() else factor_data_df
        aligned_frames = pl.align_frames(factor_data_df_truncated, local_symbol_returns_df, on=self.factor_datetime_col, how="inner")
        aligned_factor_df, aligned_symbol_returns_df = aligned_frames[0], aligned_frames[1]
        if aligned_factor_df.is_empty():
            self._write_log("Data empty after alignment.", level=WARNING)
            return None
        analysis_data = self._prepare_analysis_data(aligned_factor_df, aligned_symbol_returns_df)
        if analysis_data.is_empty():
            self._write_log("Analysis data preparation failed.", level=ERROR)
            return None
        self.perform_quantile_analysis(analysis_data, num_quantiles)
        self.perform_long_short_analysis(analysis_data, long_short_percentile)
        if self.long_short_portfolio_returns_df.is_empty():
            self.calculate_performance_metrics()
        report_content = self.generate_report_data(
            factor_key=factor_instance.factor_key,
            factor_class_name=factor_instance.__class__.__name__,
            factor_parameters=factor_instance.get_nested_params_for_optimizer(),
            analysis_start_dt=aligned_factor_df.get_column(self.factor_datetime_col).min() or analysis_start_dt,
            analysis_end_dt=aligned_factor_df.get_column(self.factor_datetime_col).max() or analysis_end_dt,
            tested_vt_symbols=[col for col in aligned_factor_df.columns if col != self.factor_datetime_col]
        )
        return self.save_report(report_content, report_filename_prefix)

    def _write_log(self, msg: str, level: int = INFO) -> None:
        log_msg = f"[{self.engine_name}] {msg}"
        contextual_logger = logger.bind(gateway_name=self.engine_name)
        if level == DEBUG:
            contextual_logger.debug(log_msg)
        elif level == INFO:
            contextual_logger.info(log_msg)
        elif level == WARNING:
            contextual_logger.warning(log_msg)
        elif level == ERROR:
            contextual_logger.error(log_msg)
        else:
            contextual_logger.info(log_msg)

    def close(self) -> None:
        self._write_log("FactorAnalyser closed.", level=INFO)
