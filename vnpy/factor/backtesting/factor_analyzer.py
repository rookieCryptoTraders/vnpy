from dataclasses import dataclass
import json
import traceback
from datetime import datetime
from logging import INFO, DEBUG, WARNING, ERROR 
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import re

# Third-party imports
import numpy as np
import pandas as pd 
import polars as pl
import quantstats as qs
from scipy import stats

# VnTrader imports
from vnpy.factor.template import FactorTemplate # For type hinting factor_instance
from vnpy.factor.base import APP_NAME
from vnpy.factor.setting import get_factor_path

# Default datetime column name
DEFAULT_DATETIME_COL = "datetime"

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(APP_NAME + "_FactorAnalyser")

def safe_filename(name: str) -> str:
    name = re.sub(r'[^\w\.\-@]', '_', name)
    return name

class FactorAnalyser:
    """
    Analyses pre-calculated factor data against market returns.
    """
    engine_name = APP_NAME + "FactorAnalyser"

    def __init__(self, output_data_dir_for_reports: Optional[str] = None):
        self.output_data_dir: Path = Path(output_data_dir_for_reports) if output_data_dir_for_reports else get_factor_path("factor_reports")
        self.factor_datetime_col: str = DEFAULT_DATETIME_COL
        self.vt_symbols: List[str] = [] 

        # Analysis results attributes
        self.symbol_returns_df: Optional[pl.DataFrame] = None
        self.quantile_analysis_results: Optional[Dict[str, pl.DataFrame]] = None
        self.long_short_portfolio_returns_df: Optional[pl.DataFrame] = None
        self.long_short_stats: Optional[Dict[str, float]] = None
        self.performance_metrics: Optional[Dict[str, Any]] = None
        
        self._write_log(f"FactorAnalyser initialized. Report dir: {self.output_data_dir}", level=INFO)
        self._prepare_output_directory()

    def _prepare_output_directory(self) -> None:
        try:
            self.output_data_dir.mkdir(parents=True, exist_ok=True)
            self._write_log(f"Report directory ensured at: {self.output_data_dir}", level=INFO)
        except OSError as e:
            self._write_log(f"Error creating report directory {self.output_data_dir}: {e}. Reports may fail to save.", level=ERROR)

    def prepare_symbol_returns(
        self,
        market_close_prices_df: pl.DataFrame, 
        reference_datetime_series: pl.Series 
    ) -> bool:
        self._write_log("Preparing symbol forward returns data...", level=INFO)
        if market_close_prices_df.is_empty() or self.factor_datetime_col not in market_close_prices_df.columns:
            self._write_log("Market close prices data is invalid or missing datetime column.", level=ERROR); return False
        if reference_datetime_series.is_empty():
            self._write_log("Reference datetime series is empty.", level=ERROR); return False
        
        # Derive vt_symbols from the market_close_prices_df columns (excluding datetime)
        # These are the symbols for which returns can potentially be calculated.
        available_symbols = [col for col in market_close_prices_df.columns if col != self.factor_datetime_col]
        if not available_symbols:
            self._write_log("No symbols found in market_close_prices_df (excluding datetime column).", level=ERROR); return False
        
        # Align close_prices_df with the reference_datetime_series
        # This ensures returns are calculated only for timestamps present in the factor data.
        aligned_close_prices = market_close_prices_df.join(
            reference_datetime_series.to_frame(self.factor_datetime_col), # Ensure reference is a frame for join
            on=self.factor_datetime_col,
            how="inner" # Keep only common datetimes
        )
        if aligned_close_prices.is_empty():
            self._write_log("Close prices DataFrame empty after aligning with reference datetimes. Cannot prepare returns.", level=ERROR); return False

        # Use symbols that are actually present in the aligned_close_prices for returns calculation
        self.vt_symbols = [col for col in aligned_close_prices.columns if col != self.factor_datetime_col]
        if not self.vt_symbols:
            self._write_log("No symbols remaining in aligned_close_prices_df after join. Cannot prepare returns.", level=ERROR); return False

        returns_cols = [pl.col(self.factor_datetime_col)]
        for symbol in self.vt_symbols:
            # Check if symbol column exists (it should, due to how vt_symbols is now derived)
            if symbol not in aligned_close_prices.columns: continue 
            returns_cols.append(
                ((pl.col(symbol).shift(-1) / pl.col(symbol)) - 1).fill_nan(0.0).fill_null(0.0).alias(symbol)
            )
        if len(returns_cols) <= 1: # Only datetime column means no symbols were processed
            self._write_log("No symbols were processed for returns calculation (returns_cols only has datetime).", level=ERROR)
            self.symbol_returns_df = None
            return False
        try:
            self.symbol_returns_df = aligned_close_prices.select(returns_cols).sort(self.factor_datetime_col)
            self._write_log(f"Symbol forward returns prepared. Shape: {self.symbol_returns_df.shape}", level=INFO)
            return True
        except Exception as e:
            self._write_log(f"Error calculating symbol forward returns: {e}", level=ERROR)
            self.symbol_returns_df = None
            return False

    def perform_quantile_analysis(self, factor_data_df: pl.DataFrame, num_quantiles: int = 5, returns_look_ahead_period: int = 1) -> bool:
        self._write_log(f"Performing quantile analysis: {num_quantiles} quantiles, using {returns_look_ahead_period}-period forward returns.", level=INFO)
        if factor_data_df is None or factor_data_df.is_empty(): self._write_log("Factor data empty for quantile analysis.", level=ERROR); return False
        if self.symbol_returns_df is None or self.symbol_returns_df.is_empty(): self._write_log("Symbol returns data empty for quantile analysis.", level=ERROR); return False
        
        # Symbols for analysis should be common to factor_data_df and self.vt_symbols (derived from returns data)
        symbols_in_factor = [col for col in factor_data_df.columns if col != self.factor_datetime_col]
        symbols_for_analysis = [sym for sym in symbols_in_factor if sym in self.vt_symbols]

        if not symbols_for_analysis: self._write_log("No common symbols between factor data and returns data for quantile analysis.", level=ERROR); return False
        if returns_look_ahead_period != 1: self._write_log(f"Warning: Quantile analysis with look_ahead={returns_look_ahead_period}, ensure symbol_returns_df is prepared accordingly.", level=WARNING)
        
        factor_long_df = factor_data_df.select([self.factor_datetime_col] + symbols_for_analysis).melt(id_vars=[self.factor_datetime_col], value_vars=symbols_for_analysis, variable_name="symbol", value_name="factor_value").drop_nulls(subset=["factor_value"])
        returns_long_df = self.symbol_returns_df.select([self.factor_datetime_col] + symbols_for_analysis).melt(id_vars=[self.factor_datetime_col], value_vars=symbols_for_analysis, variable_name="symbol", value_name="forward_return")
        merged_df = factor_long_df.join(returns_long_df, on=[self.factor_datetime_col, "symbol"], how="inner").drop_nulls(subset=["forward_return"])
        if merged_df.is_empty(): self._write_log("No valid data after merging for quantile analysis.", level=ERROR); return False

        quantiled_df = merged_df.with_columns(
            ((pl.col("factor_value").rank(method="average").over(self.factor_datetime_col) -1 ) * num_quantiles / pl.col("symbol").count().over(self.factor_datetime_col)).floor().cast(pl.Int8) + 1 
            .alias("quantile")
        )
        if quantiled_df.is_empty(): self._write_log("Quantiled DataFrame empty.", level=ERROR); return False
        
        quantile_returns_by_time_df = quantiled_df.group_by([self.factor_datetime_col, "quantile"]).agg(pl.col("forward_return").mean().alias("mean_quantile_return")).sort([self.factor_datetime_col, "quantile"])
        average_quantile_returns_overall_df = quantile_returns_by_time_df.group_by("quantile").agg(pl.col("mean_quantile_return").mean().alias("average_return"), pl.col("mean_quantile_return").std().alias("std_dev_return"), pl.col("mean_quantile_return").count().alias("count_periods")).sort("quantile")
        self.quantile_analysis_results = {"by_time": quantile_returns_by_time_df, "overall_average": average_quantile_returns_overall_df}
        self._write_log(f"Quantile analysis completed. Overall:\n{average_quantile_returns_overall_df if not average_quantile_returns_overall_df.is_empty() else 'Empty'}", level=INFO)
        return True


    def perform_long_short_analysis(self, factor_data_df: pl.DataFrame, returns_look_ahead_period: int = 1, long_percentile_threshold: float = 0.7, short_percentile_threshold: float = 0.3) -> bool:
        self._write_log(f"Performing L/S analysis. Long > {long_percentile_threshold*100:.0f}th, Short <= {short_percentile_threshold*100:.0f}th.", level=INFO)
        if factor_data_df is None or factor_data_df.is_empty(): self._write_log("Factor data empty for L/S analysis.", level=ERROR); return False
        if self.symbol_returns_df is None or self.symbol_returns_df.is_empty(): self._write_log("Symbol returns empty for L/S analysis.", level=ERROR); return False
        
        symbols_in_factor = [col for col in factor_data_df.columns if col != self.factor_datetime_col]
        symbols_for_analysis = [sym for sym in symbols_in_factor if sym in self.vt_symbols]

        if not symbols_for_analysis: self._write_log("No common symbols for L/S analysis.", level=ERROR); return False
        if returns_look_ahead_period != 1: self._write_log(f"Warning: L/S analysis with look_ahead={returns_look_ahead_period}, ensure symbol_returns_df is prepared accordingly.", level=WARNING)

        factor_long_df = factor_data_df.select([self.factor_datetime_col] + symbols_for_analysis).melt(id_vars=[self.factor_datetime_col], value_vars=symbols_for_analysis, variable_name="symbol", value_name="factor_value").drop_nulls(subset=["factor_value"])
        returns_long_df = self.symbol_returns_df.select([self.factor_datetime_col] + symbols_for_analysis).melt(id_vars=[self.factor_datetime_col], value_vars=symbols_for_analysis, variable_name="symbol", value_name="forward_return")
        merged_df = factor_long_df.join(returns_long_df, on=[self.factor_datetime_col, "symbol"], how="inner").drop_nulls(subset=["forward_return"])
        if merged_df.is_empty(): self._write_log("No valid data after merging for L/S analysis.", level=ERROR); return False

        ranked_df = merged_df.with_columns([(pl.col("factor_value").rank(method="average").over(self.factor_datetime_col) / pl.col("symbol").count().over(self.factor_datetime_col)).alias("rank_percentile")])
        portfolio_df = ranked_df.with_columns([(pl.col("rank_percentile") > long_percentile_threshold).alias("long_leg"),(pl.col("rank_percentile") <= short_percentile_threshold).alias("short_leg")])
        long_ret = portfolio_df.filter(pl.col("long_leg")).group_by(self.factor_datetime_col).agg(pl.col("forward_return").mean().alias("mean_long_return"))
        short_ret = portfolio_df.filter(pl.col("short_leg")).group_by(self.factor_datetime_col).agg(pl.col("forward_return").mean().alias("mean_short_return"))
        ls_calc_df = long_ret.join(short_ret, on=self.factor_datetime_col, how="outer").sort(self.factor_datetime_col)
        ls_calc_df = ls_calc_df.with_columns([pl.col("mean_long_return").fill_null(0.0), pl.col("mean_short_return").fill_null(0.0)]).with_columns((pl.col("mean_long_return") - pl.col("mean_short_return")).alias("ls_portfolio_return"))
        self.long_short_portfolio_returns_df = ls_calc_df.select([self.factor_datetime_col, "ls_portfolio_return", "mean_long_return", "mean_short_return"])

        if self.long_short_portfolio_returns_df.is_empty(): self._write_log("L/S portfolio returns empty.", level=ERROR); self.long_short_stats = {"t_statistic": float('nan'), "p_value": float('nan'), "mean_return": float('nan')}; return False
        
        port_rets_ttest = self.long_short_portfolio_returns_df["ls_portfolio_return"].drop_nulls()
        if port_rets_ttest.len() >= 2:
            mean_ret = port_rets_ttest.mean()
            if mean_ret is None: self.long_short_stats = {"t_statistic": float('nan'), "p_value": float('nan'), "mean_return": float('nan')}
            else: t_stat, p_val = stats.ttest_1samp(port_rets_ttest.to_numpy(), 0, nan_policy='omit'); self.long_short_stats = {"t_statistic": t_stat, "p_value": p_val, "mean_return": mean_ret}
            self._write_log(f"L/S Stats: {self.long_short_stats}", level=INFO)
        else: self.long_short_stats = {"t_statistic": float('nan'), "p_value": float('nan'), "mean_return": port_rets_ttest.mean() if not port_rets_ttest.is_empty() else float('nan')}; self._write_log("Not enough L/S returns for t-test.", level=WARNING)
        return True

    def calculate_performance_metrics(self) -> bool:
        self._write_log("Calculating L/S portfolio performance metrics (quantstats)...", level=INFO)
        if self.long_short_portfolio_returns_df is None or self.long_short_portfolio_returns_df.is_empty(): self._write_log("L/S portfolio returns empty for metrics.", level=ERROR); return False
        try:
            returns_pl_filled = self.long_short_portfolio_returns_df.sort(self.factor_datetime_col).with_columns(pl.col("ls_portfolio_return").fill_null(0.0))
            portfolio_pd_df = returns_pl_filled.select([self.factor_datetime_col, "ls_portfolio_return"]).to_pandas().set_index(self.factor_datetime_col)
            returns_series_pd = portfolio_pd_df["ls_portfolio_return"].astype(float)
            if returns_series_pd.empty: self._write_log("Pandas returns series empty for metrics.", level=ERROR); return False
            if returns_series_pd.isnull().any() or np.isinf(returns_series_pd).any(): returns_series_pd = returns_series_pd.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        except Exception as e: self._write_log(f"Error preparing data for quantstats: {e}", level=ERROR); return False

        try:
            metrics = {'sharpe': qs.stats.sharpe(returns_series_pd, smart=True), 'max_drawdown': qs.stats.max_drawdown(returns_series_pd), 'cumulative_returns_total': qs.stats.comp(returns_series_pd).iloc[-1], 'cagr': qs.stats.cagr(returns_series_pd), 'volatility_annualized': qs.stats.volatility(returns_series_pd, annualize=True), 'win_rate': qs.stats.win_rate(returns_series_pd), 'avg_win_return': qs.stats.avg_win(returns_series_pd), 'avg_loss_return': qs.stats.avg_loss(returns_series_pd), 'sortino': qs.stats.sortino(returns_series_pd, smart=True), 'calmar': qs.stats.calmar(returns_series_pd), 'skew': qs.stats.skew(returns_series_pd), 'kurtosis': qs.stats.kurtosis(returns_series_pd), 'value_at_risk_daily': qs.stats.value_at_risk(returns_series_pd), 'expected_shortfall_daily': qs.stats.conditional_value_at_risk(returns_series_pd), 'profit_factor': qs.stats.profit_factor(returns_series_pd), 'common_sense_ratio': qs.stats.common_sense_ratio(returns_series_pd), 'information_ratio': qs.stats.information_ratio(returns_series_pd)}
            self.performance_metrics = {k: (None if isinstance(v, (float, np.generic)) and (np.isnan(v) or np.isinf(v)) else (v.to_dict() if isinstance(v, pd.Series) else v)) for k, v in metrics.items()}
            self._write_log(f"Performance metrics calculated. Sharpe: {self.performance_metrics.get('sharpe', 'N/A'):.3f}", level=INFO)
            return True
        except Exception as e: self._write_log(f"Error during quantstats calculation: {e}", level=ERROR); self.performance_metrics = None; return False


    def generate_report_data(
        self,
        factor_key: str,
        factor_class_name: str,
        factor_parameters: Dict[str, Any],
        analysis_start_dt: datetime,
        analysis_end_dt: datetime,
        tested_vt_symbols: List[str]
        ) -> Dict[str, Any]:
        self._write_log("Generating report data dictionary...", level=INFO)
        report_data: Dict[str, Any] = {
            "factor_key": factor_key,
            "factor_class_name": factor_class_name,
            "factor_parameters": factor_parameters,
            "backtest_run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vt_symbols_tested": tested_vt_symbols,
            "data_period_analyzed": {
                "start_datetime": analysis_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end_datetime": analysis_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "analysis_results": {}
        }
        if self.quantile_analysis_results:
            qa_overall = self.quantile_analysis_results.get("overall_average")
            report_data["analysis_results"]["quantile_analysis"] = {
                "overall_average_returns_per_quantile": qa_overall.to_dicts() if qa_overall is not None and not qa_overall.is_empty() else None,
            }
        if self.long_short_stats:
            report_data["analysis_results"]["long_short_portfolio_statistics"] = self.long_short_stats
        if self.performance_metrics:
            report_data["analysis_results"]["performance_metrics_quantstats"] = self.performance_metrics
        
        if not report_data["analysis_results"]:
            self._write_log("No analysis results were available for the report.", level=WARNING)
        return report_data

    def save_report(self, report_data: Dict[str, Any], report_filename_prefix: str = "factor_analysis_report") -> Optional[Path]:
        if not report_data: self._write_log("Report data empty. Nothing to save.", level=WARNING); return None
        if not self.output_data_dir.exists():
            try: self.output_data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e: self._write_log(f"Error creating report dir {self.output_data_dir}: {e}", level=ERROR); return None

        factor_key_safe = safe_filename(report_data.get("factor_key", "unknown_factor"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_filename_prefix}_{factor_key_safe}_{timestamp}.json"
        filepath = self.output_data_dir.joinpath(filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False, default=str)
            self._write_log(f"Report saved: {filepath}", level=INFO)
            return filepath
        except Exception as e:
            self._write_log(f"Error saving report to {filepath}: {e}", level=ERROR); return None

    def run_analysis_and_report(
        self,
        factor_data_df: pl.DataFrame, 
        market_close_prices_df: pl.DataFrame, 
        factor_instance: FactorTemplate, 
        analysis_start_dt: datetime, 
        analysis_end_dt: datetime,   
        num_quantiles: int = 5,
        returns_look_ahead_period: int = 1,
        long_percentile_threshold: float = 0.7,
        short_percentile_threshold: float = 0.3,
        report_filename_prefix: str = "factor_analysis_report"
    ) -> Optional[Path]:
        """
        Runs all analyses on pre-computed factor data and generates a report.
        """
        self._write_log(f"Starting analysis for factor: {factor_instance.factor_key}", level=INFO)
        
        if factor_data_df.is_empty():
            self._write_log("Input factor_data_df is empty. Cannot perform analysis.", level=ERROR); return None
        if market_close_prices_df.is_empty():
            self._write_log("Input market_close_prices_df is empty. Cannot perform analysis.", level=ERROR); return None

        # Use symbols from factor_data_df for analysis consistency, as it's the primary subject
        self.vt_symbols = [col for col in factor_data_df.columns if col != self.factor_datetime_col]
        if not self.vt_symbols:
            self._write_log("No symbols found in factor_data_df (excluding datetime column). Cannot perform analysis.", level=ERROR); return None
        self._write_log(f"Analyser using symbols derived from factor data: {self.vt_symbols}", level=DEBUG)

        reference_dt_series = factor_data_df[self.factor_datetime_col]

        if not self.prepare_symbol_returns(market_close_prices_df, reference_dt_series):
            self._write_log("Preparing symbol returns failed. Subsequent analyses might be affected or skipped.", level=ERROR)
        else:
            self.perform_quantile_analysis(factor_data_df, num_quantiles, returns_look_ahead_period)
            self.perform_long_short_analysis(factor_data_df, returns_look_ahead_period, long_percentile_threshold, short_percentile_threshold)
            if self.long_short_portfolio_returns_df is not None and not self.long_short_portfolio_returns_df.is_empty():
                self.calculate_performance_metrics()
            else:
                self._write_log("Skipping performance metrics: L/S portfolio returns unavailable.", INFO)

        report_content = self.generate_report_data(
            factor_key=factor_instance.factor_key,
            factor_class_name=factor_instance.__class__.__name__,
            factor_parameters=factor_instance.get_params(),
            analysis_start_dt=analysis_start_dt,
            analysis_end_dt=analysis_end_dt,
            tested_vt_symbols=self.vt_symbols # These are the symbols used in the analysis
        )
        
        saved_path = None
        if report_content:
            saved_path = self.save_report(report_content, report_filename_prefix)
        
        self._write_log(f"Analysis and reporting for {factor_instance.factor_key} completed.", level=INFO)
        return saved_path


    def _write_log(self, msg: str, level: int = INFO) -> None:
        log_msg = f"[{self.engine_name}] {msg}"
        level_map = {
            DEBUG: logger.debug, INFO: logger.info,
            WARNING: logger.warning, ERROR: logger.error
        }
        log_func = level_map.get(level, logger.info) 
        log_func(log_msg)

    def close(self) -> None: 
        self._write_log("FactorAnalyser closed.", level=INFO)
