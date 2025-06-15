"""
Factor Optimization Script

This script demonstrates how to use the FactorOptimizer to find the best
parameters for a factor (e.g., MACD) using a grid search methodology.
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from loguru import logger

# --- VnPy Framework Imports ---
from vnpy.trader.constant import Interval
from vnpy.factor.backtesting.backtesting import BacktestEngine
from vnpy.factor.backtesting.optimizer import FactorOptimizer # Import the optimizer
from vnpy.factor.setting import (
    get_backtest_data_cache_path,
    get_backtest_report_path,
    get_factor_definitions_filename, # Use the updated setting function
)
from vnpy.trader.utility import load_json


# --- Helper Function for Fake Data Generation ---
# This function is the same as in your original script.
def generate_fake_ohlcv_wide_dict(
    start_date: datetime,
    end_date: datetime,
    interval: str,
    vt_symbols: list[str]
) -> dict[str, pl.DataFrame]:
    """
    Generates a dictionary of Polars DataFrames with realistic fake OHLCV data.
    """
    if not vt_symbols:
        return {}

    datetimes = pl.datetime_range(
        start=start_date, end=end_date, interval=interval, time_unit="us", eager=True, time_zone="UTC"
    )
    if len(datetimes) < 2:
        logger.warning("Date range is too short for data generation.")
        return {}

    n_dates = len(datetimes)
    n_symbols = len(vt_symbols)
    base_prices = 100 + np.arange(n_symbols) * 50
    drifts = np.random.uniform(0.05, 0.25, n_symbols)
    volatilities = np.random.uniform(0.3, 0.8, n_symbols)
    
    corr_matrix = np.full((n_symbols, n_symbols), 0.7)
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix += np.random.uniform(-0.15, 0.15, (n_symbols, n_symbols))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)

    std_devs = np.diag(volatilities)
    cov_matrix = std_devs @ corr_matrix @ std_devs
    dt = 1 / n_dates
    mean_returns = (drifts - 0.5 * volatilities**2) * dt
    random_shocks = np.random.multivariate_normal(np.zeros(n_symbols), cov_matrix, n_dates)
    
    log_returns = mean_returns + random_shocks * np.sqrt(dt)
    price_paths = base_prices * np.exp(np.cumsum(log_returns, axis=0))

    close_prices = price_paths
    open_prices = np.vstack([close_prices[0] * (1 + (np.random.rand(n_symbols) - 0.5) * 0.01), close_prices[:-1]])
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 0.01, (n_dates, n_symbols)) * close_prices
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 0.01, (n_dates, n_symbols)) * close_prices
    low_prices = np.maximum(low_prices, 0.001)

    price_change = np.abs(close_prices - open_prices) / open_prices
    base_volume = np.random.uniform(500, 1000, (n_dates, n_symbols))
    volume = base_volume * (1 + price_change * np.random.uniform(5, 15, n_symbols))

    all_data_frames = [
        pl.DataFrame({
            "datetime": datetimes, "symbol": symbol, "open": open_prices[:, i],
            "high": high_prices[:, i], "low": low_prices[:, i],
            "close": close_prices[:, i], "volume": volume[:, i],
        }) for i, symbol in enumerate(vt_symbols)
    ]
    if not all_data_frames: return {}
    
    flat_df = pl.concat(all_data_frames)
    ohlcv_dict = {
        ohlcv_type: flat_df.pivot(values=ohlcv_type, index="datetime", columns="symbol").sort("datetime")
        for ohlcv_type in ["open", "high", "low", "close", "volume"]
    }
    return ohlcv_dict


def main():
    """Main function to run the factor optimization."""
    
    # --- 1. Initialization ---
    logger.info("Initializing Backtest Engine and Factor Optimizer...")
    
    backtest_engine = BacktestEngine(
        factor_module_name="vnpy.factor.factors",
        output_data_dir_for_analyser_reports=get_backtest_report_path(),
        output_data_dir_for_calculator_cache=get_backtest_data_cache_path()
    )
    optimizer = FactorOptimizer(backtest_engine)

    # --- 2. Define Optimization Parameters ---
    logger.info("Defining optimization parameters...")
    
    # Symbols to be used in the optimization
    vt_symbols = [
        'btcusdt.BINANCE', 'ethusdt.BINANCE', 'xrpusdt.BINANCE', 'ltcusdt.BINANCE',
        'bchusdt.BINANCE', 'adausdt.BINANCE', 'solusdt.BINANCE', 'dogeusdt.BINANCE'
    ]

    # Load the factor definitions and select the MACD factor as a template
    factor_definitions_filename = get_factor_definitions_filename()
    factor_definitions = load_json(factor_definitions_filename)
    macd_factor_template = factor_definitions[2]  # Using the MACD factor definition dict

    # Define the grid of parameters to search over for the MACD factor.
    # The keys use dot-notation to target parameters in nested dependency factors.
    parameter_grid = {
        "fast_ema.period": [10, 12, 15],
        "slow_ema.period": [20, 26, 30],
        "signal_period":   [7, 9, 12],
    }

    # Define time range for the data
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=400)
    time_interval = Interval.HOUR

    # --- 3. Prepare Data ---
    # In a real scenario, the optimizer's _load_bar_data_engine would fetch data
    # from a database. For this script, we generate fake data and manually
    # place it into the backtest engine's memory. The optimizer will then use it.
    logger.info("Generating fake OHLCV data for the optimization...")
    ohlcv_data_dictionary = generate_fake_ohlcv_wide_dict(
        start_date=start_dt,
        end_date=end_dt,
        interval="1h",  # Use string format for data generator
        vt_symbols=vt_symbols
    )
    backtest_engine.memory_bar = ohlcv_data_dictionary
    backtest_engine.num_data_rows = backtest_engine.memory_bar["close"].height
    logger.info(f"Generated {backtest_engine.num_data_rows} rows of data.")

    # --- 4. Run the Optimization ---
    logger.info("Starting factor optimization process...")
    
    best_params, search_results, report_path = optimizer.optimize_factor(
        factor_definition_template=macd_factor_template,
        parameter_grid=parameter_grid,
        start_datetime=start_dt,       # These are now mainly for informational purposes
        end_datetime=end_dt,         # as the data is already in memory.
        vt_symbols=vt_symbols,
        data_interval=time_interval,
        factor_json_conf_path=str(factor_definitions_filename),
        test_size_ratio=0.3,
        num_quantiles=5,
        long_short_percentile=0.5,
        report_filename_prefix="macd_optimization_result"
    )

    # --- 5. Display Results ---
    if best_params:
        logger.success("--- Optimization Complete ---")
        logger.info(f"Best Parameters Found: {best_params}")
        logger.info(f"Best Sharpe Ratio (on training set): {search_results.get('best_score', 'N/A'):.4f}")
        logger.info(f"Final report for best parameters generated at: {report_path}")

        print("\n--- Grid Search Results (Top 5) ---")
        sorted_results = sorted(zip(search_results['params'], search_results['scores']), key=lambda item: item[1], reverse=True)
        for params, score in sorted_results[:5]:
            print(f"Params: {params} -> Sharpe Score: {score:.4f}")
    else:
        logger.error("--- Optimization Failed ---")
        logger.warning("Could not find suitable parameters or an error occurred during the process.")


if __name__ == "__main__":
    main()
