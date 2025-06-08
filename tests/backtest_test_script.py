"""
Backtest script converted from backtest_test.ipynb
This script demonstrates backtesting functionality using vnpy factor engine.
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from loguru import logger

from vnpy.factor.backtesting.backtesting import BacktestEngine
from vnpy.factor.setting import (
    get_backtest_data_cache_path,
    get_backtest_report_path,
    FACTOR_DEFINITIONS_FILEPATH
)
from vnpy.factor.utils.factor_utils import load_factor_setting, init_factors


# Schema for the intermediate flat DataFrame
_OHLCV_FLAT_SCHEMA = {
    "datetime": pl.Datetime(time_unit="us", time_zone='UTC'),
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def generate_fake_ohlcv_wide_dict(
    start_date: datetime,
    end_date: datetime,
    interval: str,
    vt_symbols: list[str]
) -> dict[str, pl.DataFrame]:
    """
    Generates a dictionary of Polars DataFrames with more realistic fake OHLCV data.

    This improved version models key financial data characteristics:
    1.  **Geometric Brownian Motion:** Prices follow a random walk with drift, creating trends.
    2.  **Cross-Asset Correlation:** Symbol returns are correlated using a covariance matrix.
    3.  **Realistic OHLC Bars:** Open is the previous close, and High/Low are structured around the O-C range.
    4.  **Price-Volume Correlation:** Volume is higher during periods of larger price changes.
    5.  **Vectorized Generation:** Uses NumPy for high-performance data generation, avoiding slow loops.

    Args:
        start_date: The start datetime for the data generation.
        end_date: The end datetime for the data generation.
        interval: Polars interval string for fake data generation (e.g., "1m", "1h").
        vt_symbols: List of symbol strings (e.g., ["BTCUSDT", "ETHUSDT"]).

    Returns:
        A dictionary where keys are OHLCV types and values are "wide" DataFrames.
    """
    if not vt_symbols:
        return {}

    # --- 1. Setup Time Range and Parameters ---
    try:
        datetimes = pl.datetime_range(
            start=start_date, end=end_date, interval=interval, time_unit="us", eager=True, time_zone="UTC"
        )
    except Exception as e:
        logger.error(f"Failed to generate date range: {e}")
        return {}

    if len(datetimes) < 2:
        logger.warning("Date range is too short to generate meaningful OHLCV data.")
        return {}

    n_dates = len(datetimes)
    n_symbols = len(vt_symbols)

    # --- 2. Generate Correlated Price Returns (The Core Engine) ---
    # Define base prices and parameters for each symbol
    base_prices = 100 + np.arange(n_symbols) * 50
    # Annualized drift (trend) and volatility for each symbol
    drifts = np.random.uniform(0.05, 0.25, n_symbols)  # 5% to 25% annual drift
    volatilities = np.random.uniform(0.3, 0.8, n_symbols) # 30% to 80% annual volatility

    # Create a realistic-looking correlation matrix
    corr_matrix = np.full((n_symbols, n_symbols), 0.7) # Base correlation of 0.7
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix += np.random.uniform(-0.15, 0.15, (n_symbols, n_symbols)) # Add some noise
    corr_matrix = (corr_matrix + corr_matrix.T) / 2 # Ensure symmetry
    np.fill_diagonal(corr_matrix, 1.0) # Ensure diagonal is 1

    # Create covariance matrix from volatilities and correlation
    std_devs = np.diag(volatilities)
    cov_matrix = std_devs @ corr_matrix @ std_devs

    # Generate correlated random shocks for all timesteps and symbols at once
    # dt = 1 / (365 * 24 * 60) # Time step fraction of a year (for 1m interval)
    # A simpler approach for fake data is to just scale by number of steps
    dt = 1 / n_dates
    mean_returns = (drifts - 0.5 * volatilities**2) * dt
    random_shocks = np.random.multivariate_normal(np.zeros(n_symbols), cov_matrix, n_dates)

    # Calculate log returns and then the price paths
    log_returns = mean_returns + random_shocks * np.sqrt(dt)
    price_paths = base_prices * np.exp(np.cumsum(log_returns, axis=0))

    # --- 3. Construct Realistic OHLCV Bars ---
    close_prices = price_paths
    # Open is the previous bar's close. First open is based on the first close.
    open_prices = np.vstack([close_prices[0] * (1 + (np.random.rand(n_symbols) - 0.5) * 0.01), close_prices[:-1]])

    # High and Low are structured around the open-close range
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 0.01, (n_dates, n_symbols)) * close_prices
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 0.01, (n_dates, n_symbols)) * close_prices
    low_prices = np.maximum(low_prices, 0.001) # Ensure price doesn't go to or below zero

    # --- 4. Generate Price-Correlated Volume ---
    price_change = np.abs(close_prices - open_prices) / open_prices
    base_volume = np.random.uniform(500, 1000, (n_dates, n_symbols))
    volume = base_volume * (1 + price_change * np.random.uniform(5, 15, n_symbols)) # Amplify volume on big moves

    # --- 5. Assemble into a "Long" DataFrame (Efficient) ---
    all_data_frames = []
    for i, symbol in enumerate(vt_symbols):
        symbol_df = pl.DataFrame({
            "datetime": datetimes,
            "symbol": symbol,
            "open": open_prices[:, i],
            "high": high_prices[:, i],
            "low": low_prices[:, i],
            "close": close_prices[:, i],
            "volume": volume[:, i],
        })
        all_data_frames.append(symbol_df)

    if not all_data_frames:
        return {}

    flat_df = pl.concat(all_data_frames)

    # --- 6. Pivot to Final "Wide" Dictionary Format ---
    ohlcv_dict: dict[str, pl.DataFrame] = {}
    ohlcv_types = ["open", "high", "low", "close", "volume"]

    for ohlcv_type in ohlcv_types:
        try:
            # Use the much faster pivot method on the complete long-form data
            pivoted_df = flat_df.pivot(
                values=ohlcv_type,
                index="datetime",
                columns="symbol"
            ).sort("datetime")
            ohlcv_dict[ohlcv_type] = pivoted_df
        except Exception as e:
            logger.error(f"Error pivoting data for {ohlcv_type}: {e}")
            # Fallback for safety, though unlikely with this new method
            ohlcv_dict[ohlcv_type] = pl.DataFrame({"datetime": datetimes})

    return ohlcv_dict


def main():
    """Main function to run the backtest"""
    backtest_engine = BacktestEngine(
        factor_module_name="vnpy.factor.factors",
        output_data_dir_for_analyser_reports=get_backtest_report_path(),
        output_data_dir_for_calculator_cache=get_backtest_data_cache_path()
    )

    # Define symbols
    vt_symbols = ['btcusdt.BINANCE', 'ethusdt.BINANCE', 'xrpusdt.BINANCE', 'ltcusdt.BINANCE',
                  'bchusdt.BINANCE', 'adausdt.BINANCE', 'solusdt.BINANCE', 'dogeusdt.BINANCE']

    # Load factor definitions
    factor_definitions = load_factor_setting(FACTOR_DEFINITIONS_FILEPATH)
    macd_factor_definition = factor_definitions[2]  # Using the third factor definition

    # Initialize MACD factor
    import importlib
    factor_module = importlib.import_module("vnpy.factor.factors")
    macd_factor = init_factors(
        module_for_primary_classes=factor_module,
        settings_data=[macd_factor_definition],
        dependencies_module_lookup_for_instances=factor_module
    )[0]

    # Initialize factor and calculator
    target_factor_instance, flattened_factors = backtest_engine._init_and_flatten_factor(
        macd_factor, vt_symbols
    )
    calculator = backtest_engine._create_calculator()

    # Generate fake data
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=400)
    time_interval = "1h"

    ohlcv_data_dictionary = generate_fake_ohlcv_wide_dict(
        start_date=start_dt,
        end_date=end_dt,
        interval=time_interval,
        vt_symbols=vt_symbols
    )

    # Set data in backtest engine
    backtest_engine.memory_bar = ohlcv_data_dictionary
    backtest_engine.num_data_rows = backtest_engine.memory_bar["close"].height

    # Run factor computation
    factor_df = backtest_engine._run_factor_computation(
        calculator=calculator,
        target_factor_instance=target_factor_instance,
        flattened_factors=flattened_factors,
        vt_symbols_for_run=vt_symbols,
    )

    # Clean up calculator
    calculator.close()

    # Prepare for analysis
    market_close_prices_df = backtest_engine.memory_bar["close"].clone()
    actual_analysis_start_dt = factor_df.select(pl.col('datetime').min()).item()
    actual_analysis_end_dt = factor_df.select(pl.col('datetime').max()).item()

    # Run factor analysis
    report_path = backtest_engine._run_factor_analysis(
        factor_df=factor_df,
        market_close_prices_df=market_close_prices_df,
        target_factor_instance=target_factor_instance,
        analysis_start_dt=actual_analysis_start_dt,
        analysis_end_dt=actual_analysis_end_dt,
        num_quantiles=5,
        long_short_percentile=0.5,
        report_filename_prefix='test',
    )
    print(report_path)


if __name__ == "__main__":
    main()
