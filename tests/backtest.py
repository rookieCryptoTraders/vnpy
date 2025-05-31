from vnpy.factor.backtesting.backtesting import BacktestEngine
from vnpy.factor.setting import get_backtest_data_cache_path, get_backtest_report_path


backtest_engine = BacktestEngine(
    factor_module_name="vnpy.factor.factors",
    output_data_dir_for_analyser_reports=get_backtest_report_path(),
    output_data_dir_for_calculator_cache=get_backtest_data_cache_path()
)