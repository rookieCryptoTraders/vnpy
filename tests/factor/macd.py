from vnpy.factor.base import FactorMode
from vnpy.factor.factors import MACDFactor
from vnpy.factor.factors import EMAFactor
from vnpy.trader.constant import Interval

macd = MACDFactor(
    setting={"params": {'signal_period': 9}}
)

fast_ema = EMAFactor(
    setting={"factor_name": "fast_ema","params": {'period': 12}}
)

slow_ema = EMAFactor(
    setting={"factor_name":"slow_ema","params": {'period': 26}}
)

macd.dependencies_factor = [fast_ema, slow_ema]
macd.freq = Interval.MINUTE
macd.factor_mode = FactorMode.BACKTEST

s = macd.to_setting()

print(s)