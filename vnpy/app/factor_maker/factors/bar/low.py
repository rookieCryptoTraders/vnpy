from vnpy.app.factor_maker.factors.bar.bar_base import BarFactor
from vnpy.trader.object import TickData, BarData, Exchange, Interval, FactorData


class LOW(BarFactor):
    factor_name = "low"

    def on_bar(self, bar: BarData) -> FactorData:
        value = bar.low_price
        return FactorData(gateway_name="FactorTemplate", symbol=bar.symbol, exchange=bar.exchange,
                          datetime=bar.datetime,
                          value=value, factor_name=self.factor_name, interval=bar.interval)

    def on_tick(self, tick: TickData) -> None:
        pass

    def on_factor(self, factor: FactorData) -> FactorData:
        pass
