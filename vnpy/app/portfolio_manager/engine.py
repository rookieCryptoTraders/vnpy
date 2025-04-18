import logging
from typing import Any, Dict, Set, Optional, Tuple
from datetime import datetime

from vnpy.event import Event
from vnpy.trader.engine import (
    MainEngine,
    EventEngine,
    BaseEngine
)
from vnpy.trader.event import (
    EVENT_ORDER,
    EVENT_CONTRACT,
    EVENT_TIMER,
    EVENT_TRADE
)
from vnpy.trader.object import (
    ContractData,
    OrderData,
    TradeData,
    SubscribeRequest
)
from vnpy.trader.utility import load_json, save_json

from .base import ContractResult, PortfolioResult, PortfolioStatistics


APP_NAME = "PortfolioManager"

EVENT_PM_CONTRACT = "ePmContract"
EVENT_PM_PORTFOLIO = "ePmPortfolio"
EVENT_PM_TRADE = "ePmTrade"


class PortfolioEngine(BaseEngine):
    """"""
    setting_filename: str = "portfolio_manager_setting.json"
    data_filename: str = "portfolio_manager_data.json"
    order_filename: str = "portfolio_manager_order.json"

    # Fix incorrect type annotation - key should be Tuple[str, str] not set
    contract_results: Dict[Tuple[str, str], ContractResult]

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """Initialize portfolio engine with main engine and event engine."""
        super().__init__(main_engine, event_engine, APP_NAME)

        # Remove logger initialization
        self.get_tick = self.main_engine.get_tick
        self.get_contract = self.main_engine.get_contract

        self.subscribed: Set[str] = set()
        self.result_symbols: Set[str] = set()
        self.order_reference_map: Dict[str, str] = {}
        self.contract_results: Dict[Tuple[str, str], ContractResult] = {}
        self.portfolio_results: Dict[str, PortfolioResult] = {}
        self.portfolio_statistics: Dict[str, PortfolioStatistics] = {}

        self.timer_count: int = 0
        self.timer_interval: int = 5

        self.load_setting()
        self.load_data()
        self.load_order()
        self.register_event()

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data

        if order.vt_orderid not in self.order_reference_map:
            self.order_reference_map[order.vt_orderid] = order.reference
        else:
            order.reference = self.order_reference_map[order.vt_orderid]

    def process_trade_event(self, event: Event) -> None:
        """Process new trade data and update portfolio accordingly."""
        trade: TradeData = event.data

        reference: str = self.order_reference_map.get(trade.vt_orderid, "")
        if not reference:
            self.write_log(f"Trade {trade.vt_tradeid} has no reference", logging.DEBUG)
            return

        # Fix key type from set to tuple
        key: Tuple[str, str] = (reference, trade.vt_symbol)

        contract_result: Optional[ContractResult] = self.contract_results.get(key, None)
        if not contract_result:
            contract_result: ContractResult = ContractResult(self, reference, trade.vt_symbol)
            self.contract_results[key] = contract_result

        contract_result.update_trade(trade)

        # 添加成交数据
        trade.reference = reference
        self.event_engine.put(Event(EVENT_PM_TRADE, trade))

        # 自动订阅tick数据
        if trade.vt_symbol in self.subscribed:
            return

        contract: Optional[ContractData] = self.main_engine.get_contract(trade.vt_symbol)
        if not contract:
            return

        req: SubscribeRequest = SubscribeRequest(contract.symbol, contract.exchange)
        self.main_engine.subscribe(req, contract.gateway_name)

    def process_timer_event(self, event: Event) -> None:
        """Process timer event and update PnL calculations."""
        # Add validation
        if not event:
            self.write_log("Received empty timer event", logging.ERROR)
            return

        self.timer_count += 1
        if self.timer_count < self.timer_interval:
            return
        self.timer_count = 0
        datetime_str: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for portfolio_result in self.portfolio_results.values():
            portfolio_result.clear_pnl()

        for contract_result in self.contract_results.values():
            contract_result.calculate_pnl()

            portfolio_result: PortfolioResult = self.get_portfolio_result(contract_result.reference)
            portfolio_result.trading_pnl += contract_result.trading_pnl
            portfolio_result.holding_pnl += contract_result.holding_pnl
            portfolio_result.total_pnl += contract_result.total_pnl
            portfolio_result.value += contract_result.value

            event: Event = Event(EVENT_PM_CONTRACT, contract_result.get_data())
            self.event_engine.put(event)

        for reference, portfolio_result in self.portfolio_results.items():
            event: Event = Event(EVENT_PM_PORTFOLIO, portfolio_result.get_data())
            portfolio_statistics: PortfolioStatistics = self.portfolio_statistics.get(reference, None)
            if not portfolio_statistics:
                portfolio_statistics = PortfolioStatistics(reference)
                self.portfolio_statistics[reference] = portfolio_statistics
            portfolio_statistics.record_value(datetime_str, portfolio_result.get_total_value())
            self.event_engine.put(event)

    def process_contract_event(self, event: Event) -> None:
        """"""
        contract: ContractData = event.data
        if contract.vt_symbol not in self.result_symbols:
            return

        req: SubscribeRequest = SubscribeRequest(contract.symbol, contract.exchange)
        self.main_engine.subscribe(req, contract.gateway_name)

        self.subscribed.add(contract.vt_symbol)

    def load_data(self) -> None:
        """"""
        data: Optional[dict] = load_json(self.data_filename)
        if not data:
            return

        today: str = datetime.now().strftime("%Y-%m-%d")
        date_changed: bool = False

        date: str = data.pop("date")
        for key, d in data.items():
            reference, vt_symbol = key.split(",")

            if date == today:
                pos: float = d["open_pos"]  # If same day, use open position
            else:
                pos: float = d["last_pos"]  # If new day, last position becomes open position
                date_changed: bool = True

            self.result_symbols.add(vt_symbol)
            self.contract_results[(reference, vt_symbol)] = ContractResult(
                self,
                reference,
                vt_symbol,
                pos
            )

        # 当数据改变时重新保存
        if date_changed:
            self.save_data()

    def save_data(self) -> None:
        """"""
        data: Dict[str, Any] = {"date": datetime.now().strftime("%Y-%m-%d")}

        for contract_result in self.contract_results.values():
            key: str = f"{contract_result.reference},{contract_result.vt_symbol}"
            data[key] = {
                "open_pos": contract_result.open_pos,
                "last_pos": contract_result.last_pos
            }

        save_json(self.data_filename, data)

    def load_setting(self) -> None:
        """"""
        setting: Optional[dict] = load_json(self.setting_filename)
        if "timer_interval" in setting:
            self.timer_interval = setting["timer_interval"]

    def save_setting(self) -> None:
        """"""
        setting: Dict[str, int] = {"timer_interval": self.timer_interval}
        save_json(self.setting_filename, setting)

    def load_order(self) -> None:
        """"""
        order_data: Optional[dict] = load_json(self.order_filename)

        date: str = order_data.get("date", "")
        today: str = datetime.now().strftime("%Y-%m-%d")
        if date == today:
            self.order_reference_map = order_data["data"]

    def save_order(self) -> None:
        """"""
        order_data: Dict[str, Any] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "data": self.order_reference_map
        }
        save_json(self.order_filename, order_data)

    def close(self) -> None:
        """"""
        self.save_setting()
        self.save_data()
        self.save_order()

    def get_portfolio_result(self, reference: str) -> PortfolioResult:
        """"""
        portfolio_result: Optional[PortfolioResult] = self.portfolio_results.get(reference, None)
        if not portfolio_result:
            portfolio_result = PortfolioResult(reference)
            self.portfolio_results[reference] = portfolio_result
        return portfolio_result

    def set_timer_interval(self, interval: int) -> None:
        """"""
        self.timer_interval = interval

    def get_timer_interval(self) -> int:
        """"""
        return self.timer_interval
    
    def write_log(self, msg: str, level: int = logging.INFO) -> None:
        """Write log message with appropriate level."""
        self.main_engine.write_log(msg, APP_NAME, level)
