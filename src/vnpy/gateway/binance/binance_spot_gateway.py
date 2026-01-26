import json
import logging
import time
import traceback
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from threading import Lock
from typing import Any, Literal

from binance.spot import Spot
from binance.websocket.websocket_client import BinanceWebsocketClient

from vnpy.event import Event, EventEngine
from vnpy.trader.constant import (
    Direction,
    Exchange,
    Interval,
    OrderType,
    Product,
    Status,
)
from vnpy.trader.event import EVENT_TIMER
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    AccountData,
    BarData,
    CancelRequest,
    ContractData,
    HistoryRequest,
    OrderData,
    OrderRequest,
    SubscribeRequest,
    TickData,
    TradeData,
)
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import round_to, round_volume

SYSTEM_MODE = SETTINGS.get("system.mode", "LIVE")

# 实盘REST API地址
REST_HOST: str = "https://api.binance.com"

# 实盘Websocket API地址
WEBSOCKET_TRADE_HOST: str = "wss://stream.binance.com:9443"
WEBSOCKET_DATA_HOST: str = "wss://stream.binance.com:9443"

# 模拟盘REST API地址
TESTNET_REST_HOST: str = "https://testnet.binance.vision"

# 模拟盘Websocket API地址
TESTNET_WEBSOCKET_TRADE_HOST: str = "wss://testnet.binance.vision"
TESTNET_WEBSOCKET_DATA_HOST: str = "wss://testnet.binance.vision"

# 委托状态映射
STATUS_BINANCE2VT: dict[str, Status] = {
    "NEW": Status.NOTTRADED,
    "PARTIALLY_FILLED": Status.PARTTRADED,
    "FILLED": Status.ALLTRADED,
    "CANCELED": Status.CANCELLED,
    "REJECTED": Status.REJECTED,
    "EXPIRED": Status.CANCELLED,
}

# 委托类型映射
ORDERTYPE_VT2BINANCE: dict[OrderType, str] = {
    OrderType.LIMIT: "LIMIT",
    OrderType.MARKET: "MARKET",
}
ORDERTYPE_BINANCE2VT: dict[str, OrderType] = {
    v: k for k, v in ORDERTYPE_VT2BINANCE.items()
}

# 买卖方向映射
DIRECTION_VT2BINANCE: dict[Direction, str] = {
    Direction.LONG: "BUY",
    Direction.SHORT: "SELL",
}
DIRECTION_BINANCE2VT: dict[str, Direction] = {
    v: k for k, v in DIRECTION_VT2BINANCE.items()
}

# 数据频率映射
INTERVAL_VT2BINANCE: dict[Interval, str] = {
    Interval.MINUTE: "1m",
    Interval.HOUR: "1h",
    Interval.DAILY: "1d",
}

# 时间间隔映射
TIMEDELTA_MAP: dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}

# 合约数据全局缓存字典
symbol_contract_map: dict[str, ContractData] = {}

# proxies - default to port 1082 for VPN
proxies: dict[str, str] = SETTINGS.get(
    "gateway.proxies",
    {"http": "http://127.0.0.1:1082", "https": "http://127.0.0.1:1082"},
)

# ----- Sharding Configuration for High-Performance Kline Monitoring -----
DEFAULT_SHARD_COUNT = 4  # Number of WebSocket connections for market data
BATCH_SUBSCRIBE_SIZE = 25  # Symbols to subscribe in one message

# Try to use orjson for faster JSON parsing
try:
    import orjson

    def fast_json_loads(s):
        return orjson.loads(s)

    JSON_LIBRARY = "orjson"
except ImportError:

    def fast_json_loads(s):
        return json.loads(s)

    JSON_LIBRARY = "json"


@dataclass
class ShardStats:
    """Statistics for a single shard (WebSocket connection)"""

    shard_id: int
    symbols: list[str] = field(default_factory=list)
    message_count: int = 0
    bar_count: int = 0
    last_message_time: datetime | None = None
    is_connected: bool = False


class BinanceSpotGateway(BaseGateway):
    """
    vn.py用于对接币安现货账户的交易接口。

    Optimized for high-performance kline monitoring of 100+ symbols.
    Uses sharded WebSocket connections internally.
    """

    default_name: str = "BINANCE_SPOT"

    default_setting: dict[str, Any] = {
        "key": "",
        "secret": "",
        "server": ["REAL", "TESTNET"],
    }

    exchanges: list[Exchange] = [Exchange.BINANCE]

    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        """构造函数"""
        super().__init__(event_engine, gateway_name)
        # 订阅交易数据 比如order变化，仓位变化
        self.trade_ws_api: BinanceSpotTradeWebsocketApi = BinanceSpotTradeWebsocketApi(
            self
        )

        # 订阅市场数据 比如Kline， ticker， depth (now with internal sharding)
        self.market_ws_api: BinanceSpotDataWebsocketApi = BinanceSpotDataWebsocketApi(
            self
        )

        # 与binance交互， 比如下单，撤单
        self.rest_api: BinanceSpotRestAPi = BinanceSpotRestAPi(self)

        self.orders: dict[str, OrderData] = {}

        # update exchanges by settings
        self.exchanges = [Exchange(e) for e in SETTINGS.get("gateway.exchanges", [])]

    def connect(self, setting: dict):
        """连接交易接口"""
        key: str = setting["gateway.api_key"]
        secret: str = setting["gateway.api_secret"]
        server: str = setting["gateway.server"]

        self.rest_api.connect(key, secret, server)
        self.market_ws_api.connect(server)

        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def subscribe(self, req: SubscribeRequest) -> None:
        """订阅行情 (internally uses sharded connections for klines)"""
        self.market_ws_api.subscribe(req)

    def send_order(self, req: OrderRequest) -> str:
        """委托下单"""
        return self.rest_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        """委托撤单"""
        self.rest_api.cancel_order(req)

    def query_account(self) -> None:
        """查询资金"""
        pass

    def query_position(self) -> None:
        """查询持仓"""
        pass

    def query_history(self, req: HistoryRequest) -> list[BarData] | list[dict]:
        """查询历史数据"""
        return self.rest_api.query_history(req)

    def close(self) -> None:
        """关闭连接"""
        self.rest_api.stop()
        self.trade_ws_api.stop()
        self.market_ws_api.stop()

    def process_timer_event(self, event: Event) -> None:
        """定时事件处理"""
        self.rest_api.keep_user_stream()

    def on_order(self, order: OrderData) -> None:
        """推送委托数据"""
        self.orders[order.orderid] = copy(order)
        super().on_order(order)

    def get_order(self, orderid: str) -> OrderData:
        """查询委托数据"""
        return self.orders.get(orderid, None)


class BinanceSpotRestAPi:
    """币安现货REST API"""

    def __init__(self, gateway: BinanceSpotGateway) -> None:
        """构造函数"""
        super().__init__()

        self.gateway: BinanceSpotGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.trade_ws_api: BinanceSpotTradeWebsocketApi = self.gateway.trade_ws_api

        self.key: str = ""
        self.secret: str = ""

        self.user_stream_key: str = ""
        self.keep_alive_count: int = 0
        self.recv_window: int = 5000
        self.time_offset: int = 0

        self.order_count: int = 1_000_000
        self.order_count_lock: Lock = Lock()
        self.connect_time: int = 0

        self._active: bool = False

    def connect(self, key: str, secret: str, server: str) -> None:
        """连接REST服务器"""
        self.key = key
        self.secret = secret
        self.server = server

        self._client = Spot(api_key=self.key, api_secret=self.secret, proxies=proxies)

        self.connect_time = self._client.time()["serverTime"]

        self.gateway.write_log("REST API启动成功")
        self._active = True

        self.query_time()
        self.query_account()
        self.query_order()
        self.query_contract()
        self.start_user_stream()

    def query_time(self) -> None:
        """查询时间"""
        data = self._client.time()
        self.on_query_time(data)

    def query_account(self) -> None:
        """查询资金"""
        self.on_query_account(self._client.account())

    def query_order(self) -> None:
        """查询未成交委托"""
        self.on_query_order(self._client.get_open_orders())

    def query_contract(self) -> None:
        """查询合约信息"""
        self.on_query_contract(self._client.exchange_info())

    @lru_cache(maxsize=128)
    def get_cached_contract(self, symbol: str) -> ContractData | None:
        """从缓存中获取合约信息"""
        return symbol_contract_map.get(symbol)

    def _new_order_id(self) -> int:
        """生成本地委托号"""
        with self.order_count_lock:
            self.order_count += 1
            return self.order_count

    def send_order(self, req: OrderRequest) -> str:
        """委托下单"""
        # 生成本地委托号
        orderid: str = str(self.connect_time + self._new_order_id())

        # 推送提交中事件
        order: OrderData = req.create_order_data(orderid, self.gateway_name)
        self.gateway.on_order(order)

        contract: ContractData | None = self.get_cached_contract(req.symbol)
        if contract:
            req.volume = round_volume(
                float(req.volume), contract.min_volume, self.commission_rate
            )
        params: dict = {
            "symbol": req.symbol.upper(),
            "side": DIRECTION_VT2BINANCE[req.direction],
            "type": ORDERTYPE_VT2BINANCE[req.type],
            "quantity": format(req.volume, "f"),
            "newClientOrderId": order.orderid,
            "newOrderRespType": "FULL",
        }

        if req.type == OrderType.LIMIT:
            params["timeInForce"] = "GTC"
            params["price"] = str(req.price)
        elif req.type == OrderType.STOP:
            params["type"] = "STOP_MARKET"
            params["stopPrice"] = float(req.price)

        try:
            data = self._client.new_order(**params)
            print(data)
            self.on_send_order(data, order)
        except Exception as e:
            self.on_send_order_error(e, order)
            raise e

        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        """委托撤单"""

        params: dict = {"symbol": req.symbol.upper(), "origClientOrderId": req.orderid}
        print(params)

        order: OrderData = self.gateway.get_order(req.orderid)

        data = self._client.cancel_order(**params)
        self.on_cancel_order(data, order)

    def start_user_stream(self) -> None:
        """生成listenKey"""
        data = self._client.new_listen_key()
        self.on_start_user_stream(data)

    def keep_user_stream(self) -> None:
        """延长listenKey有效期"""
        self.keep_alive_count += 1
        if self.keep_alive_count < 600:
            return
        self.keep_alive_count = 0

        try:
            data = self._client.renew_listen_key(listenKey=self.user_stream_key)
            self.on_keep_user_stream(data)
        except Exception as e:
            self.on_keep_user_stream_error(e)

    def on_query_time(self, data: dict) -> None:
        """时间查询回报"""
        local_time = int(time.time() * 1000)
        server_time = int(data["serverTime"])
        self.time_offset = local_time - server_time

    def on_query_account(self, data: dict) -> None:
        """资金查询回报"""
        self.commission_rate = float(data["commissionRates"].get("taker", 0.0015))
        for account_data in data["balances"]:
            account: AccountData = AccountData(
                accountid=account_data["asset"],
                balance=float(account_data["free"]) + float(account_data["locked"]),
                frozen=float(account_data["locked"]),
                gateway_name=self.gateway_name,
            )

            if account.balance:
                self.gateway.on_account(account)

        self.gateway.write_log("账户资金查询成功")

    def on_query_order(self, data: dict) -> None:
        """未成交委托查询回报"""
        for d in data:
            # 过滤不支持类型的委托
            if d["type"] not in ORDERTYPE_BINANCE2VT:
                continue

            order: OrderData = OrderData(
                orderid=d["clientOrderId"],
                symbol=d["symbol"].lower(),
                exchange=Exchange.BINANCE,
                price=float(d["price"]),
                volume=float(d["origQty"]),
                type=ORDERTYPE_BINANCE2VT[d["type"]],
                direction=DIRECTION_BINANCE2VT[d["side"]],
                traded=float(d["executedQty"]),
                status=STATUS_BINANCE2VT.get(d["status"], None),
                datetime=datetime.fromtimestamp(int(d["time"]) / 1000),
                gateway_name=self.gateway_name,
            )
            self.gateway.on_order(order)

        self.gateway.write_log("委托信息查询成功")

    def on_query_contract(self, data: dict) -> None:
        """合约信息查询回报"""
        for d in data["symbols"]:
            base_currency: str = d["baseAsset"]
            quote_currency: str = d["quoteAsset"]
            name: str = f"{base_currency.upper()}/{quote_currency.upper()}"

            pricetick: float = 1
            min_volume: float = 1

            for f in d["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    pricetick = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    min_volume = float(f["stepSize"])

            contract: ContractData = ContractData(
                symbol=d["symbol"].lower(),
                exchange=Exchange.BINANCE,
                name=name,
                pricetick=pricetick,
                size=1,
                min_volume=min_volume,
                product=Product.SPOT,
                history_data=True,
                gateway_name=self.gateway_name,
                stop_supported=True,
            )
            self.gateway.on_contract(contract)

            symbol_contract_map[contract.symbol] = contract

        self.gateway.write_log("合约信息查询成功")

    def on_send_order(self, data: dict, order: OrderData) -> None:
        """

        Parameters
        ----------
        data :
            binance response
        order :

        Returns
        -------

        """
        """委托下单回报"""
        if data["status"] not in STATUS_BINANCE2VT:
            self.on_send_order_failed(data, order)
        order.status = STATUS_BINANCE2VT[data["status"]]
        self.gateway.on_order(order)

    def on_send_order_failed(self, data: dict, order: OrderData) -> None:
        """委托下单失败服务器报错回报"""
        order.status = Status.REJECTED
        self.gateway.on_order(order)

        self.gateway.write_log(
            f"{order.vt_orderid}委托失败，状态码：{data['code']}, 信息：{data['msg']}"
        )

    def on_send_order_error(self, exception, order: OrderData) -> None:
        """委托下单回报函数报错回报"""
        order.status = Status.REJECTED
        self.gateway.on_order(order)

        self.gateway.write_log(f"{order.vt_orderid}委托失败：{exception}")

    def on_cancel_order(self, data: dict, order: OrderData) -> None:
        """委托撤单回报"""
        print("on_cancel_order", data)
        if data.get("code", None):
            self.on_cancel_failed(data, order)

    def on_cancel_failed(self, data: dict, order: OrderData) -> None:
        """撤单回报函数报错回报"""
        if order:
            order.status = Status.REJECTED
            self.gateway.on_order(order)

        msg = f"撤单失败，状态码：{data['code']}，信息：{data['msg']}"
        self.gateway.write_log(msg)

    def on_start_user_stream(self, data: dict) -> None:
        """生成listenKey回报"""
        self.user_stream_key = data["listenKey"]
        self.keep_alive_count = 0

        if self.server == "REAL":
            url = WEBSOCKET_TRADE_HOST
        else:
            url = TESTNET_WEBSOCKET_TRADE_HOST

        self.trade_ws_api.connect(url, self.user_stream_key)

    def on_keep_user_stream(self, data) -> None:
        """延长listenKey有效期回报"""
        pass

    def on_keep_user_stream_error(self, exception) -> None:
        """延长listenKey有效期函数报错回报"""
        # 当延长listenKey有效期时，忽略超时报错
        self.gateway.write_log(f"延长listenKey有效期失败：{exception}; reconnecting...")
        try:
            self.start_user_stream()
        except Exception as e:
            self.gateway.write_log(f"重连失败：{e}")

    def query_history(
        self,
        req: HistoryRequest,
        ret: Literal["list_dict", "list_bar_data"] = "list_dict",
    ) -> list[BarData] | list[dict]:
        """查询历史数据

        Args:
            req ():
            ret (Literal): return what type, 'list_dict' or 'list_bar_data'

        Returns:
            list[BarData] | list[dict]
        """
        history: list[BarData] | list[dict] = []
        limit: int = 1000
        start_time: int = int(datetime.timestamp(req.start))

        sleep_seconds = 0.5
        while True:
            # 创建查询参数
            params: dict = {
                "symbol": req.symbol.upper(),
                "interval": INTERVAL_VT2BINANCE[req.interval],
                "limit": limit,
                "startTime": start_time * 1000,  # 转换成毫秒
            }

            if req.end:
                end_time: int = int(datetime.timestamp(req.end))
                interval_seconds = int(
                    TIMEDELTA_MAP[req.interval].total_seconds()
                )  # 周期的秒数
                if (
                    end_time // interval_seconds
                    == datetime.now().timestamp() // interval_seconds
                ):  # 结束于现在，则不要当下未完成的柱子
                    end_time -= interval_seconds

                params["endTime"] = end_time * 1000

            try:
                data = self._client.klines(**params)
                if isinstance(data, str):
                    data = json.loads(data)
                if isinstance(data, dict) and data["code"]:
                    if data["code"] == 429:
                        self.gateway.write_log(
                            f"获取历史数据失败：error code {data['code']}, {data['msg']}"
                        )
                        self.gateway.write_log(
                            f"{sleep_seconds=} for retring connection"
                        )
                        time.sleep(sleep_seconds)
                        sleep_seconds *= 2
                        continue
                    else:
                        sleep_seconds = 0.5

                    if data["code"] // 100 != 2:
                        self.gateway.write_log(
                            f"获取历史数据失败：error code {data['code']}, {data['msg']}"
                        )
                        break
                elif isinstance(data, list):
                    if ret == "list_dict":
                        for row in data:
                            bar = {
                                "datetime": datetime.fromtimestamp(
                                    row[0] / 1000
                                ),  # Convert ms to seconds
                                "open": float(row[1]),
                                "high": float(row[2]),
                                "low": float(row[3]),
                                "close": float(row[4]),
                                "volume": float(row[5]),
                                # "close_time": datetime.datetime.fromtimestamp(row[6] / 1000),  # Convert ms to seconds
                                "quote_asset_volume": float(row[7]),
                                "number_of_trades": int(row[8]),
                                "taker_buy_base_asset_volume": float(row[9]),
                                "taker_buy_quote_asset_volume": float(row[10]),
                            }
                            history.append(bar)

                    elif ret == "list_bar_data":  # list_bar_data
                        for row in data:
                            bar: BarData = BarData(
                                symbol=req.symbol,
                                exchange=req.exchange,
                                datetime=datetime.fromtimestamp(row[0] / 1000),
                                interval=req.interval,
                                open_price=float(row[1]),
                                high_price=float(row[2]),
                                low_price=float(row[3]),
                                close_price=float(row[4]),
                                volume=float(row[5]),
                                quote_asset_volume=float(row[7]),
                                number_of_trades=int(row[8]),
                                taker_buy_base_asset_volume=float(row[9]),
                                taker_buy_quote_asset_volume=float(row[10]),
                                gateway_name=self.gateway_name,
                            )
                            history.append(bar)

                    if ret == "list_dict":
                        begin: datetime = history[0]["datetime"]
                        end: datetime = history[-1]["datetime"]
                    elif ret == "list_bar_data":
                        begin: datetime = history[0].datetime
                        end: datetime = history[-1].datetime
                    else:
                        raise RuntimeError("unknown return type")
                    self.gateway.write_log(
                        f"获取历史数据成功，{req.symbol} - {req.interval.value}, {begin} - {end}"
                    )

                if len(data) < limit:
                    break

                else:
                    raise RuntimeError("unknown data format")
            except Exception as e:
                self.gateway.write_log(data, level=logging.ERROR)
                self.gateway.write_log(
                    f"{traceback.format_tb(e.__traceback__)}", level=logging.ERROR
                )
                raise e

        return history

    def stop(self) -> None:
        """停止"""
        self._active = False
        self._client.close_listen_key(listenKey=self.user_stream_key)


class BinanceSpotTradeWebsocketApi:
    """币安现货交易Websocket API"""

    def __init__(self, gateway: BinanceSpotGateway) -> None:
        """构造函数"""
        super().__init__()

        self.gateway: BinanceSpotGateway = gateway
        self.gateway_name = gateway.gateway_name
        self._active: bool = False
        self._client: SpotWebsocketStreamClient_vnpy | None = None  # 数据源

    def connect(self, stream_url: str, listen_key: str) -> None:
        """连接Websocket交易频道"""

        is_combined = False
        if self._client:
            url_with_mode = self._client.socket_manager.stream_url.split("?timeUnit=")[
                0
            ]
            if is_combined and url_with_mode == stream_url + "/stream":
                pass
            elif not is_combined and url_with_mode == stream_url + "/ws":
                pass
            else:
                self._client.logger.warning(
                    "BinanceSpotTradeWebsocketApi.connect: 重连不同模式的Websocket，先断开旧连接"
                )
                self._client.stop()

        self._client = SpotWebsocketStreamClient_vnpy(
            stream_url=stream_url,
            on_message=self.on_packet,
            on_close=self.on_disconnected,
            is_combined=False,
        )
        self._client.user_data(listen_key)

        self._active = True

        self.on_connected()

    def on_connected(self) -> None:
        """连接成功回报"""
        self.gateway.write_log("交易Websocket API连接成功")

    def on_packet(self, _, packet) -> None:
        """推送数据回报"""
        if isinstance(packet, str):
            packet = json.loads(packet)
        event_type = packet.get("e", None)
        if event_type == "outboundAccountPosition":
            self.on_account(packet)
        elif event_type == "executionReport":
            self.on_order(packet)
        elif event_type == "listenKeyExpired":
            self.on_listen_key_expired()

    def on_listen_key_expired(self) -> None:
        """ListenKey过期"""
        self.gateway.write_log("listenKey过期")
        self.disconnect()

    def disconnect(self) -> None:
        """ "主动断开webscoket链接"""
        self._active = False
        if self._client:
            self._client.stop()
            self.gateway.write_log(
                "BinanceSpotTradeWebsocketApi.disconnect: 交易Websocket API断开"
            )

    def on_account(self, packet: dict) -> None:
        """资金更新推送"""
        for d in packet["B"]:
            account: AccountData = AccountData(
                accountid=d["a"],
                balance=float(d["f"]) + float(d["l"]),
                frozen=float(d["l"]),
                gateway_name=self.gateway_name,
            )

            if account.balance:
                self.gateway.on_account(account)

    def on_order(self, packet: dict) -> None:
        """委托更新推送"""
        # 过滤不支持类型的委托
        if packet["o"] not in ORDERTYPE_BINANCE2VT:
            return

        if packet["C"] == "":
            orderid: str = packet["c"]
        else:
            orderid: str = packet["C"]

        offset = (
            self.gateway.get_order(orderid).offset
            if self.gateway.get_order(orderid)
            else None
        )
        reference = (
            self.gateway.get_order(orderid).reference
            if self.gateway.get_order(orderid)
            else None
        )

        order: OrderData = OrderData(
            symbol=packet["s"].lower(),
            exchange=Exchange.BINANCE,
            orderid=orderid,
            type=ORDERTYPE_BINANCE2VT[packet["o"]],
            direction=DIRECTION_BINANCE2VT[packet["S"]],
            price=float(packet["p"]),
            volume=float(packet["q"]),
            traded=float(packet["z"]),
            status=STATUS_BINANCE2VT[packet["X"]],
            datetime=datetime.fromtimestamp(packet["O"] / 1000),
            gateway_name=self.gateway_name,
            offset=offset,
            reference=reference,
        )

        self.gateway.on_order(order)

        # 将成交数量四舍五入到正确精度
        trade_volume = float(packet["l"])
        contract: ContractData = symbol_contract_map.get(order.symbol, None)
        if contract:
            trade_volume = round_to(trade_volume, contract.min_volume)

        if not trade_volume:
            return

        trade: TradeData = TradeData(
            symbol=order.symbol,
            exchange=order.exchange,
            orderid=order.orderid,
            tradeid=packet["t"],
            direction=order.direction,
            price=float(packet["L"]),
            volume=trade_volume,
            datetime=datetime.fromtimestamp(packet["T"] / 1000),
            gateway_name=self.gateway_name,
            offset=offset,
            reference=reference,
        )
        self.gateway.on_trade(trade)

    def on_disconnected(self, *args) -> None:
        """连接断开回报"""
        self.gateway.write_log(
            "BinanceSpotTradeWebsocketApi.on_disconnected:交易Websocket API断开"
        )
        self.gateway.rest_api.start_user_stream()

    def stop(self):
        self.disconnect()
        self._active = False


class KlineWebsocketShard:
    """
    A single WebSocket connection that handles a subset of symbols for kline streaming.
    Used by ShardedKlineManager for high-performance monitoring of 100+ symbols.
    """

    def __init__(
        self,
        shard_id: int,
        stream_url: str,
        gateway: "BinanceSpotGateway",
        logger: logging.Logger,
    ):
        self.shard_id = shard_id
        self.stream_url = stream_url
        self.gateway = gateway
        self.logger = logger

        self.symbols: list[str] = []
        self.bars: dict[str, BarData] = {}
        self.last_bar_times: dict[str, datetime] = {}  # For gap detection

        self._client: SpotWebsocketStreamClient_vnpy | None = None
        self._active: bool = False
        self._lock: Lock = Lock()

        # Statistics
        self.stats = ShardStats(shard_id=shard_id)

    def connect(self) -> None:
        """Establish WebSocket connection using SpotWebsocketStreamClient_vnpy"""
        self._client = SpotWebsocketStreamClient_vnpy(
            stream_url=self.stream_url,
            on_message=self._on_message,
            on_close=self._on_disconnected,
            on_error=self._on_error,
            is_combined=True,
            proxies=proxies,
        )

        self._active = True
        self.stats.is_connected = True
        self.logger.info(f"Shard {self.shard_id} connecting to {self.stream_url}")

    def add_symbol(self, symbol: str, interval: Interval = Interval.MINUTE) -> None:
        """Add a symbol to this shard"""
        with self._lock:
            if symbol in self.symbols:
                return
            self.symbols.append(symbol)
            self.stats.symbols = self.symbols.copy()

            # Create bar placeholder
            self.bars[symbol] = BarData(
                symbol=symbol,
                exchange=Exchange.BINANCE,
                datetime=datetime.fromtimestamp(0, tz=timezone.utc),
                gateway_name=self.gateway.gateway_name,
                interval=interval,
            )

    def subscribe_all(self, interval: str = "1m") -> None:
        """Subscribe to kline streams for all symbols in this shard"""
        if not self._client or not self.symbols:
            return

        # Use the client's kline method which handles batching
        self._client.kline(self.symbols, interval)
        self.logger.info(
            f"Shard {self.shard_id}: Subscribed to {len(self.symbols)} symbols"
        )

    def _on_message(self, _, message) -> None:
        """Handle incoming WebSocket message"""
        try:
            if isinstance(message, str):
                data = fast_json_loads(message)
            else:
                data = message

            self.stats.message_count += 1
            self.stats.last_message_time = datetime.now()

            # Handle subscription response
            if "result" in data:
                return

            # Handle combined stream format
            stream = data.get("stream", "")
            if not stream or "@kline_" not in stream:
                return

            kdata = data["data"]["k"]
            symbol = data["data"]["s"].lower()
            is_closed = kdata["x"]

            # Only process closed bars
            if not is_closed:
                return

            bar_time = datetime.fromtimestamp(kdata["t"] / 1000)

            # Gap detection
            if symbol in self.last_bar_times:
                expected_time = self.last_bar_times[symbol] + timedelta(minutes=1)
                if bar_time > expected_time:
                    gap_minutes = int((bar_time - expected_time).total_seconds() / 60)
                    self.logger.warning(
                        f"Shard {self.shard_id}: Gap detected for {symbol}! "
                        f"Missing {gap_minutes} bar(s)"
                    )

            self.last_bar_times[symbol] = bar_time

            # Build bar data
            bar = self.bars.get(symbol)
            if bar:
                bar.datetime = bar_time
                bar.open_price = float(kdata["o"])
                bar.high_price = float(kdata["h"])
                bar.low_price = float(kdata["l"])
                bar.close_price = float(kdata["c"])
                bar.volume = float(kdata["v"])
                bar.turnover = float(kdata["q"])
                bar.quote_asset_volume = float(kdata["q"])
                bar.number_of_trades = float(kdata["n"])
                bar.taker_buy_base_asset_volume = float(kdata["V"])
                bar.taker_buy_quote_asset_volume = float(kdata["Q"])

                self.stats.bar_count += 1
                self.gateway.on_bar(copy(bar))

        except Exception as e:
            self.logger.error(f"Shard {self.shard_id}: Error processing message: {e}")

    def _on_disconnected(self, *args) -> None:
        """Handle WebSocket disconnection"""
        self.stats.is_connected = False
        self.logger.warning(f"Shard {self.shard_id} disconnected")

    def _on_error(self, _, error) -> None:
        """Handle WebSocket error"""
        self.logger.error(f"Shard {self.shard_id} error: {error}")

    def stop(self) -> None:
        """Stop the shard"""
        self._active = False
        if self._client:
            self._client.stop()
        self.stats.is_connected = False


class ShardedKlineManager:
    """
    Manages multiple WebSocket shards for high-performance kline streaming.
    Automatically distributes symbols across shards using consistent hashing.

    Usage:
        manager = ShardedKlineManager(gateway, shard_count=4)
        manager.connect("REAL")
        manager.subscribe_batch(["btcusdt", "ethusdt", ...])
    """

    def __init__(
        self, gateway: "BinanceSpotGateway", shard_count: int = DEFAULT_SHARD_COUNT
    ):
        self.gateway = gateway
        self.shard_count = shard_count
        self.shards: list[KlineWebsocketShard] = []
        self.symbol_to_shard: dict[str, int] = {}

        self.logger = logging.getLogger("ShardedKlineManager")
        self._lock = Lock()

    def connect(self, server: str = "REAL") -> None:
        """Initialize and connect all shards"""
        if server == "REAL":
            stream_url = WEBSOCKET_DATA_HOST
        else:
            stream_url = TESTNET_WEBSOCKET_DATA_HOST

        self.logger.info(
            f"Initializing {self.shard_count} shards using {JSON_LIBRARY} for JSON parsing"
        )

        for i in range(self.shard_count):
            shard = KlineWebsocketShard(
                shard_id=i,
                stream_url=stream_url,
                gateway=self.gateway,
                logger=self.logger,
            )
            shard.connect()
            self.shards.append(shard)

        self.gateway.write_log(
            f"ShardedKlineManager: {self.shard_count} shards initialized"
        )

    def subscribe(self, symbol: str, interval: Interval = Interval.MINUTE) -> None:
        """Subscribe to a symbol's kline stream"""
        with self._lock:
            if symbol in self.symbol_to_shard:
                return

            # Assign to shard using consistent hashing
            shard_id = hash(symbol) % self.shard_count
            shard = self.shards[shard_id]

            shard.add_symbol(symbol, interval)
            self.symbol_to_shard[symbol] = shard_id

            # Subscribe immediately
            shard.subscribe_all(INTERVAL_VT2BINANCE.get(interval, "1m"))

    def subscribe_batch(
        self, symbols: list[str], interval: Interval = Interval.MINUTE
    ) -> None:
        """Subscribe to multiple symbols at once (more efficient for 100+ symbols)"""
        with self._lock:
            # Group symbols by shard
            shard_symbols: dict[int, list[str]] = {
                i: [] for i in range(self.shard_count)
            }

            for symbol in symbols:
                if symbol in self.symbol_to_shard:
                    continue
                shard_id = hash(symbol) % self.shard_count
                shard_symbols[shard_id].append(symbol)
                self.symbol_to_shard[symbol] = shard_id

            # Add symbols to each shard and subscribe
            for shard_id, syms in shard_symbols.items():
                if not syms:
                    continue
                shard = self.shards[shard_id]
                for sym in syms:
                    shard.add_symbol(sym, interval)
                shard.subscribe_all(INTERVAL_VT2BINANCE.get(interval, "1m"))
                self.gateway.write_log(
                    f"Shard {shard_id}: Subscribed to {len(syms)} symbols"
                )

    def get_stats(self) -> dict:
        """Get statistics for all shards"""
        return {
            "shard_count": self.shard_count,
            "json_library": JSON_LIBRARY,
            "total_symbols": len(self.symbol_to_shard),
            "shards": [
                {
                    "id": s.stats.shard_id,
                    "symbols": len(s.stats.symbols),
                    "bars_received": s.stats.bar_count,
                    "messages_received": s.stats.message_count,
                    "connected": s.stats.is_connected,
                    "last_message": s.stats.last_message_time.isoformat()
                    if s.stats.last_message_time
                    else None,
                }
                for s in self.shards
            ],
        }

    def stop(self) -> None:
        """Stop all shards"""
        for shard in self.shards:
            shard.stop()


class BinanceSpotDataWebsocketApi:
    """
    币安现货行情Websocket API

    Optimized for high-performance kline monitoring of 100+ symbols.
    Uses sharded WebSocket connections internally for better performance.
    API remains unchanged - use subscribe(req) as before.
    """

    def __init__(self, gateway: BinanceSpotGateway) -> None:
        """构造函数"""
        super().__init__()

        self.gateway: BinanceSpotGateway = gateway
        self.gateway_name: str = gateway.gateway_name
        self._server: str = "REAL"

        # Sharded manager for kline subscriptions (high-performance)
        self._sharded_manager: ShardedKlineManager | None = None

        # Legacy single client for tick/depth subscriptions
        self._client: SpotWebsocketStreamClient_vnpy | None = None

        self.ticks: dict[str, TickData] = {}
        self.bars: dict[str, BarData] = {}
        self.reqid: int = 0
        self._active: bool = False

    def connect(self, server: str):
        """连接Websocket行情频道"""
        self._server = server

        if server == "REAL":
            stream_url = WEBSOCKET_DATA_HOST
        else:
            stream_url = TESTNET_WEBSOCKET_DATA_HOST

        # Initialize sharded manager for kline subscriptions
        self._sharded_manager = ShardedKlineManager(self.gateway, DEFAULT_SHARD_COUNT)
        self._sharded_manager.connect(server)

        # Legacy client for tick/depth (if needed)
        is_combined = True
        if self._client:
            url_with_mode = self._client.socket_manager.stream_url.split("?timeUnit=")[
                0
            ]
            if is_combined and url_with_mode == stream_url + "/stream":
                pass
            elif not is_combined and url_with_mode == stream_url + "/ws":
                pass
            else:
                self._client.logger.warning(
                    "BinanceSpotDataWebsocketApi.connect: 重连不同模式的Websocket，先断开旧连接"
                )
                self._client.stop()

        self._client = SpotWebsocketStreamClient_vnpy(
            stream_url=stream_url,
            on_message=self.on_packet,
            on_close=self.on_disconnected,
            is_combined=is_combined,
        )
        self._active = True
        self.on_connected()

    def on_connected(self) -> None:
        """连接成功回报"""
        self.gateway.write_log(
            f"行情Websocket API连接成功 (Sharded mode, {DEFAULT_SHARD_COUNT} connections)"
        )

        # 重新订阅tick行情
        if self.ticks:
            for symbol in self.ticks.keys():
                self._client.ticker(symbol)
                self._client.partial_book_depth(symbol)

        # 重新订阅kline行情 (via sharded manager)
        if self.bars:
            for symbol, bar in self.bars.items():
                interval = getattr(bar, "interval", Interval.MINUTE)
                self._sharded_manager.subscribe(symbol, interval)

    def subscribe(self, req: SubscribeRequest) -> None:
        """订阅行情, 并send_message_to_server (same API as before)"""
        if req.exchange.value != req.exchange.value:
            return
        if req.symbol in self.bars:
            return

        if req.symbol not in symbol_contract_map:
            self.gateway.write_log(f"找不到该标的代码{req.symbol}")
            return

        self.reqid += 1

        bar: BarData = BarData(
            symbol=req.symbol,
            exchange=Exchange.BINANCE,
            datetime=datetime.fromtimestamp(0, tz=timezone.utc),
            gateway_name=self.gateway_name,
            interval=req.interval,
        )
        self.bars[req.symbol] = bar

        # Use sharded manager for kline subscription (high-performance)
        if self._sharded_manager:
            self._sharded_manager.subscribe(req.symbol, req.interval)

    def on_packet(self, _, packet: dict) -> None:
        """push data event when receives websocket response (for tick/depth data)"""
        if isinstance(packet, str):
            packet = fast_json_loads(packet)
        stream: str | None = packet.get("stream", None)

        if not stream:
            return

        data: dict = packet["data"]
        symbol, channel = stream.split("@", 1)

        # Kline data is handled by sharded manager, skip here
        if channel.startswith("kline_"):
            return

        # Subscribe to tick data
        tick: TickData = self.ticks.get(symbol, None)
        if not tick:
            tick = TickData(
                symbol=symbol,
                name=symbol_contract_map[symbol].name,
                exchange=Exchange.BINANCE,
                datetime=datetime.now(),
                gateway_name=self.gateway_name,
            )
            self.ticks[symbol] = tick

        if channel == "ticker":
            tick.volume = float(data["v"])
            tick.turnover = float(data["q"])
            tick.open_price = float(data["o"])
            tick.high_price = float(data["h"])
            tick.low_price = float(data["l"])
            tick.last_price = float(data["c"])
            tick.datetime = datetime.fromtimestamp(float(data["E"]) / 1000)
        elif channel.startswith("depth"):
            bids: list = data["bids"]
            for n in range(min(5, len(bids))):
                price, volume = bids[n]
                tick.__setattr__("bid_price_" + str(n + 1), float(price))
                tick.__setattr__("bid_volume_" + str(n + 1), float(volume))

            asks: list = data["asks"]
            for n in range(min(5, len(asks))):
                price, volume = asks[n]
                tick.__setattr__("ask_price_" + str(n + 1), float(price))
                tick.__setattr__("ask_volume_" + str(n + 1), float(volume))

        if tick.last_price:
            tick.localtime = datetime.now()
            self.gateway.on_tick(copy(tick))

    def on_disconnected(self, *args) -> None:
        """连接断开回报"""
        if self._client:
            self._client.stop()
        self.gateway.write_log("行情Websocket API断开")

    def stop(self):
        """停止所有连接"""
        self._active = False
        if self._client:
            self._client.stop()
        if self._sharded_manager:
            self._sharded_manager.stop()


class SpotWebsocketStreamClient_vnpy(BinanceWebsocketClient):
    def __init__(
        self,
        stream_url="wss://stream.binance.com:9443",
        on_message=None,
        on_open=None,
        on_close=None,
        on_error=None,
        on_ping=None,
        on_pong=None,
        is_combined=False,
        timeout=None,
        logger=None,
        proxies: dict | None = proxies,
    ):
        if is_combined:
            stream_url = stream_url + "/stream"
        else:
            stream_url = stream_url + "/ws"
        super().__init__(
            stream_url,
            on_message=on_message,
            on_open=on_open,
            on_close=on_close,
            on_error=on_error,
            on_ping=on_ping,
            on_pong=on_pong,
            timeout=timeout,
            logger=logger,
            proxies=proxies,
        )

    def agg_trade(self, symbol: str | list[str], id=None, action=None, **kwargs):
        """Aggregate Trade Streams

        The Aggregate Trade Streams push trade information that is aggregated for a single taker order.

        Stream Name: <symbol>@aggTrade

        Update Speed: Real-time
        """
        if isinstance(symbol, str):
            symbol = [symbol]
        stream_name = [f"{s.lower()}@aggTrade" for s in symbol]

        self.send_message_to_server(stream_name, action=action, id=id)

    def trade(self, symbol: str | list[str], id=None, action=None, **kwargs):
        """Trade Streams

        The Trade Streams push raw trade information; each trade has a unique buyer and seller.

        Stream Name: <symbol>@trade

        Update Speed: Real-time
        """

        if isinstance(symbol, str):
            symbol = [symbol]
        stream_name = [f"{s.lower()}@trade" for s in symbol]

        self.send_message_to_server(stream_name, action=action, id=id)

    def kline(self, symbol: str | list[str], interval: str, id=None, action=None):
        """Kline/Candlestick Streams

        The Kline/Candlestick Stream push updates to the current klines/candlestick every second.

        Stream Name: <symbol>@kline_<interval>

        interval:
        m -> minutes; h -> hours; d -> days; w -> weeks; M -> months

        - 1m
        - 3m
        - 5m
        - 15m
        - 30m
        - 1h
        - 2h
        - 4h
        - 6h
        - 8h
        - 12h
        - 1d
        - 3d
        - 1w
        - 1M

        Update Speed: 2000ms
        """
        if isinstance(symbol, str):
            symbol = [symbol]
        stream_name = [f"{s.lower()}@kline_{interval}" for s in symbol]

        self.send_message_to_server(stream_name, action=action, id=id)

    def mini_ticker(self, symbol=None, id=None, action=None, **kwargs):
        """Individual symbol or all symbols mini ticker

        24hr rolling window mini-ticker statistics.
        These are NOT the statistics of the UTC day, but a 24hr rolling window for the previous 24hrs

        Stream Name: <symbol>@miniTicker or
        Stream Name: !miniTicker@arr

        Update Speed: 1000ms
        """

        if symbol is None:
            stream_name = "!miniTicker@arr"
        else:
            stream_name = f"{symbol.lower()}@miniTicker"

        self.send_message_to_server(stream_name, action=action, id=id)

    def ticker(self, symbol=None, id=None, action=None, **kwargs):
        """Individual symbol or all symbols ticker

        24hr rolling window ticker statistics for a single symbol.
        These are NOT the statistics of the UTC day, but a 24hr rolling window for the previous 24hrs.

        Stream Name: <symbol>@ticker or
        Stream Name: !ticker@arr

        Update Speed: 1000ms
        """

        if symbol is None:
            stream_name = "!ticker@arr"
        else:
            stream_name = f"{symbol.lower()}@ticker"
        self.send_message_to_server(stream_name, action=action, id=id)

    def book_ticker(self, symbol, id=None, action=None, **kwargs):
        """Individual symbol book ticker

        Pushes any update to the best bid or ask's price or quantity in real-time for a specified symbol.

        Stream Name: <symbol>@bookTicker

        Update Speed: realtime
        """

        self.send_message_to_server(
            f"{symbol.lower()}@bookTicker", action=action, id=id
        )

    def partial_book_depth(
        self, symbol: str, level=5, speed=1000, id=None, action=None, **kwargs
    ):
        """Partial Book Depth Streams

        Top bids and asks, Valid are 5, 10, or 20.

        Stream Names: <symbol>@depth<levels> OR <symbol>@depth<levels>@100ms.

        Update Speed: 1000ms or 100ms
        """
        self.send_message_to_server(
            f"{symbol.lower()}@depth{level}@{speed}ms", id=id, action=action
        )

    def rolling_window_ticker(self, symbol: str, windowSize: str, id=None, action=None):
        """Rolling window ticker statistics for a single symbol, computed over multiple windows.

        Stream Name: <symbol>@ticker_<window_size>

        Window Sizes: 1h, 4h, 1d

        Update Speed: 1000ms

        Note: This stream is different from the <symbol>@ticker stream. The open time "O" always starts on a minute, while the closing time "C" is the current time of the update. As such, the effective window might be up to 59999ms wider that <window_size>.
        """
        self.send_message_to_server(
            f"{symbol.lower()}@ticker_{windowSize}", id=id, action=action
        )

    def rolling_window_ticker_all_symbols(self, windowSize: str, id=None, action=None):
        """All Market Rolling Window Statistics Streams

        Rolling window ticker statistics for all market symbols, computed over multiple windows. Note that only tickers that have changed will be present in the array.

        Stream Name: !ticker_<window-size>@arr

        Window Size: 1h, 4h, 1d

        Update Speed: 1000ms
        """
        self.send_message_to_server(f"!ticker_{windowSize}@arr", id=id, action=action)

    def diff_book_depth(self, symbol: str, speed=1000, id=None, action=None, **kwargs):
        """Diff. Depth Stream

        Stream Name: <symbol>@depth OR <symbol>@depth@100ms

        Update Speed: 1000ms or 100ms

        Order book price and quantity depth updates used to locally manage an order book.
        """

        self.send_message_to_server(
            f"{symbol.lower()}@depth@{speed}ms", action=action, id=id
        )

    def user_data(self, listen_key: str, id=None, action=None, **kwargs):
        """Listen to user data by using the provided listen_key"""
        self.send_message_to_server(listen_key, action=action, id=id)
