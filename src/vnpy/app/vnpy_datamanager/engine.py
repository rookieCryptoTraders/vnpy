from __future__ import annotations

import csv
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from logging import INFO, ERROR
from typing import List, Optional, Union, TYPE_CHECKING, Literal

import polars as pl

from vnpy.config import match_format_string
from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import BaseDatabase, BarOverview, DB_TZ, TV_BaseOverview, TickOverview, \
    FactorOverview, TimeRange, VTSYMBOL_OVERVIEW
from vnpy.trader.datafeed import BaseDatafeed, get_datafeed
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.event import EVENT_LOG, EVENT_BAR_FILLING
from vnpy.trader.event import EVENT_DATAMANAGER_LOAD_BAR_REQUEST, EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST, \
    EVENT_DATAMANAGER_LOAD_BAR_RESPONSE, EVENT_DATAMANAGER_LOAD_FACTOR_RESPONSE
from vnpy.trader.object import ContractData
from vnpy.trader.object import HistoryRequest, BarData, TickData, FactorData
from vnpy.trader.object import LogData
from vnpy.trader.utility import ZoneInfo
from .base import APP_NAME

if TYPE_CHECKING:
    from vnpy_clickhouse.clickhouse_database import ClickhouseDatabase


class DataManagerEngine(BaseEngine):
    """"""

    def __init__(
            self,
            main_engine: MainEngine,
            event_engine: EventEngine,
            database: Union[
                          BaseDatabase, ClickhouseDatabase] | None = None
    ) -> None:
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

        # self.database: BaseDatabase = get_database()
        self.database: Union[
            BaseDatabase, ClickhouseDatabase] = database  # fixme: database should not affiliated to data_recorder. database is event driven
        self.datafeed: BaseDatafeed = get_datafeed()

    def init_engine(self):
        self.register_event()
        self.write_log(f"finish init engine")

    def register_event(self) -> None:
        """
        Register event handlers.
        """
        self.event_engine.register(EVENT_DATAMANAGER_LOAD_BAR_REQUEST, self.on_load_bar_data)
        self.event_engine.register(EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST, self.on_load_factor_data)

    def on_load_bar_data(self, event: Event) -> None:
        event_type, data = event.type, event.data
        if isinstance(data, list) and isinstance(data[0], dict):
            res = []
            for d in data:
                tmp=self.database.load_bar_data(**d)
                if tmp is not None:
                    res.append(tmp)
                else:
                    self.write_log(f"on_load_bar_data: no data found for {d}", level=INFO)
            res = pl.concat(res, how="vertical", rechunk=True)
        elif isinstance(data, dict):
            res = self.database.load_bar_data(
                **data
            )
        else:
            self.write_log(f"Invalid data format for {EVENT_DATAMANAGER_LOAD_BAR_REQUEST}: {data}", level=ERROR)
            raise TypeError(f"Invalid data format for {EVENT_DATAMANAGER_LOAD_BAR_REQUEST}: {data}")
        self.write_log(
            f"Successfully processed {EVENT_DATAMANAGER_LOAD_BAR_REQUEST}, response count: {len(res) if res is not None else 0}")
        self.put_event(Event(EVENT_DATAMANAGER_LOAD_BAR_RESPONSE, data=res))

    def on_load_factor_data(self, event: Event) -> None:
        event_type, data = event.type, event.data
        if isinstance(data, list) and isinstance(data[0], dict):
            res = []
            for d in data:
                res.append(self.database.load_factor_data(**d))
            res = pl.concat(res, how="vertical", rechunk=True)
        elif isinstance(data, dict):
            res = self.database.load_factor_data(
                **data
            )
        else:
            self.write_log(f"Invalid data format for {EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST}: {data}", level=ERROR)
            raise TypeError(f"Invalid data format for {EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST}: {data}")
        self.write_log(
            f"Successfully processed {EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST}, response count: {len(res)}")
        self.put_event(Event(EVENT_DATAMANAGER_LOAD_FACTOR_RESPONSE, data=res))

    def on_bar_filling(self, bar: BarData) -> None:
        """
        Bar event push.
        Bar event of a specific vt_symbol is also pushed.
        """
        self.put_event(Event(EVENT_BAR_FILLING, bar))
        self.put_event(Event(EVENT_BAR_FILLING + bar.vt_symbol, bar))

    def union_gaps(self,
                   gaps: dict[str, list[TimeRange]],
                   start: datetime = None,
                   end: datetime = None
                   ) -> dict[str, list[TimeRange]]:
        """
        Union gaps in the overview.
        """
        res = defaultdict(list)
        for overview_key, time_ranges in gaps.items():
            info = match_format_string(VTSYMBOL_OVERVIEW, overview_key)
            for time_range in time_ranges:
                if time_range.start < start:
                    time_range.start = start
                if time_range.end > end:
                    time_range.end = end
                res[overview_key].append(time_range)
        return res

    def import_data_from_csv(
            self,
            file_path: str,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            tz_name: str,
            datetime_head: str,
            open_head: str,
            high_head: str,
            low_head: str,
            close_head: str,
            volume_head: str,
            turnover_head: str,
            open_interest_head: str,
            datetime_format: str
    ) -> tuple:
        """"""
        with open(file_path, "rt") as f:
            buf: list = [line.replace("\0", "") for line in f]

        reader: csv.DictReader = csv.DictReader(buf, delimiter=",")

        bars: List[BarData] = []
        start: Optional[datetime] = None
        count: int = 0
        tz = ZoneInfo(tz_name)

        for item in reader:
            if datetime_format:
                dt: datetime = datetime.strptime(item[datetime_head], datetime_format)
            else:
                dt: datetime = datetime.fromisoformat(item[datetime_head])
            dt = dt.replace(tzinfo=tz)

            turnover = item.get(turnover_head, 0)
            open_interest = item.get(open_interest_head, 0)

            bar: BarData = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                interval=interval,
                volume=float(item[volume_head]),
                open_price=float(item[open_head]),
                high_price=float(item[high_head]),
                low_price=float(item[low_head]),
                close_price=float(item[close_head]),
                turnover=float(turnover),
                open_interest=float(open_interest),
                gateway_name="DB",
            )

            bars.append(bar)

            # do some statistics
            count += 1
            if not start:
                start = bar.datetime

        end: datetime = bar.datetime

        # insert into database
        self.database.save_bar_data(bars)

        return start, end, count

    def output_data_to_csv(
            self,
            file_path: str,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime
    ) -> bool:
        """"""
        bars: List[BarData] = self.load_bar_data(symbol, exchange, interval, start, end)

        fieldnames: list = [
            "symbol",
            "exchange",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "open_interest"
        ]

        try:
            with open(file_path, "w") as f:
                writer: csv.DictWriter = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()

                for bar in bars:
                    d: dict = {
                        "symbol": bar.symbol,
                        "exchange": bar.exchange.value,
                        "datetime": bar.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        "open": bar.open_price,
                        "high": bar.high_price,
                        "low": bar.low_price,
                        "close": bar.close_price,
                        "turnover": bar.turnover,
                        "volume": bar.volume,
                        "open_interest": bar.open_interest,
                    }
                    writer.writerow(d)

            return True
        except PermissionError:
            return False

    def get_bar_overview(self) -> dict[str, BarOverview]:
        return self.database.get_bar_overview()

    def get_tick_overview(self) -> dict[str, TickOverview]:
        return self.database.get_tick_overview()

    def get_factor_overview(self) -> dict[str, FactorOverview]:
        return self.database.get_factor_overview()

    def load_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime
    ) -> List[BarData]:
        """"""
        bars: List[BarData] = self.database.load_bar_data(
            symbol,
            exchange,
            interval,
            start,
            end
        )

        return bars

    def load_factor_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            factor_list: Union[list[str], str],
            start: datetime,
            end: datetime,
            ret: Literal["rows", "numpy", "pandas", "polars"] = "polars",
    ) -> list[FactorData]:
        factors: List[FactorData] = self.database.load_factor_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            factor_list=factor_list,
            start=start,
            end=end,
            ret=ret
        )
        return factors

    def delete_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval
    ) -> int:
        """"""
        count: int = self.database.delete_bar_data(
            symbol,
            exchange,
            interval
        )

        return count

    def download_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Union[str, Interval],
            start: datetime,
            end: datetime = None,
            save: bool = False
    ) -> Union[int, List[BarData]]:
        """
        Query bar data from datafeed.
        """
        if end is None:
            end = datetime.now(DB_TZ)
        if isinstance(interval, str):
            interval = Interval(interval)
        req: HistoryRequest = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )

        vt_symbol: str = f"{symbol}.{exchange.value}"
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        # If history data provided in gateway, then query
        if contract and contract.history_data:
            data: List[BarData] = self.main_engine.query_history(
                req, contract.gateway_name
            )
        # Otherwise use datafeed to query data
        else:
            data: List[dict] = self.datafeed.query_bar_history(req=req, output=self.write_log)

        if save:
            if data:
                status = self.database.save_bar_data(data)
                return status
        else:
            # If not saving, just return the data
            return data

        return 0

    def download_bar_data_20250629(
            self,
            requests: list[HistoryRequest],
            gateway_name: str = "",
    ) -> Union[int, List[BarData]]:
        """
        Query bar data from datafeed.
        """

        bar_data_list: List[BarData] = []
        for req in requests:
            data: List[BarData] = self.main_engine.query_history(
                req, gateway_name
            )
            bar_data_list.extend(data)

        return bar_data_list

    def download_tick_data(
            self,
            symbol: str,
            exchange: Exchange,
            start: datetime,
            output: Callable
    ) -> int:
        """
        Query tick data from datafeed.
        """
        req: HistoryRequest = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=datetime.now(DB_TZ)
        )

        data: List[TickData] = self.datafeed.query_tick_history(req, output)

        if data:
            self.database.save_tick_data(data)
            return len(data)

        return 0

    def download_missing_bars(self, requests: HistoryRequest) -> Union[TV_BaseOverview, None]:
        return None

    def download_bar_data_gaps(self, gap_dict: dict[str, list[TimeRange]]):
        """
        Download bar data for gaps in the overview.
        """
        res = defaultdict(list)
        for overview_key, time_ranges in gap_dict.items():
            start_dt = min([time_range.start for time_range in time_ranges])
            end_dt = max([time_range.end for time_range in time_ranges])
            self.write_log(f"download_bar_data_gaps: {overview_key}, {start_dt} - {end_dt}")
            info = match_format_string(VTSYMBOL_OVERVIEW, overview_key)
            for time_range in time_ranges:
                res[overview_key].extend(self.download_bar_data(symbol=info['symbol'],
                                                                exchange=Exchange(info['exchange']),
                                                                interval=Interval(info['interval']),
                                                                start=time_range.start,
                                                                end=time_range.end,
                                                                save=False))
        return res

    def write_log(self, msg: str, level=INFO) -> None:
        """
        Write log message to the log file.
        """
        log: LogData = LogData(msg=msg, gateway_name=APP_NAME, level=level)
        event = Event(type=EVENT_LOG, data=log)
        self.put_event(event)

    def put_event(self, event):
        """
        Put event to the event engine.
        """
        self.event_engine.put(event)

    def close(self) -> None:
        pass
