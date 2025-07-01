import csv
from collections.abc import Callable
from datetime import datetime
from logging import INFO
from typing import List, Optional, Union
from collections import defaultdict

from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import BaseDatabase, get_database, BarOverview, DB_TZ, TV_BaseOverview, TickOverview, \
    FactorOverview, TimeRange, DataRange, VTSYMBOL_OVERVIEW
from vnpy.trader.datafeed import BaseDatafeed, get_datafeed
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.event import EVENT_LOG
from vnpy.trader.object import BarData, LogData
from vnpy.trader.object import TickData, ContractData, HistoryRequest
from vnpy.trader.utility import ZoneInfo
from vnpy.trader.object import HistoryRequest, SubscribeRequest, BarData, TickData
from vnpy.config import match_format_string

from .base import APP_NAME


class DataManagerEngine(BaseEngine):
    """"""

    def __init__(
            self,
            main_engine: MainEngine,
            event_engine: EventEngine,
            database: BaseDatabase | None = None
    ) -> None:
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

        # self.database: BaseDatabase = get_database()
        self.database: Union[
            BaseDatabase] = database  # fixme: database should not affiliated to data_manager. database is event driven
        self.datafeed: BaseDatafeed = get_datafeed()

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
            print(f"download_bar_data_gaps Processing overview key: {overview_key}")
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
        self.event_engine.put(event)

    def close(self) -> None:
        pass
