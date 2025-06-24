import atexit
import copy
import datetime
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from importlib import import_module
from types import ModuleType
from typing import Dict, List, Literal, Optional, Union
from typing import TypeVar, Self

import polars as pl

from vnpy.config import BAR_OVERVIEW_FILENAME, FACTOR_OVERVIEW_FILENAME, TICK_OVERVIEW_FILENAME, VTSYMBOL_KLINE
from vnpy.config import VTSYMBOL_FACTORDATA
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.engine import Event, EventEngine
from vnpy.trader.event import EVENT_BAR
from vnpy.trader.object import HistoryRequest, SubscribeRequest, BarData, TickData, FactorData
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import extract_vt_symbol, get_file_path
from vnpy.trader.utility import load_json, save_json
from vnpy.utils.datetimes import DatetimeUtils
from .utility import ZoneInfo

from vnpy.adapters.overview import OverviewDecoder, OverviewEncoder

DB_TZ = ZoneInfo(SETTINGS["database.timezone"])


def convert_tz(dt: datetime) -> datetime:
    """
    Convert timezone of datetime object to DB_TZ.
    """
    dt: datetime = dt.astimezone(DB_TZ)
    return dt.replace(tzinfo=None)


def ensure_datetime(dt: Union[datetime, int, float, str]) -> datetime:
    """Ensure a value is a datetime object"""
    if isinstance(dt, datetime):
        return dt
    elif isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt)
    elif isinstance(dt, str):
        try:
            return datetime.fromisoformat(dt)
        except ValueError:
            try:
                return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    return datetime.strptime(dt, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Could not parse datetime from {dt}")
    else:
        raise ValueError(f"Could not convert {type(dt)} to datetime")


class IntervalUtil:
    """Utility for handling intervals"""

    @staticmethod
    def get_interval_timedelta(interval: Interval) -> timedelta:
        """Convert interval to timedelta"""
        if interval == Interval.MINUTE:
            return timedelta(minutes=1)
        elif interval == Interval.HOUR:
            return timedelta(hours=1)
        elif interval == Interval.DAILY:
            return timedelta(days=1)
        elif interval == Interval.WEEKLY:
            return timedelta(weeks=1)
        else:
            return timedelta(minutes=1)


class TimeRange:
    """A time range with gap detection capabilities"""

    def __init__(self, start: Union[datetime, int, float, str], end: Union[datetime, int, float, str],
                 interval: Optional[Interval] = None):
        self._start = ensure_datetime(start)
        self._end = ensure_datetime(end)  # for system consistency, this attr indicates the start of the last bar
        self.interval = interval
        self._max_gap_ms = DatetimeUtils.interval2unix(self.interval, ret_unit='ms')
        if self._start > self._end:
            raise ValueError(f"Start time ({self._start}) must <= end time ({self._end})")

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def end(self) -> datetime:
        return self._end

    def overlaps(self, other: Self, max_gap_ms: int = None) -> bool:
        """Check if this range overlaps with another

        Parameters
        ----------
        other :
        max_gap_ms : int
            valid maximum gap in milliseconds to consider as overlapping

        Returns
        -------

        """
        assert self.interval == other.interval
        max_gap_ms = max_gap_ms if max_gap_ms else self._max_gap_ms
        if not max_gap_ms: raise ValueError("max_gap_ms must be provided for overlap check")
        max_gap = timedelta(milliseconds=max_gap_ms)
        return not (self.end + max_gap < other.start or other.end + max_gap < self.start)

    def merge_if_continuous(self, other: Self, max_gap_ms: Optional[Union[timedelta, int]]=None) -> Optional[Self]:
        """Try to merge with another range if they are continuous or close enough"""
        assert self.interval==other.interval
        if not max_gap_ms:
            max_gap_ms=self.interval
        if isinstance(max_gap_ms, timedelta):
            max_gap_ms = max_gap_ms.total_seconds() * 1000
        if self.overlaps(other,max_gap_ms=max_gap_ms):
            return TimeRange(
                start=min(self.start, other.start),
                end=max(self.end, other.end),
                interval=self.interval
            )
        gap = self.get_gap_with(other)
        if gap and (gap.end - gap.start).total_seconds() * 1000 <= max_gap_ms:
            return TimeRange(
                start=min(self.start, other.start),
                end=max(self.end, other.end),
                interval=self.interval
            )
        return None

    def get_gap_with(self, other: Self) -> Optional[Self]:
        """Get the gap between this range and another"""
        if self.overlaps(other):
            return None
        if self.end < other.start:
            return TimeRange(start=self.end, end=other.start,
                             interval=self.interval)
        if other.end < self.start:
            return TimeRange(start=other.end, end=self.start,
                             interval=self.interval)
        return None

    def __str__(self) -> str:
        return f"TimeRange({self.interval.value}: {self.start} - {self.end})"


class DataRange:
    """Manages a collection of time ranges with gap detection"""

    def __init__(self, interval: Optional[Interval] = None):
        """Initialize with optional interval for gap calculation"""
        self.ranges: List[TimeRange] = []
        self.interval = interval
        self._start_dt: Optional[datetime] = None
        self._end_dt: Optional[datetime] = None
        # Calculate maximum allowable gap
        self.max_gap_timedelta = timedelta(minutes=1)
        if self.interval is not None:
            self.max_gap_timedelta = IntervalUtil.get_interval_timedelta(self.interval)

    @property
    def start(self) -> Optional[datetime]:
        """Get earliest start time"""
        return self._start_dt

    @property
    def end(self) -> Optional[datetime]:
        """Get latest end time"""
        return self._end_dt

    def _update_bounds(self) -> None:
        """Update the start and end timestamps"""
        if not self.ranges:
            self._start_dt = None
            self._end_dt = None
            return

        self._start_dt = min(r.start for r in self.ranges)
        self._end_dt = max(r.end for r in self.ranges)

    def add_range(self, start: Union[datetime, int, float, str], end: Union[datetime, int, float, str]) -> List[
        TimeRange]:
        """Add a new time range and return any gaps found"""
        # Convert start/end to datetime if needed
        start_dt = ensure_datetime(start)
        end_dt = ensure_datetime(end)

        new_range = TimeRange(start=start_dt, end=end_dt, interval=self.interval)
        # gaps: List[TimeRange] = []

        if not self.ranges:
            self.ranges = [new_range]
            self._start_dt = start_dt
            self._end_dt = end_dt
            return []

        # Sort ranges by start time
        self.ranges.sort(key=lambda x: x.start)

        # Try to merge with existing ranges
        merged = False
        for i, existing in enumerate(self.ranges):
            merged_range = existing.merge_if_continuous(new_range, self.max_gap_timedelta)
            if merged_range:
                # gap = existing.get_gap_with(new_range)
                # if gap and (gap.end - gap.start) > self.max_gap_timedelta:
                #     gaps.append(gap)
                self.ranges[i] = merged_range
                merged = True

                # Try to merge with subsequent ranges
                j = i + 1
                while j < len(self.ranges):
                    next_merged = merged_range.merge_if_continuous(self.ranges[j], self.max_gap_timedelta)
                    if next_merged:
                        self.ranges[i] = next_merged
                        self.ranges.pop(j)
                    else:
                        # if it cannot be merged, escape loop
                        break
                break

        if not merged:
            # If we couldn't merge, insert the new range in order
            insert_pos = 0
            while insert_pos < len(self.ranges) and self.ranges[insert_pos].start < new_range.start:
                insert_pos += 1

            self.ranges.insert(insert_pos, new_range)

            # Check for gaps with adjacent ranges
            # if insert_pos > 0:
            # gap = self.ranges[insert_pos - 1].get_gap_with(new_range)
            # if gap and (gap.end - gap.start) > self.max_gap_timedelta:
            #     gaps.append(gap)
            # if insert_pos < len(self.ranges) - 1:
            #     gap = new_range.get_gap_with(self.ranges[insert_pos + 1])
            #     if gap and (gap.end - gap.start) > self.max_gap_timedelta:
            #         gaps.append(gap)

        self._update_bounds()
        # return gaps
        return []

    def get_gaps(self) -> List[TimeRange]:
        """Get all gaps in the current ranges"""
        gaps = []
        if not self.ranges:
            return gaps

        # Iterate through ranges and find gaps
        for i in range(len(self.ranges) - 1):
            gap = self.ranges[i].get_gap_with(self.ranges[i + 1])
            if gap:
                gaps.append(gap)

        return gaps


@dataclass
class BaseOverview:
    symbol: str = ""
    exchange: Exchange = None
    interval: Interval = None
    count: int = 0
    # start: datetime = None  # replaced by property
    # end: datetime = None  # replaced by property
    time_ranges: List[TimeRange] = field(default_factory=list)

    vt_symbol: str = ""
    VTSYMBOL_TEMPLATE: str = field(default=None, init=False)

    def add_range(self, start: datetime, end: datetime) -> List[TimeRange]:
        """Add a time range and get any gaps"""
        new_range = TimeRange(start=start, end=end, interval=self.interval)
        if not self.time_ranges:
            self.time_ranges = [new_range]
            return []

        # Other time range operations are handled by DataRange class
        data_range = DataRange(interval=self.interval)
        data_range.ranges = self.time_ranges
        gaps = data_range.add_range(start=start, end=end)
        self.time_ranges = data_range.ranges
        return gaps

    @property
    def start(self) -> Optional[datetime]:
        """Get the earliest start time"""
        if not self.time_ranges:
            return None
        return min(r.start for r in self.time_ranges)

    @property
    def end(self) -> Optional[datetime]:
        """Get the latest end time"""
        if not self.time_ranges:
            return None
        return max(r.end for r in self.time_ranges)


@dataclass
class BarOverview(BaseOverview):
    """
    BarOverview of bar data stored in database.
    """

    VTSYMBOL_TEMPLATE = VTSYMBOL_KLINE

    def __post_init__(self):
        self.vt_symbol = self.VTSYMBOL_TEMPLATE.format(
            interval=self.interval.value,
            symbol=self.symbol,
            exchange=self.exchange.value
        )


@dataclass
class TickOverview(BaseOverview):
    """
    BarOverview of tick data stored in database.
    """

    VTSYMBOL_TEMPLATE = ""

    def __post_init__(self):
        raise NotImplementedError("TickOverview is not implemented yet.")
        self.vt_symbol = self.VTSYMBOL_TEMPLATE.format(
            interval=self.interval.value,
            symbol=self.symbol,
            exchange=self.exchange.value
        )


@dataclass
class FactorOverview(BaseOverview):
    """
    BarOverview of bar data stored in database.
    """

    factor_name: str = ""
    VTSYMBOL_TEMPLATE = VTSYMBOL_FACTORDATA

    def __post_init__(self):
        self.vt_symbol = self.VTSYMBOL_TEMPLATE.format(
            interval=self.interval.value,
            symbol=self.symbol,
            factorname=self.factor_name,
            exchange=self.exchange.value
        )


class BaseDatabase(ABC):
    """
    Abstract database class for connecting to different database.
    """

    @abstractmethod
    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """
        Save bar data into database.
        """
        pass

    @abstractmethod
    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """
        Save tick data into database.
        """
        pass

    @abstractmethod
    def load_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime
    ) -> List[BarData]:
        """
        Load bar data from database.
        """
        pass

    @abstractmethod
    def load_tick_data(
            self,
            symbol: str,
            exchange: Exchange,
            start: datetime,
            end: datetime
    ) -> List[TickData]:
        """
        Load tick data from database.
        """
        pass

    @abstractmethod
    def delete_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval
    ) -> int:
        """
        Delete all bar data with given symbol + exchange + interval.
        """
        pass

    @abstractmethod
    def delete_tick_data(
            self,
            symbol: str,
            exchange: Exchange
    ) -> int:
        """
        Delete all tick data with given symbol + exchange.
        """
        pass

    @abstractmethod
    def get_bar_overview(self) -> Union[List[BarOverview], dict[str, BarOverview]]:
        """
        Return bar data available in database.
        """
        pass

    @abstractmethod
    def get_tick_overview(self) -> Union[List[TickOverview], dict[str, TickOverview]]:
        """
        Return tick data available in database.
        """
        pass

    @abstractmethod
    def get_factor_overview(self) -> Union[List[FactorOverview], dict[str, FactorOverview]]:
        """
        Return factor data available in database.
        """
        pass


database: Optional[BaseDatabase] = None
TV_BaseOverview = TypeVar('TV_BaseOverview', bound=BaseOverview)  # TV means TypeVar


def get_database(*args, **kwargs) -> BaseDatabase:
    """"""
    # Return database object if already inited
    global database
    if database:
        return database

    # Read database related global setting
    database_name: str = SETTINGS["database.name"]
    module_name: str = f"vnpy_{database_name}"

    # Try to import database module
    try:
        module: ModuleType = import_module(module_name)
    except ModuleNotFoundError:
        print(f"Cannot find database driver {module_name}, using default SQLite database")
        module: ModuleType = import_module("vnpy_sqlite")

    # Create database object from module
    database = module.Database(args, kwargs)
    return database


class OverviewHandler:
    """Manages data overviews with gap detection and data retrieval"""
    bar_overview_filepath = str(get_file_path(BAR_OVERVIEW_FILENAME))
    tick_overview_filepath = str(get_file_path(TICK_OVERVIEW_FILENAME))
    factor_overview_filepath = str(get_file_path(FACTOR_OVERVIEW_FILENAME))

    def __init__(self, event_engine: Optional[EventEngine] = None):
        """Initialize the overview manager"""
        self.overview_dict: Dict[str, BarOverview] = {}  # Stores metadata in memory
        # init database overview file
        # todo: collect them in one dict
        self.bar_overview = self.load_overview_hyf(filename=self.bar_overview_filepath, overview_cls=BarOverview)
        self.tick_overview = self.load_overview_hyf(filename=self.tick_overview_filepath, overview_cls=TickOverview)
        self.factor_overview = self.load_overview_hyf(filename=self.factor_overview_filepath,
                                                      overview_cls=FactorOverview)

        # Register the save function to execute when the program exits
        atexit.register(self.save_overview_hyf, type_='bar')
        atexit.register(self.save_overview_hyf, type_='factor')
        atexit.register(self.save_overview_hyf, type_='tick')
        self.data_ranges: Dict[str, Dict[str, DataRange]] = {
            "bar": {},
            "factor": {},
            "tick": {}
        }
        self._event_engine = event_engine

    def load_overview_hyf(self, filename: str, overview_cls: TV_BaseOverview.__class__) -> Dict[str, TV_BaseOverview]:

        # use vnpy load json
        overviews: Dict[str, TV_BaseOverview] = {}
        overview_dict = load_json(filename=filename, cls=OverviewDecoder)
        for k, v in overview_dict.items():
            overviews[k] = overview_cls(**v)
        return overviews

    def save_overview_hyf(self, type_: Literal['bar', 'factor', 'tick']) -> None:
        if type_ == 'bar':
            overview_data = self.bar_overview
            filename = os.path.basename(self.bar_overview_filepath)
        elif type_ == 'factor':
            overview_data = self.factor_overview
            filename = os.path.basename(self.factor_overview_filepath)
        elif type_ == 'tick':
            overview_data = self.tick_overview
            filename = os.path.basename(self.tick_overview_filepath)
        else:
            raise ValueError(f"task_type {type_} is not supported.")

        # convert overview_data to dict
        overview_data_dict = {k: v.__dict__ for k, v in overview_data.items()}  # v is TV_BaseOverview

        # use vnpy save json
        save_json(filename, overview_data_dict, cls=OverviewEncoder, mode='w')

    def get_overview_dict(self, type_: Optional[Literal["bar", "factor", "tick"]]) -> Dict[str, TV_BaseOverview]:
        if type_ == 'bar':
            return self.bar_overview
        elif type_ == 'factor':
            return self.factor_overview
        elif type_ == 'tick':
            return self.tick_overview
        else:
            raise ValueError(f"task_type {type_} is not supported.")

    def __update_overview_dict__(self, type_: Optional[Literal["bar", "factor", "tick"]] = None,
                                 overview_dict: Dict[str, TV_BaseOverview] = None):
        if type_ == 'bar':
            self.bar_overview = overview_dict
        elif type_ == 'factor':
            self.factor_overview = overview_dict
        elif type_ == 'tick':
            self.tick_overview = overview_dict
        else:
            raise ValueError(f"task_type {type_} is not supported.")

    def update_overview_20250302(self, type_: Literal["bar", "factor", "tick"],
                                 overview_dict: Dict[str, TV_BaseOverview] = None):
        self.__update_overview_dict__(type_=type_, overview_dict=overview_dict)

    def update_bar_overview(self, symbol: str, exchange: Exchange, interval: Interval, bars: List[tuple]):
        """
        Update the in-memory overview data when new bars arrive.

        Parameters:
            symbol (str): Trading symbol (e.g., BTCUSDT).
            exchange (Exchange): Exchange (e.g., Binance, CME).
            interval (Interval): Candlestick interval (e.g., 1m, 1h, 1d).
            bars (List[tuple]): List of bar data in tuple format (datetime, price, volume, etc.).
        """
        # todo: update overview here
        if not bars:
            return

        vt_symbol = VTSYMBOL_KLINE.format(interval=interval.value, symbol=symbol, exchange=exchange.name)

        # If this symbol has no stored overview, create a new entry
        if vt_symbol not in self.overview_dict:
            self.overview_dict[vt_symbol] = BarOverview(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start=bars[0][0],  # First bar's timestamp
                end=bars[-1][0],  # Last bar's timestamp
                count=len(bars)
            )
        else:
            overview = self.overview_dict[vt_symbol]

            # Update start/end timestamps and bar count
            overview.start = min(overview.start, bars[0][0])
            overview.end = max(overview.end, bars[-1][0])
            overview.count += len(bars)

        print(f"OverviewHandler: Updated {vt_symbol} with {len(bars)} new bars.")

    def check_missing_data(self) -> List[HistoryRequest]:
        """
        Scan all overview records and detect missing historical data.
        Generates a list of HistoryRequest objects for any missing data.

        Returns:
            List[HistoryRequest]: List of missing data requests.
        """

        missing_requests = []
        current_time = datetime.datetime.now(tz=datetime.timezone.utc)

        for vt_symbol, overview in self.overview_dict.items():
            if overview.start is None:
                expected_end = datetime.datetime(2024, 1, 1, 0, 0, 0)
            else:
                expected_end = overview.end + DatetimeUtils.interval2unix(overview.interval)

            # If current time is significantly ahead, request missing data
            if current_time > expected_end:
                print(
                    f"OverviewVT: Missing data detected for {vt_symbol}. Expected end: {expected_end}, Current time: {current_time}")

                missing_requests.append(
                    HistoryRequest(
                        symbol=overview.symbol,
                        exchange=overview.exchange,
                        start=expected_end,
                        end=current_time,
                        interval=overview.interval
                    )
                )

        return missing_requests

    def check_subscribe_stream(self) -> List[SubscribeRequest]:
        """
        Scan all overview records and generate subscription requests for real-time tick data updates.

        Returns:
            List[SubscribeRequest]: A list of tick data subscription requests.
        """
        subscribe_requests = []
        for vt_symbol, bar_overview in self.overview_dict.items():
            print(f"OverviewHandler: Subscribing to real-time data for {vt_symbol}.")
            subscribe_requests.append(
                SubscribeRequest(
                    symbol=bar_overview.symbol,
                    exchange=bar_overview.exchange,
                    interval=bar_overview.interval
                )
            )

        return subscribe_requests

    @property
    def event_engine(self) -> Optional[EventEngine]:
        return self._event_engine

    def add_data(self,
                 type_: Literal["bar", "factor", "tick"],
                 vt_symbol: str,
                 exchange: Exchange,
                 start: Union[datetime, int, float, str],
                 end: Union[datetime, int, float, str],
                 interval: Optional[Interval] = None,
                 ):
        """Add data range and optionally handle gaps"""
        if type_ == "bar" and not interval:
            raise ValueError("interval is required for bar data")

        if not isinstance(exchange, Exchange):
            raise ValueError(f"exchange must be an Exchange instance, got {type(exchange)}")

        # Convert times to datetime
        start_dt = ensure_datetime(start)
        end_dt = ensure_datetime(end)

        # Get or create data range
        ranges = self.data_ranges[type_]
        if vt_symbol not in ranges:
            ranges[vt_symbol] = DataRange(interval=interval)

        # Add range and get gaps
        ranges[vt_symbol].add_range(start=start_dt, end=end_dt)

    def get_missing_data_requests(self, end_time: Optional[datetime] = None, start_time: Optional[datetime] = None) -> List[HistoryRequest]:
        """Get requests for all missing data up to current_time"""
        if end_time is None:
            end_time = datetime.now()

        requests = []
        for type_, ranges in self.data_ranges.items():
            for vt_symbol, data_range in ranges.items():
                if not data_range.end:
                    continue

                # Calculate expected next data point
                gaps= data_range.get_gaps()
                if start_time and start_time < data_range.start:
                    gaps=[TimeRange(start=start_time, end=data_range.start, interval=data_range.interval)] + gaps
                if end_time > data_range.end+IntervalUtil.get_interval_timedelta(data_range.interval):
                    gaps.append(TimeRange(start=data_range.end, end=end_time, interval=data_range.interval))

                if gaps:
                    # We have missing data
                    # For bar data, extract interval from vt_symbol
                    interval = data_range.interval

                    # Extract symbol and exchange
                    base_symbol, exchange_str = vt_symbol.rsplit(".", 1)
                    exchange = Exchange(exchange_str)

                    requests.extend([
                        HistoryRequest(
                            symbol=base_symbol,
                            exchange=exchange,
                            start=gap.start,
                            end=gap.end,
                            interval=interval
                        ) for gap in gaps]
                    )

        return requests

    def process_missing_data(self, download_requests: List[HistoryRequest]) -> None:
        """Process missing data requests by downloading and processing the data"""
        if not self.event_engine:
            raise RuntimeError("EventEngine is required for processing missing data")

        if not download_requests:
            return

        # Sort requests by time to maintain order
        download_requests.sort(key=lambda x: (x.symbol, x.start))

        for request in download_requests:
            try:
                # Create and emit bar events
                # Note: The actual data downloading should be handled by a data manager
                # This is just the event emission part
                if request.interval:  # Bar data
                    event = Event(
                        type=EVENT_BAR,
                        data=BarData(
                            symbol=request.symbol,
                            exchange=request.exchange,
                            datetime=request.start,
                            interval=request.interval,
                            volume=0,
                            open_price=0,
                            high_price=0,
                            low_price=0,
                            close_price=0,
                            open_interest=0,
                            turnover=0,
                        )
                    )
                    self.event_engine.put(event)
            except Exception as e:
                print(f"Error processing request for {request.symbol}: {str(e)}")
                continue
