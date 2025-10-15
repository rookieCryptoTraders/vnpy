from __future__ import annotations

import atexit
import copy
import json
import logging
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import field
from datetime import date, datetime, timedelta
from importlib import import_module
from itertools import product
from logging import INFO,WARNING
from pathlib import Path
from types import ModuleType
from typing import Literal, TypeVar, Optional, Union, Any, Type

from typing_extensions import Self

from vnpy.config import BAR_OVERVIEW_FILENAME, FACTOR_OVERVIEW_FILENAME, TICK_OVERVIEW_FILENAME, BAR_OVERVIEW_KEY, \
    TICK_OVERVIEW_KEY, \
    FACTOR_OVERVIEW_KEY
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.engine import Event, EventEngine
from vnpy.trader.event import EVENT_BAR, EVENT_LOG
from vnpy.trader.object import HistoryRequest, BarData, TickData,LogData
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import get_file_path, load_json
from vnpy.utils.atomic_writer_config import AtomicWriterConfig, ConfiguredAtomicWriter
from vnpy.utils.datetimes import DatetimeUtils, TimeFreq
from vnpy.utils.graceful_shutdown import get_shutdown_handler
from .utility import ZoneInfo

# if TYPE_CHECKING:
# from vnpy.trader.database import TimeRange,DataRange


DB_TZ = ZoneInfo(SETTINGS["database.timezone"])
SYSTEM_MODE = SETTINGS.get("system.mode", "LIVE")


def convert_tz(dt: datetime) -> datetime:
    """
    Convert timezone of datetime object to DB_TZ.
    """
    dt: datetime = dt.astimezone(DB_TZ)
    return dt.replace(tzinfo=None)


def ensure_datetime(dt: datetime | int | float | str) -> datetime:
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

    def __init__(
            self,
            start: datetime | int | float | str,
            end: datetime | int | float | str,
            interval: Interval | None = None,
    ):
        self.start = ensure_datetime(start)
        self.end = ensure_datetime(
            end
        )  # for system consistency, this attr indicates the start of the last bar
        self.interval = interval
        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}) must <= end time ({self.end})")

    @property
    def max_gap_ms(self):
        return DatetimeUtils.interval2unix(self.interval, ret_unit='ms')

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
        max_gap_ms = max_gap_ms if max_gap_ms else self.max_gap_ms
        if not max_gap_ms:
            raise ValueError("max_gap_ms must be provided for overlap check")
        max_gap = timedelta(milliseconds=max_gap_ms)
        return not (
                self.end + max_gap < other.start or other.end + max_gap < self.start
        )

    def merge_if_continuous(
            self,
            other: Self,
            max_gap_ms: timedelta | int | None = None,
    ) -> Self | None:
        """Try to merge with another range if they are continuous or close enough. else return self"""
        assert self.interval == other.interval
        if not max_gap_ms:
            max_gap_ms = self.interval
        if isinstance(max_gap_ms, timedelta):
            max_gap_ms = max_gap_ms.total_seconds() * 1000
        elif isinstance(max_gap_ms, Interval):
            max_gap_ms = DatetimeUtils.interval2unix(max_gap_ms, ret_unit="ms")
        else:
            raise ValueError("max_gap_ms must be timedelta or Interval")
        if self.overlaps(other, max_gap_ms=max_gap_ms):
            return self.union(other)

        # not overlapping
        return self

    def get_gap_with(self, other: Self) -> Self | None:
        """Get the gap between this range and another"""
        if self.overlaps(other):
            return None
        if self.end < other.start:
            return TimeRange(start=self.end, end=other.start, interval=self.interval)
        if other.end < self.start:
            return TimeRange(start=other.end, end=self.start, interval=self.interval)
        return None

    def __str__(self) -> str:
        return f"TimeRange({self.interval.value}: {self.start} - {self.end})"

    def intersection(self, other: Self) -> Self | None:
        """Get the intersection of this range with another"""
        assert self.interval == other.interval
        if not self.overlaps(other):
            return None
        return TimeRange(
            start=max(self.start, other.start),
            end=min(self.end, other.end),
            interval=self.interval
        )

    def union(self, other: Self) -> Self | None:
        """Get the union of this range with another if they overlap"""
        assert self.interval == other.interval
        if not self.overlaps(other):
            return None
        # they are overlapping
        return TimeRange(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            interval=self.interval
        )

    def subtract(self, other: Self) -> list[Self]:
        """Subtract another range from this one, returning the remaining parts"""
        assert self.interval == other.interval
        if not self.overlaps(other):
            return [copy.deepcopy(self)]

        result = []
        if self.start < other.start:
            result.append(
                TimeRange(start=self.start, end=other.start, interval=self.interval)
            )
        if self.end > other.end:
            result.append(
                TimeRange(start=other.end, end=self.end, interval=self.interval)
            )

        return result


class DataRange:
    """Manages a collection of time ranges with gap detection"""

    def __init__(
            self, interval: Interval | None = None, ranges: list[TimeRange] = None
    ):
        """Initialize with optional interval for gap calculation"""
        self.ranges: list[TimeRange] = ranges if ranges else []
        self.interval = interval
        self._start_dt: Optional[datetime] = None
        self._end_dt: Optional[datetime] = None
        self._start_timerange: Optional[TimeRange] = None
        self._end_timerange: Optional[TimeRange] = None
        self.max_gap_timedelta = timedelta(minutes=1)  # maximum allowable gap
        if self.interval is not None:
            self.max_gap_timedelta = IntervalUtil.get_interval_timedelta(self.interval)
        self._update_bounds()

    @property
    def start(self) -> datetime | None:
        """Get the earliest start time"""
        return self._start_dt

    @property
    def end(self) -> datetime | None:
        """Get the latest end time"""
        return self._end_dt

    def _update_bounds(self) -> None:
        """Update the start and end timestamps"""
        if not self.ranges:
            return

        self._start_dt = min(r.start for r in self.ranges)
        self._end_dt = max(r.end for r in self.ranges)
        self._start_timerange = min(self.ranges, key=lambda x: x.start)
        self._end_timerange = max(
            self.ranges, key=lambda x: x.end
        )  # todo: check if this is correct. should I use x.end?

    def add_ranges(self, ranges: list[TimeRange], inplace=True,
                   method: Literal["union", "intersection"] = "union",
                   ) -> None:
        """Add multiple time ranges and update bounds

        Parameters
        ----------
        ranges :
        inplace : bool
            if True, modify self.ranges in place and return None. if False, return the new list of ranges without modifying self.ranges
        method :

        Returns
        -------

        """
        if not ranges:
            return
        for time_range in ranges:
            assert time_range.interval == self.interval
            self.ranges = self.add_range(start=time_range.start, end=time_range.end, method=method, inplace=inplace)
        self._update_bounds()

    # This function was adjusted by Gemini
    def add_range(self, start: Optional[Union[datetime, int, float, str]] = None,
                  end: Optional[Union[datetime, int, float, str]] = None,
                  inplace=True,
                  timerange: Optional[TimeRange] = None,
                  method: Literal['union', 'intersection'] = 'union') -> Optional[list[TimeRange]]:
        """Add a new time range and merge/intersect with existing ones.

        Parameters
        ----------
        start :
        end :
        inplace : bool
            if True, modify self.ranges in place and return None. if False, return the new list of ranges without modifying self.ranges
        timerange :
        method :

        Returns
        -------

        """
        # Validate inputs: either timerange or both start and end must be provided
        if (timerange is None and (start is None or end is None)) or (
                timerange is not None and (start is not None or end is not None)
        ):
            raise ValueError(
                "Provide either a TimeRange object or both start and end times."
            )

        # Ensure start and end are valid datetime objects
        if start is not None and end is not None:
            start_dt = ensure_datetime(start)
            end_dt = ensure_datetime(end)
            if start_dt > end_dt:
                raise ValueError(
                    f"Start time ({start_dt}) must be <= end time ({end_dt})."
                )
        elif timerange:
            start_dt, end_dt = timerange.start, timerange.end
        else:
            raise ValueError("Invalid start/end time inputs.")

        new_range = TimeRange(start=start_dt, end=end_dt, interval=self.interval)
        ranges = self.ranges if inplace else copy.deepcopy(self.ranges)

        if method == "union":
            ranges.append(new_range)
            ranges.sort(key=lambda r: r.start)

            merged = []
            if ranges:
                merged.append(ranges[0])
                for r in ranges[1:]:
                    if merged[-1].overlaps(r):
                        merged[-1] = merged[-1].union(r)
                    else:
                        merged.append(r)

            if inplace:
                self.ranges = merged
                self._update_bounds()
            return merged

        elif method == "intersection":
            if not ranges:
                if inplace:
                    self.ranges = []
                    self._update_bounds()
                return []

            intersected_ranges = []
            for r in ranges:
                intersection = r.intersection(new_range)
                if intersection:
                    intersected_ranges.append(intersection)

            if inplace:
                self.ranges = intersected_ranges
                self._update_bounds()
            return intersected_ranges
        else:
            raise ValueError(f"Unsupported method: {method}")

    def get_gaps(
            self, start: datetime | None = None, end: datetime | None = None
    ) -> list[TimeRange]:
        """Get all gaps in the current ranges

        Notes
        --------
        equals to gaps_between([start]+[common time ranges]+[end])
        """
        if not self.ranges:
            if start and end:
                return [TimeRange(start=start, end=end, interval=self.interval)]
            return []

        # This function was adjusted by Gemini to prevent side effects.
        temp_ranges = copy.deepcopy(self.ranges)

        if start:
            temp_ranges.append(
                TimeRange(start=start, end=start, interval=self.interval)
            )
        if end:
            temp_ranges.append(TimeRange(start=end, end=end, interval=self.interval))

        temp_ranges.sort(key=lambda x: (x.start, x.end))

        # Merge overlapping ranges to handle cases where start/end markers overlap with existing data.
        merged = []
        if temp_ranges:
            current_merged = temp_ranges[0]
            for r in temp_ranges[1:]:
                if current_merged.overlaps(r):
                    current_merged = current_merged.union(r)
                else:
                    merged.append(current_merged)
                    current_merged = r
            merged.append(current_merged)

        # Iterate through ranges and find gaps
        gaps = []
        for i in range(len(merged) - 1):
            gap = merged[i].get_gap_with(merged[i + 1])
            if gap:
                gaps.append(gap)
        return gaps

    # def get_common(self):
    #     """get common time ranges existing in all ranges"""
    #     if not self.ranges:
    #         return []
    #
    #     common_ranges = []
    #     current_common = self.ranges[0]
    #
    #     for i in range(1, len(self.ranges)):
    #         next_range = self.ranges[i]
    #         if current_common.overlaps(next_range):
    #             current_common = current_common.intersection(next_range)
    #             if current_common is None:
    #                 # no common range
    #                 return []
    #         else:
    #             common_ranges.append(current_common)
    #             current_common = next_range
    #
    #     common_ranges.append(current_common)
    #     return common_ranges


class BaseOverview:
    symbol: str = ""
    exchange: Exchange = None
    interval: Interval = None
    count: int = 0
    data_range: DataRange = field(default=None, init=True)

    # vt_symbol: str = ""
    overview_key: str = ""

    def __init__(self, symbol: str = ""
                 , exchange: Exchange = None
                 , interval: Interval = None
                 , count: int = 0
                 , data_range: DataRange = None
                 , vt_symbol: str = ""
                 , overview_key: str = ""
                 ):
        self.symbol = symbol
        self.exchange = Exchange(exchange)
        self.interval = Interval(interval)
        self.count = count
        # self.vt_symbol = vt_symbol
        self.overview_key = overview_key  # which will be overwritten in post init
        self.data_range = data_range if data_range else DataRange(interval=self.interval)

    # def __post_init__(self):
    #     self.vt_symbol = VTSYMBOL.format(
    #         symbol=self.symbol,
    #         exchange=self.exchange.value,
    #     )

    def add_range(self, start: datetime, end: datetime, inplace=True):
        """Add a time range and get any gaps"""
        new_range = TimeRange(start=start, end=end, interval=self.interval)

        # Other time range operations are handled by DataRange class
        self.data_range.add_range(timerange=new_range,
                                  inplace=inplace)  # the default value of inplace in overview's add range should be true

    @property
    def start(self) -> datetime | None:
        """Get the earliest start time"""
        if not self.time_ranges:
            return None
        return min(r.start for r in self.time_ranges)

    @property
    def end(self) -> datetime | None:
        """Get the latest end time"""
        if not self.time_ranges:
            return None
        return max(r.end for r in self.time_ranges)

    @property
    def time_ranges(self) -> list[TimeRange]:
        return self.data_range.ranges


class BarOverview(BaseOverview):

    def __init__(self, symbol: str = "", exchange: Exchange = None, interval: Interval = None, count: int = 0,
                 data_range: DataRange = None, vt_symbol: str = "", overview_key: str = ""
                 ):
        super().__init__(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            count=count,
            # vt_symbol=vt_symbol,
            data_range=data_range,
            overview_key=overview_key  # which will be overwritten in post init. put it here to avoid error
        )

    # def __post_init__(self):
        # super().__post_init__()
        self.overview_key = BAR_OVERVIEW_KEY.format(
            interval=self.interval.value,
            symbol=self.symbol,
            exchange=self.exchange.value
        )


class TickOverview(BaseOverview):

    def __init__(self, symbol: str = "", exchange: Exchange = None, interval: Interval = None, count: int = 0,
                 data_range: DataRange = None, vt_symbol: str = "", overview_key: str = ""
                 ):
        super().__init__(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            count=count,
            # vt_symbol=vt_symbol,
            data_range=data_range,
            overview_key=overview_key  # which will be overwritten in post init. put it here to avoid error
        )

    # def __post_init__(self):
    #     super().__post_init__()
        self.overview_key = TICK_OVERVIEW_KEY.format(
            interval=self.interval.value,
            symbol=self.symbol,
            exchange=self.exchange.value
        )


class FactorOverview(BaseOverview):

    def __init__(self, symbol: str = "", exchange: Exchange = None, interval: Interval = None, count: int = 0,
                 data_range: DataRange = None, vt_symbol: str = "", overview_key: str = "", factor_name: str = "",
                 factor_key: str = "",
                 ):
        super().__init__(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            count=count,
            # vt_symbol=vt_symbol,
            data_range=data_range,
            overview_key=overview_key  # which will be overwritten in post init. put it here to avoid error
        )
        self.factor_name = factor_name
        self.factor_key = factor_key

    # def __post_init__(self):
    #     super().__post_init__()
        self.overview_key = FACTOR_OVERVIEW_KEY.format(
            interval=self.interval.value,
            symbol=self.symbol,
            exchange=self.exchange.value,
            factor_key=self.factor_key
        )


class OverviewEncoder(json.JSONEncoder):
    # This class was refactored by Gemini for simplicity.
    def default(self, o):
        if o is None:
            return None
        elif isinstance(o, (Exchange, Interval)):
            return o.value
        elif isinstance(o, datetime):
            return o.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, date):
            return o.strftime("%Y-%m-%d")
        elif isinstance(o, DataRange):
            return {
                'class': o.__class__.__name__,
                'start': self.default(o.start),
                'end': self.default(o.end),
                'interval': self.default(o.interval),
                'ranges': [self.default(r) for r in o.ranges]
            }
        elif isinstance(o, TimeRange):
            return {
                'class': o.__class__.__name__,
                'start': self.default(o.start),
                'end': self.default(o.end),
                'interval': self.default(o.interval),
                # 'max_gap_ms': o.max_gap_ms
            }
        elif hasattr(o, '__dict__'):
            dic = {'class': o.__class__.__name__}
            dic.update(o.__dict__.copy())
            return dic
        else:
            return super().default(o)


class OverviewDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.dict_to_object, *args, **kwargs)

    def dict_to_object(self, d):
        if not isinstance(d, dict):
            return d
        # simple attr transformation, change the things in d
        if 'exchange' in d:
            d['exchange'] = Exchange(d['exchange'])
        if 'interval' in d:
            d['interval'] = Interval(d['interval'])
        for dt_field in ['start', 'end']:
            if dt_field in d:
                try:
                    if d[dt_field] is not None:
                        d[dt_field] = datetime.strptime(d[dt_field], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    d[dt_field] = datetime.strptime(d[dt_field], '%Y-%m-%d')
                except TypeError:
                    print(f"TypeError in {self.__class__.__name__}", d[dt_field])

        # complex attr transformation, create new objects
        if 'class' in d and d['class'] == 'TimeRange':
            d = TimeRange(
                start=d['start'],
                end=d['end'],
                interval=Interval(d['interval']),
            )
        elif 'class' in d and d['class'] == 'DataRange':
            dd = DataRange(
                interval=Interval(d['interval']), ranges=[self.dict_to_object(r) for r in d['ranges']]
            )
            return dd
        return d


class OverviewHandler:
    """
    Manages data overviews with gap detection and data retrieval.
    # This class was refactored by Gemini for unified overview management.
    """

    bar_overview_filepath = str(get_file_path(BAR_OVERVIEW_FILENAME))
    tick_overview_filepath = str(get_file_path(TICK_OVERVIEW_FILENAME))
    factor_overview_filepath = str(get_file_path(FACTOR_OVERVIEW_FILENAME))

    def __init__(self, event_engine: Optional[EventEngine] = None, atomic_config: AtomicWriterConfig = None, *args,
                 **kwargs):
        """Initialize the overview manager"""
        # Register the save function to execute when the program exits
        atexit.register(self.save_all_overviews)
        signal.signal(
            signal.SIGINT, lambda sig, frame: (self.save_all_overviews(), sys.exit(0))
        )
        signal.signal(
            signal.SIGTERM, lambda sig, frame: (self.save_all_overviews(), sys.exit(0))
        )

        # basic infos
        self._event_engine = event_engine
        self.data_ranges: dict[str, dict[str, DataRange]] = {
            "bar": {},
            "factor": {},
            "tick": {}
        }
        self.intervals: list[Interval] = [
            Interval(interval) for interval in SETTINGS.get("gateway.intervals", [])
        ]
        self.minimum_freq: TimeFreq = min(
            [DatetimeUtils.interval2freq(intvl) for intvl in self.intervals]
        )
        self.symbols = SETTINGS.get("gateway.symbols", [])
        self.exchanges: list[Exchange] = [
            Exchange(ex) for ex in SETTINGS.get("gateway.exchanges", [])
        ]
        self.vt_symbols: list[str] = [
            f"{s}.{e.value}" for s, e in product(self.symbols, self.exchanges)
        ]
        self.mode: str = SYSTEM_MODE

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # migrate from zc's code
        self.config = atomic_config or AtomicWriterConfig(
            sync_mode="fsync",  # Ensure data is written to disk
            max_retries=2,  # Retry failed writes
            min_disk_space_mb=5,  # Require 5MB free space
            validate_permissions=True,
            cleanup_temp_files=True

        )
        self.atomic_writer = ConfiguredAtomicWriter(self.config)
        self.shutdown_handler = get_shutdown_handler()
        self.logger = logging.getLogger(__name__)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # init database overview file
        # todo: collect them in one dict
        self.overview_dict: dict[str, Type[BaseOverview]] = {}  # Stores metadata in memory
        self.bar_overview = self.load_overview(filename=self.bar_overview_filepath, overview_cls=BarOverview)
        self.tick_overview = self.load_overview(filename=self.tick_overview_filepath, overview_cls=TickOverview)
        self.factor_overview = self.load_overview(filename=self.factor_overview_filepath,
                                                  overview_cls=FactorOverview)
        # init all overview dicts if empty
        self.init_overviews()

    def init_overviews(self):
        """Initialize overview dictionaries if they are empty"""
        if not self.bar_overview:
            for symbol in self.symbols:
                for exchange in self.exchanges:
                    for interval in self.intervals:
                        overview = BarOverview(
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            count=0,
                        )
                        self.bar_overview[overview.overview_key] = overview
            self.save_overview(type_='bar')
            self.write_log("Initialized bar overview.",level=WARNING)

        if not self.tick_overview:
            for symbol in self.symbols:
                for exchange in self.exchanges:
                    for interval in self.intervals:
                        overview = TickOverview(
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            count=0,
                        )
                        self.tick_overview[overview.overview_key] = overview
            self.save_overview(type_='tick')
            self.write_log("Initialized tick overview.",level=WARNING)

        if not self.factor_overview:
            # factor overview is initialized when new factor is created
            pass

    def load_overview(self, filename: str, overview_cls: TV_BaseOverview.__class__) -> dict[str, TV_BaseOverview]:

        # use vnpy load json
        overviews: dict[str, TV_BaseOverview] = {}
        overview_dict = load_json(filename=filename, cls=OverviewDecoder)
        for k, v in overview_dict.items():
            overviews[k] = overview_cls(**v)
        return overviews

    def save_overview(self, type_: Literal['bar', 'factor', 'tick']) -> None:
        operation_id = f"save_overview_{type_}_{time.time()}"

        try:
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

            # Register operation for graceful shutdown
            self.shutdown_handler.register_operation(
                operation_id,
                "json_write",
                filename
            )

            # convert overview_data to dict
            overview_data_dict = {k: v.__dict__ for k, v in overview_data.items()}  # v is TV_BaseOverview

            # use vnpy save json
            # save_json(filename, overview_data_dict, cls=OverviewEncoder, mode='w')
            filepath: Path = get_file_path(filename)

            self.atomic_writer.write_atomic(overview_data_dict, filepath, mode="w+", cls=OverviewEncoder)
        except Exception as e:
            self.logger.error(f"Error saving {type_} overview data: {e}")
            raise
        finally:
            # Always unregister operation
            self.shutdown_handler.unregister_operation(operation_id)

    def get_overview_dict(self, type_: Optional[Literal["bar", "factor", "tick"]]) -> dict[str, TV_BaseOverview]:
        if type_ == 'bar':
            return self.bar_overview
        elif type_ == 'factor':
            return self.factor_overview
        elif type_ == 'tick':
            return self.tick_overview
        else:
            raise ValueError(f"task_type {type_} is not supported.")

    def __update_overview_dict__(self, type_: Optional[Literal["bar", "factor", "tick"]] = None,
                                 overview_dict: dict[str, TV_BaseOverview] = None):
        if type_ == 'bar':
            self.bar_overview = copy.deepcopy(overview_dict)
        elif type_ == 'factor':
            self.factor_overview = copy.deepcopy(overview_dict)
        elif type_ == 'tick':
            self.tick_overview = copy.deepcopy(overview_dict)
        else:
            raise ValueError(f"task_type {type_} is not supported.")

    def update_overview(self, type_: Literal["bar", "factor", "tick"],
                        overview_dict: dict[str, TV_BaseOverview] = None):
        self.__update_overview_dict__(type_=type_, overview_dict=overview_dict)

    # def update_bar_overview(self, symbol: str, exchange: Exchange, interval: Interval, bars: list[tuple]):
    #     """
    #     Update the in-memory overview data when new bars arrive.
    #
    #     Parameters:
    #         symbol (str): Trading symbol (e.g., BTCUSDT).
    #         exchange (Exchange): Exchange (e.g., Binance, CME).
    #         interval (Interval): Candlestick interval (e.g., 1m, 1h, 1d).
    #         bars (list[tuple]): list of bar data in tuple format (datetime, price, volume, etc.).
    #     """
    #     # todo: update overview here
    #     if not bars:
    #         return
    #
    #     vt_symbol = VTSYMBOL_KLINE.format(interval=interval.value, symbol=symbol, exchange=exchange.name)
    #
    #     # If this symbol has no stored overview, create a new entry
    #     if vt_symbol not in self.overview_dict:
    #         self.overview_dict[vt_symbol] = BarOverview(
    #             symbol=symbol,
    #             exchange=exchange,
    #             interval=interval,
    #             start=bars[0][0],  # First bar's timestamp
    #             end=bars[-1][0],  # Last bar's timestamp
    #             count=len(bars)
    #         )
    #     else:
    #         overview = self.overview_dict[vt_symbol]
    #
    #         # Update start/end timestamps and bar count
    #         overview.start = min(overview.start, bars[0][0])
    #         overview.end = max(overview.end, bars[-1][0])
    #         overview.count += len(bars)
    #
    #     print(f"OverviewHandler: Updated {vt_symbol} with {len(bars)} new bars.")

    def save_all_overviews(self) -> None:
        """
        Save all overview data to their respective files.
        This is called automatically on program exit.
        """
        print("save_all_overviews",self.bar_overview)
        self.save_overview(type_='bar')
        self.save_overview(type_='factor')
        self.save_overview(type_='tick')

    @property
    def event_engine(self) -> EventEngine | None:
        return self._event_engine

    def add_data(
            self,
            type_: Literal["bar", "factor", "tick"],
            vt_symbol: str,
            exchange: Exchange,
            start: datetime | int | float | str,
            end: datetime | int | float | str,
            interval: Interval | None = None,
    ):
        """Add data range and optionally handle gaps"""
        if type_ == "bar" and not interval:
            raise ValueError("interval is required for bar data")

        if not isinstance(exchange, Exchange):
            raise ValueError(
                f"exchange must be an Exchange instance, got {type(exchange)}"
            )

        # Convert times to datetime
        start_dt = ensure_datetime(start)
        end_dt = ensure_datetime(end)

        # Get or create data range
        ranges = self.data_ranges[type_]
        if vt_symbol not in ranges:
            ranges[vt_symbol] = DataRange(interval=interval)

        # Add range and get gaps
        ranges[vt_symbol].add_range(start=start_dt, end=end_dt)

    def get_gaps(
            self, end_time: datetime | None = None, start_time: datetime | None = None
    ) -> dict[str, list[TimeRange]]:
        """Get requests for all missing data up to current_time"""
        # This function was adjusted by Gemini for clarity and correctness.
        if end_time is None:
            end_time = datetime.now()

        # The fixme is addressed by the logic: if overview_dict is empty, exist_dict will be empty, and no gaps are returned, which is correct.
        # get all existing data ranges and store them together
        exist_dict = {}
        for type_ in ["bar", "factor"]:
            overview_dict = self.get_overview_dict(type_=type_)
            for vt_symbol, overview in overview_dict.items():
                if exist_dict.get(overview.overview_key) is None:
                    exist_dict[overview.overview_key] = DataRange(
                        interval=overview.interval, ranges=copy.deepcopy(overview.time_ranges))
                else:
                    # To combine data from multiple sources, 'union' is generally expected.
                    # Using 'intersection' would only find data common to all sources.
                    exist_dict[overview.overview_key].add_ranges(
                        overview.time_ranges, method="union", inplace=True
                    )

        # Find and store gaps for each data range
        gap_dict = {}
        for overview_key, data_range in exist_dict.items():
            gaps = data_range.get_gaps(start=start_time, end=end_time)
            if gaps:
                gap_dict[overview_key] = gaps

        return gap_dict

    def process_missing_data(self, download_requests: list[HistoryRequest]) -> None:
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

    def write_log(self, msg: str, level: int = INFO) -> None:
        """"""
        if self.event_engine is None:
            return
        log: LogData = LogData(msg=msg, gateway_name='OverviewHandler', level=level)
        event: Event = Event(EVENT_LOG, log)
        self.event_engine.put(event)


# Factory function to create enhanced handler with default configuration
_overview_handler_instance: OverviewHandler | None = None
def get_overview_handler(
        sync_mode: Literal["fsync", "fdatasync", "none"] = "fsync",
        max_retries: int = 2,
        min_disk_space_mb: int = 5,
        event_engine=None
) -> OverviewHandler:
    """
    Get the singleton instance of the overview handler.
    """
    global _overview_handler_instance
    if _overview_handler_instance:
        return _overview_handler_instance

    config = AtomicWriterConfig(
        sync_mode=sync_mode,
        max_retries=max_retries,
        min_disk_space_mb=min_disk_space_mb,
        validate_permissions=True,
        cleanup_temp_files=True
    )
    _overview_handler_instance = OverviewHandler(event_engine=event_engine, atomic_config=config)
    return _overview_handler_instance

class BaseDatabase(ABC):
    """
    Abstract database class for connecting to different database.
    """
    overview_handler: Union[OverviewHandler, Any] = field(default_factory=get_overview_handler)
    database_name: str = ""

    @abstractmethod
    def save_bar_data(self, bars: list[BarData], stream: bool = False) -> bool:
        """
        Save bar data into database.
        """
        pass

    @abstractmethod
    def save_tick_data(self, ticks: list[TickData], stream: bool = False) -> bool:
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
    ) -> list[BarData]:
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
    ) -> list[TickData]:
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
    def get_bar_overview(self) -> list[BarOverview] | dict[str, BarOverview]:
        """
        Return bar data available in database.
        """
        pass

    @abstractmethod
    def get_tick_overview(self) -> list[TickOverview] | dict[str, TickOverview]:
        """
        Return tick data available in database.
        """
        pass

    @abstractmethod
    def get_factor_overview(self) -> list[FactorOverview] | dict[str, FactorOverview]:
        """
        Return factor data available in database.
        """
        pass

    def get_gaps(
            self, end_time: datetime | None = None, start_time: datetime | None = None
    ) -> dict[str, list[TimeRange]]:
        """
        Get gaps in data. As long as there is a missing value in the bar or factor, the time period of the missing value will be returned so that the data can be downloaded later to fill in the missing value.

        Parameters
        ----------
        end_time : Optional[datetime]
            The end time of the gap search. If None, current time is used.
        start_time : Optional[datetime]
            The start time of the gap search. If None, the earliest time in the overview is used.
        """

        gap_dict: dict = self.overview_handler.get_gaps(
            end_time=end_time, start_time=start_time
        )
        return gap_dict


# 1. Initialize the global database variable to None at the module level
database: BaseDatabase | None = None
TV_BaseOverview = TypeVar("TV_BaseOverview", bound=BaseOverview)  # TV means TypeVar


def get_database(*args, **kwargs) -> BaseDatabase:
    """
    Gets the singleton database instance for the application.

    The first time this function is called, it initializes the database
    connection based on global settings. Subsequent calls will return the
    existing instance, ignoring any new arguments.
    """
    # Use the 'global' keyword to modify the module-level variable
    global database

    # 2. Check if the instance already exists. If so, return it immediately.
    if database:
        return database

    # --- This part of the code will only run on the first call ---

    # Read database related global setting
    database_name: str = SETTINGS.get("database.name", "clickhouse")
    module_name: str = f"vnpy_{database_name}"

    # Try to import database module
    try:
        module: ModuleType = import_module(module_name)
    except ModuleNotFoundError:
        print(
            f"Cannot find database driver '{module_name}', using default SQLite database."
        )
        module: ModuleType = import_module("vnpy_sqlite")

    # 3. Create the database object and assign it to the global variable
    #    Note the use of *args and **kwargs to unpack the arguments
    database = module.Database(*args, **kwargs)
    print(f"Database instance '{database_name}' created and connected.")

    return database


