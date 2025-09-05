from __future__ import annotations

import atexit
import copy
import json
import os
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from importlib import import_module
from types import ModuleType
from typing import Literal, Self, TypeVar

from vnpy.config import (
    BAR_OVERVIEW_FILENAME,
    FACTOR_OVERVIEW_FILENAME,
    TICK_OVERVIEW_FILENAME,
    VTSYMBOL,
    VTSYMBOL_OVERVIEW,
)
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.engine import Event, EventEngine
from vnpy.trader.event import EVENT_BAR
from vnpy.trader.object import BarData, HistoryRequest, SubscribeRequest, TickData
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import get_file_path, load_json, save_json
from vnpy.utils.datetimes import DatetimeUtils

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
        self.max_gap_ms = DatetimeUtils.interval2unix(self.interval, ret_unit="ms")
        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}) must <= end time ({self.end})")

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
            interval=self.interval,
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
            interval=self.interval,
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
        self._start_dt: datetime | None = None
        self._end_dt: datetime | None = None
        # Calculate maximum allowable gap
        self.max_gap_timedelta = timedelta(minutes=1)
        if self.interval is not None:
            self.max_gap_timedelta = IntervalUtil.get_interval_timedelta(self.interval)

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
            self._start_dt = None
            self._end_dt = None
            return

        self._start_dt = min(r.start for r in self.ranges)
        self._end_dt = max(r.end for r in self.ranges)
        self._start_timerange = min(self.ranges, key=lambda x: x.start)
        self._end_timerange = max(
            self.ranges, key=lambda x: x.end
        )  # todo: check if this is correct. should I use x.end?

    def add_ranges(
        self,
        ranges: list[TimeRange],
        inplace=False,
        method: Literal["union", "intersection"] = "union",
    ) -> None:
        """Add multiple time ranges and update bounds"""
        if not ranges:
            return
        for range in ranges:
            assert range.interval == self.interval
            self.ranges = self.add_range(
                start=range.start, end=range.end, method=method, inplace=inplace
            )
        self._update_bounds()

    # This function was adjusted by Gemini
    def add_range(
        self,
        start: datetime | int | float | str | None = None,
        end: datetime | int | float | str | None = None,
        inplace=False,
        timerange: TimeRange | None = None,
        method: Literal["union", "intersection"] = "union",
    ) -> list[TimeRange] | None:
        """Add a new time range and merge/intersect with existing ones."""
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


@dataclass
class BaseOverview:
    symbol: str = ""
    exchange: Exchange = None
    interval: Interval = None
    count: int = 0
    time_ranges: list[TimeRange] = field(default_factory=list)
    data_range: DataRange = field(default=None, init=True)

    vt_symbol: str = ""
    overview_key: str = ""
    VTSYMBOL_TEMPLATE: str = field(default=None, init=False)

    def __post_init__(self):
        self.overview_key = VTSYMBOL_OVERVIEW.format(
            interval=self.interval.value if self.interval else "",
            symbol=self.symbol,
            exchange=self.exchange.value,
        )
        if self.data_range is None:
            self.data_range = DataRange(interval=self.interval, ranges=self.time_ranges)

    def add_range(self, start: datetime, end: datetime):
        """Add a time range and get any gaps"""
        self.data_range.add_range(start=start, end=end, inplace=True)
        self.time_ranges = self.data_range.ranges

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


@dataclass
class BarOverview(BaseOverview):
    VTSYMBOL_TEMPLATE = VTSYMBOL

    def __post_init__(self):
        self.vt_symbol = self.VTSYMBOL_TEMPLATE.format(
            symbol=self.symbol,
            exchange=self.exchange.value,
        )
        super().__post_init__()


@dataclass
class TickOverview(BaseOverview):
    VTSYMBOL_TEMPLATE = VTSYMBOL

    def __post_init__(self):
        self.vt_symbol = self.VTSYMBOL_TEMPLATE.format(
            symbol=self.symbol,
            exchange=self.exchange.value,
        )
        super().__post_init__()


@dataclass
class FactorOverview(BaseOverview):
    factor_name: str = ""
    factor_key: str = ""
    VTSYMBOL_TEMPLATE = VTSYMBOL

    def __post_init__(self):
        self.vt_symbol = self.VTSYMBOL_TEMPLATE.format(
            symbol=self.symbol,
            exchange=self.exchange.value,
        )
        super().__post_init__()
        # This logic was adjusted by Gemini for correctness.
        self.overview_key = f"{self.overview_key}.{self.factor_name}"


class OverviewEncoder(json.JSONEncoder):
    # This class was refactored by Gemini for simplicity.
    def default(self, o):
        if isinstance(o, (Exchange, Interval)):
            return o.value
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, BaseOverview):
            data = {
                "symbol": o.symbol,
                "exchange": o.exchange.value,
                "interval": o.interval.value if o.interval else None,
                "count": o.count,
                "start": o.start.isoformat() if o.start else None,
                "end": o.end.isoformat() if o.end else None,
            }
            if isinstance(o, FactorOverview):
                data["factor_name"] = o.factor_name
            return data
        return super().default(o)


class OverviewDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.dict_to_object, *args, **kwargs)

    def dict_to_object(self, d):
        if not isinstance(d, dict):
            return d
        # simple attr transformation, change the things in d
        if "exchange" in d:
            d["exchange"] = Exchange(d["exchange"])
        if "interval" in d and d["interval"]:
            d["interval"] = Interval(d["interval"])
        for dt_field in ["start", "end"]:
            if dt_field in d and d[dt_field]:
                d[dt_field] = datetime.fromisoformat(d[dt_field])

        return d


class OverviewHandler:
    """
    Manages data overviews with gap detection and data retrieval.
    # This class was refactored by Gemini for unified overview management.
    """

    bar_overview_filepath = str(get_file_path(BAR_OVERVIEW_FILENAME))
    tick_overview_filepath = str(get_file_path(TICK_OVERVIEW_FILENAME))
    factor_overview_filepath = str(get_file_path(FACTOR_OVERVIEW_FILENAME))

    def __init__(self, event_engine: EventEngine | None = None):
        """Initialize the overview manager"""
        atexit.register(self.save_all_overviews)
        signal.signal(
            signal.SIGINT, lambda sig, frame: (self.save_all_overviews(), sys.exit(0))
        )
        signal.signal(
            signal.SIGTERM, lambda sig, frame: (self.save_all_overviews(), sys.exit(0))
        )

        self.overview_files: dict[str, tuple[str, type[TV_BaseOverview]]] = {
            "bar": (self.bar_overview_filepath, BarOverview),
            "tick": (self.tick_overview_filepath, TickOverview),
            "factor": (self.factor_overview_filepath, FactorOverview),
        }

        self.overviews: dict[str, dict[str, TV_BaseOverview]] = {
            type_: self.load_overview(info[0], info[1])
            for type_, info in self.overview_files.items()
        }

        self.data_ranges: dict[str, dict[str, DataRange]] = {
            "bar": {},
            "factor": {},
            "tick": {},
        }
        self._event_engine = event_engine

    def load_overview(
        self, filename: str, overview_cls: type[TV_BaseOverview]
    ) -> dict[str, TV_BaseOverview]:
        """Load overview data from a JSON file."""
        if not os.path.exists(filename):
            return {}

        overview_data = load_json(filename, cls=OverviewDecoder)
        overviews = {}
        for key, item in overview_data.items():
            try:
                time_ranges = []
                if item.get("start") and item.get("end"):
                    time_ranges.append(
                        TimeRange(
                            start=item["start"],
                            end=item["end"],
                            interval=item["interval"],
                        )
                    )

                # Remove start/end from item before passing to constructor
                item.pop("start", None)
                item.pop("end", None)

                overview = overview_cls(time_ranges=time_ranges, **item)
                overviews[overview.overview_key] = overview
            except (TypeError, ValueError) as e:
                print(f"Error loading overview item {item}: {e}")
                continue
        return overviews

    def save_overview(self, type_: Literal["bar", "factor", "tick"]) -> None:
        """Save overview data to a JSON file."""
        if type_ not in self.overviews:
            raise ValueError(f"task_type {type_} is not supported.")

        overview_data = self.overviews[type_]
        filename, _ = self.overview_files[type_]
        filename = os.path.basename(filename)

        save_json(filename, overview_data, cls=OverviewEncoder, mode="w")

    def get_overview_dict(
        self, type_: Literal["bar", "factor", "tick"]
    ) -> dict[str, TV_BaseOverview]:
        """Get the overview dictionary for a specific data type."""
        if type_ not in self.overviews:
            raise ValueError(f"task_type {type_} is not supported.")
        return self.overviews[type_]

    def update_overview_dict(
        self,
        type_: Literal["bar", "factor", "tick"],
        overview_dict: dict[str, TV_BaseOverview],
    ):
        """Update the overview dictionary for a specific data type."""
        if type_ not in self.overviews:
            raise ValueError(f"task_type {type_} is not supported.")
        self.overviews[type_] = copy.deepcopy(overview_dict)

    def update_bar_overview(
        self, symbol: str, exchange: Exchange, interval: Interval, bars: list[tuple]
    ):
        """Update the in-memory bar overview data when new bars arrive."""
        if not bars:
            return

        bar_overviews = self.overviews["bar"]
        overview_key = VTSYMBOL_OVERVIEW.format(
            interval=interval.value, symbol=symbol, exchange=exchange.value
        )

        if overview_key not in bar_overviews:
            bar_overviews[overview_key] = BarOverview(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                count=len(bars),
                time_ranges=[
                    TimeRange(start=bars[0][0], end=bars[-1][0], interval=interval)
                ],
            )
        else:
            overview = bar_overviews[overview_key]
            overview.count += len(bars)
            overview.add_range(start=bars[0][0], end=bars[-1][0])

        print(f"OverviewHandler: Updated {overview_key} with {len(bars)} new bars.")

    def save_all_overviews(self) -> None:
        """Save all overview data to their respective files."""
        for type_ in self.overviews.keys():
            self.save_overview(type_=type_)

    def check_subscribe_stream(self) -> list[SubscribeRequest]:
        """Scan bar overviews and generate subscription requests."""
        subscribe_requests = []
        for _, bar_overview in self.overviews["bar"].items():
            print(
                f"OverviewHandler: Subscribing to real-time data for {bar_overview.vt_symbol}."
            )
            subscribe_requests.append(
                SubscribeRequest(
                    symbol=bar_overview.symbol,
                    exchange=bar_overview.exchange,
                    interval=bar_overview.interval,
                )
            )
        return subscribe_requests

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
                        interval=overview.interval,
                        ranges=copy.deepcopy(overview.time_ranges),
                    )
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
                        ),
                    )
                    self.event_engine.put(event)
            except Exception as e:
                print(f"Error processing request for {request.symbol}: {str(e)}")
                continue


class BaseDatabase(ABC):
    """
    Abstract database class for connecting to different database.
    """

    overview_handler: OverviewHandler = field(default_factory=OverviewHandler)
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
        end: datetime,
    ) -> list[BarData]:
        """
        Load bar data from database.
        """
        pass

    @abstractmethod
    def load_tick_data(
        self, symbol: str, exchange: Exchange, start: datetime, end: datetime
    ) -> list[TickData]:
        """
        Load tick data from database.
        """
        pass

    @abstractmethod
    def delete_bar_data(
        self, symbol: str, exchange: Exchange, interval: Interval
    ) -> int:
        """
        Delete all bar data with given symbol + exchange + interval.
        """
        pass

    @abstractmethod
    def delete_tick_data(self, symbol: str, exchange: Exchange) -> int:
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
    database_name: str = SETTINGS.get("database.name", "sqlite")
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
