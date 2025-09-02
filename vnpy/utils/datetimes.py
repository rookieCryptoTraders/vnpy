# @FilePath : utils
# @File     : datetimes.py
# @Time     : 2024/3/12 13:22
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: Date and time utility functions.

import datetime
import os
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering

import polars as pl

# Assuming vnpy is installed in the environment.
# If Interval does not have TICK, the functions will still work safely.
from vnpy.trader.constant import Interval

from enum import Enum
from functools import total_ordering


@total_ordering
class TimeFreq(Enum):
    """
    Represents time frequency. The minimum frequency is milliseconds (ms).
    The enum values are indicative and used for calculations. Note that 'months'
    and 'years' are approximations and do not have a fixed duration in milliseconds.
    """
    unknown = 0
    # us = 0.001  # microseconds
    ms = 1  # milliseconds
    s = ms * 1000  # seconds
    m = s * 60  # minutes
    h = m * 60  # hours
    d = h * 24  # days
    W = d * 7  # weeks
    M = d * 30  # months (approximated as 30 days)
    Y = d * 365  # years (approximated as 365 days)

    def faster_freq(self):
        members = sorted(self.__class__, key=lambda x: x.value)
        idx = members.index(self)
        if idx == 0:
            print(f"{self.name} has no faster member")
            return self
        return TimeFreq(members[idx - 1])

    def slower_freq(self):
        members = sorted(self.__class__, key=lambda x: x.value)
        idx = members.index(self)
        if idx == len(members) - 1:
            print(f"{self.name} has no slower member")
            return self
        return TimeFreq(members[idx + 1])

    # Arithmetic
    def __add__(self, other):
        return self.value + self._val(other)

    def __sub__(self, other):
        return self.value - self._val(other)

    def __mul__(self, other):
        return self.value * self._val(other)

    def __truediv__(self, other):
        return self.value / self._val(other)

    def __floordiv__(self, other):
        return self.value // self._val(other)

    def __mod__(self, other):
        return self.value % self._val(other)

    def __divmod__(self, other):
        return divmod(self.value, self._val(other))

    # Comparison
    def __eq__(self, other):
        return self.value == self._val(other)

    def __lt__(self, other):
        return self.value < self._val(other)

    def __le__(self, other):
        return self.value <= self._val(other)

    def __ge__(self, other):
        return self.value >= self._val(other)

    def __gt__(self, other):
        return self.value > self._val(other)

    # helper
    def _val(self, other) -> float | int:
        if isinstance(other, TimeFreq):
            return other.value
        elif isinstance(other, (int, float)):
            return other
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")


class DatetimeUtils:
    """
    A utility class providing static methods for date and time manipulation.

    Note on Timezones:
    Some methods accept a 'tz' parameter. The `set_tz` method changes the timezone
    globally by setting the 'TZ' environment variable. For applications requiring
    stable timezone behavior, it is recommended to call `set_tz` once at startup
    rather than per-operation.
    """

    # A pre-sorted list of frequency suffixes for efficient parsing.
    # Sorted by length descending to handle 'ms' before 'm' or 's'.
    _FREQ_SUFFIXES = sorted(
        [(member.name, member) for member in TimeFreq if member != TimeFreq.unknown],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    @classmethod
    def set_tz(cls, tz: str = "UTC"):
        """
        Sets the global timezone for the process by setting the TZ environment variable.

        Warning: This has global side effects and may not be safe in multi-threaded contexts.

        Parameters
        ----------
        tz : str
            The timezone identifier, e.g., 'UTC' or 'America/New_York'.
        """
        os.environ["TZ"] = tz

    @classmethod
    def normalize_time_str(cls, time_str: str) -> str:
        """
        Normalizes a time frequency string for consistency.
        Specifically, 'min' is converted to 'm'.

        Parameters
        ----------
        time_str : str
            A string representing a time frequency, e.g., '1min', '1m', '1s'.

        Returns
        -------
        str
            The normalized time string.
        """
        if time_str == "min":
            return "m"
        if time_str.endswith("min"):
            return time_str[:-3] + "m"
        return time_str

    @classmethod
    def normalize_unix(
            cls, unix: int | float, to_precision: str = "s"
    ) -> float | int:
        """
        Converts a Unix timestamp to the specified precision (seconds or milliseconds).
        This function infers the source precision based on the magnitude of the timestamp.

        Warning: The heuristic assumes timestamps with more than 10 digits (i.e., after
        Nov 20, 2286) are in milliseconds. This may not be accurate for all use cases.

        Parameters
        ----------
        unix : int or float
            The Unix timestamp.
        to_precision : str
            The target precision: 's' for seconds or 'ms' for milliseconds.

        Returns
        -------
        int or float
            The timestamp converted to the target precision.
        """
        is_ms = unix > 9999999999  # Heuristic: 10 digits for seconds.

        if to_precision == "s":
            if is_ms:
                unix /= 1000
        elif to_precision == "ms":
            if not is_ms:
                unix *= 1000
        else:
            raise NotImplementedError("Invalid 'to_precision'; use 's' or 'ms'.")

        return unix

    @classmethod
    def split_time_str(cls, time_str: str) -> tuple[int | float, TimeFreq]:
        """
        Parses a time string into its numerical value and frequency unit.
        Defaults to 1 if the numerical part is missing (e.g., 'm' -> 1, TimeFreq.m).

        Parameters
        ----------
        time_str : str
            The time string to parse, e.g., '1m', '30s', '100ms'.

        Returns
        -------
        tuple of (int or float, TimeFreq)
            A tuple containing the number and the corresponding TimeFreq enum.
        """
        time_str = cls.normalize_time_str(time_str)
        for suffix, freq_enum in cls._FREQ_SUFFIXES:
            if time_str.endswith(suffix):
                num_str = time_str[: -len(suffix)]
                if not num_str:
                    return 1, freq_enum  # Default to 1 if no number is specified

                number = float(num_str)
                return int(number) if number.is_integer() else number, freq_enum

        raise NotImplementedError(
            f"Invalid time_str '{time_str}', please check the input string."
        )

    @classmethod
    def str2freq(
            cls, time_str: str, ret_unit: TimeFreq | str = TimeFreq.m
    ) -> tuple[int, TimeFreq]:
        """
        Converts a time string into an integer multiple of a specified return unit.

        Parameters
        ----------
        time_str : str
            The time string to convert, e.g., '1m', '1s', '1ms'.
        ret_unit : TimeFreq or str
            The target time unit for the return value.

        Returns
        -------
        tuple of (int, TimeFreq)
            The converted integer value and the target time unit.
        """
        if isinstance(ret_unit, str):
            ret_unit = TimeFreq[cls.normalize_time_str(ret_unit)]

        number, freq = cls.split_time_str(time_str)
        converted_number = (number * freq.value) / ret_unit.value

        if not converted_number.is_integer():
            raise ValueError(
                f"'{time_str}' cannot be converted cleanly to an integer of '{ret_unit.name}'. Please reduce the ret_unit"
            )

        return int(converted_number), ret_unit

    @classmethod
    def freq2str(
            cls, freq: TimeFreq, ret_unit: TimeFreq | str | None = TimeFreq.m
    ) -> str:
        """
        Converts a TimeFreq enum member into a string relative to a return unit.
        Example: freq2str(TimeFreq.h, ret_unit='m') -> '60m'

        Parameters
        ----------
        freq : TimeFreq
            The input time frequency.
        ret_unit : TimeFreq or str
            The desired unit for the output string. Defaults to minutes ('m').

        Returns
        -------
        str
            The formatted time string.
        """
        if isinstance(ret_unit, str):
            ret_unit = TimeFreq[cls.normalize_time_str(ret_unit)]
        if ret_unit is None:
            ret_unit = TimeFreq.ms

        # The ratio of the two frequencies' millisecond values gives the numeric prefix.
        prefix = freq.value / ret_unit.value
        if not prefix.is_integer():
            raise ValueError(
                f"Frequency '{freq.name}' cannot be converted cleanly to unit '{ret_unit.name}'. Please reduce the ret_unit"
            )

        return f"{int(prefix)}{ret_unit.name}"

    @classmethod
    def freq2interval(cls, freq: TimeFreq) -> Interval:
        """
        Converts a TimeFreq enum member to a vnpy Interval.

        Parameters
        ----------
        freq : TimeFreq
            The input time frequency.

        Returns
        -------
        Interval
            The corresponding vnpy Interval.
        """
        freq_map = {
            # TimeFreq.s.value: Interval.TICK,
            TimeFreq.m.value: Interval.MINUTE,
            TimeFreq.h.value: Interval.HOUR,
            TimeFreq.d.value: Interval.DAILY,
            TimeFreq.W.value: Interval.WEEKLY,
        }
        if freq.value in freq_map:
            return freq_map[freq.value]
        else:
            raise NotImplementedError(
                f"Frequency '{freq.name}' cannot be converted to a vnpy Interval."
            )

    @classmethod
    def freq2unix(
            cls, freq: TimeFreq, ret_unit: TimeFreq | str | None = TimeFreq.ms
    ) -> int:
        """
        Converts a TimeFreq enum member into a Unix timestamp string relative to a return unit.
        Example: freq2unix(TimeFreq.h, ret_unit='s') -> 3600

        Parameters
        ----------
        freq : TimeFreq
            The input time frequency.
        ret_unit : TimeFreq or str
            The desired unit for the output string. Defaults to minutes ('m').

        Returns
        -------
        int
            The Unix timestamp.
        """
        if ret_unit is None:
            ret_unit = TimeFreq.ms
        if isinstance(ret_unit, str):
            ret_unit = TimeFreq[cls.normalize_time_str(ret_unit)]

        # The ratio of the two frequencies' millisecond values gives the numeric prefix.
        prefix = freq.value / ret_unit.value
        if not prefix.is_integer():
            raise ValueError(
                f"Frequency '{freq.name}' cannot be converted cleanly to unit '{ret_unit.name}'. Please reduce the ret_unit"
            )

        return int(prefix)

    @classmethod
    def unix2datetime(
            cls, unix: int | float, tz: str = "UTC"
    ) -> datetime.datetime:
        """
        Converts a Unix timestamp to a naive datetime object relative to a specified timezone.
        It uses the global 'TZ' environment variable for the conversion.

        Parameters
        ----------
        unix : int or float
            The Unix timestamp (in seconds or milliseconds).
        tz : str
            The timezone to interpret the timestamp in.

        Returns
        -------
        datetime.datetime
            The corresponding naive datetime object.
        """
        cls.set_tz(tz)
        unix_seconds = cls.normalize_unix(unix, to_precision="s")
        return datetime.datetime.fromtimestamp(unix_seconds)

    @classmethod
    def unix2ymd(cls, unix: int, tz="UTC") -> str:
        """
        Converts a Unix timestamp to a 'YYYY-MM-DD' string.

        Parameters
        ----------
        unix : int
            The Unix timestamp.
        tz : str
            The timezone for the conversion.

        Returns
        -------
        str
            The formatted date string.
        """
        dt = cls.unix2datetime(unix, tz)
        return dt.strftime("%Y-%m-%d")

    @classmethod
    def unix2datetime_polars(
            cls, df: pl.DataFrame, col: str = "datetime", tz: str = "UTC"
    ) -> pl.DataFrame:
        """
        Converts a Polars DataFrame column of Unix timestamps to a timezone-aware datetime column.
        Assumes the source timestamps are in UTC.

        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        col : str
            The name of the column with Unix timestamps.
        tz : str
            The target timezone for the datetime column.

        Returns
        -------
        pl.DataFrame
            DataFrame with the converted, timezone-aware datetime column.
        """
        time_unit = "ms" if df[col][0] > 9999999999 else "s"

        # Convert from epoch (assumed UTC) to a timezone-aware datetime.
        return df.with_columns(
            pl.from_epoch(pl.col(col), time_unit=time_unit).dt.convert_time_zone(tz)
        )

    @classmethod
    def datetime2unix(cls, dt: datetime.datetime, tz: str = "UTC") -> int:
        """
        Converts a datetime object to a Unix timestamp in milliseconds.

        Parameters
        ----------
        dt : datetime.datetime
            The input datetime object.
        tz : str
            The timezone to use for the conversion.

        Returns
        -------
        int
            The Unix timestamp in milliseconds.
        """
        cls.set_tz(tz)
        return int(dt.timestamp() * 1000)

    @classmethod
    def datetime2unix_polars(
            cls, df: pl.DataFrame, col: str, time_unit: str = "ms", tz: str = "UTC"
    ) -> pl.DataFrame:
        """
        Converts a naive datetime column in a Polars DataFrame to a Unix timestamp.
        The function localizes the naive datetime to the specified timezone first.

        Parameters
        ----------
        df : pl.DataFrame
            The input DataFrame.
        col : str
            The name of the datetime column.
        time_unit : str
            The unit for the output timestamp ('s' or 'ms').
        tz : str
            The timezone of the source naive datetime column.

        Returns
        -------
        pl.DataFrame
            DataFrame with the converted Unix timestamp column.
        """
        # Localize the naive datetime to the specified timezone before getting the epoch value.
        return df.with_columns(
            pl.col(col).dt.replace_time_zone(tz).dt.epoch(time_unit=time_unit)
        )

    @classmethod
    def interval2unix(
            cls, interval: Interval, ret_unit: TimeFreq | str = TimeFreq.ms
    ) -> int | float:
        """
        Get the time duration in specified units for a given vnpy Interval.

        Parameters:
            interval (Interval): A vnpy candlestick interval.
            ret_unit (TimeFreq or str): The desired return unit.

        Returns:
            int or float: The time duration corresponding to the interval.
        """
        if isinstance(ret_unit, str):
            ret_unit = TimeFreq[cls.normalize_time_str(ret_unit)]

        interval_map = {
            Interval.MINUTE: datetime.timedelta(minutes=1),
            Interval.HOUR: datetime.timedelta(hours=1),
            Interval.DAILY: datetime.timedelta(days=1),
            Interval.WEEKLY: datetime.timedelta(weeks=1),
        }
        # Safely handle if Interval.TICK exists and map it to 1 second.
        if hasattr(Interval, "TICK") and interval == Interval.TICK:
            interval_map[Interval.TICK] = datetime.timedelta(seconds=1)

        delta = interval_map.get(interval)
        if delta is None:
            raise NotImplementedError(f"Interval '{interval}' is not supported.")

        total_seconds = delta.total_seconds()
        if ret_unit == TimeFreq.ms:
            return total_seconds * 1000
        elif ret_unit == TimeFreq.s:
            return total_seconds
        else:
            # For other units, convert via milliseconds
            return (total_seconds * 1000) / ret_unit.value

    @classmethod
    def interval2freq(cls, interval: Interval) -> TimeFreq:
        """
        Convert a vnpy Interval to a TimeFreq enum.

        Parameters:
            interval (Interval): A vnpy candlestick interval.

        Returns:
            TimeFreq: The corresponding time frequency enum.
        """
        ms = cls.interval2unix(interval, ret_unit="ms")
        return TimeFreq(ms)

    @classmethod
    def interval2str(cls, interval: Interval) -> str:
        """
        Convert a vnpy Interval to a simple time string (e.g., '1m', '1h').
        This is an optimized, direct mapping.

        Parameters:
            interval (Interval): A vnpy candlestick interval.

        Returns:
            str: The corresponding time string.
        """
        interval_map = {
            Interval.MINUTE: "1m",
            Interval.HOUR: "1h",
            Interval.DAILY: "1d",
            Interval.WEEKLY: "1W",
        }
        if hasattr(Interval, "TICK"):
            interval_map[Interval.TICK] = "1s"

        if interval in interval_map:
            return interval_map[interval]
        else:
            raise NotImplementedError(
                f"Interval '{interval.name}' cannot be converted to a string."
            )
