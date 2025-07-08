# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/tests/utils
# @File     : test_datetimes.py
# @Time     : 2025/1/6 20:36
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import unittest
from datetime import datetime

import polars as pl

from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import HistoryRequest
# from vnpy_datafeed.vnpy_datafeed.binance_datafeed import BinanceDatafeed
from vnpy.utils.datetimes import DatetimeUtils, TimeFreq


class TestTimeFreq(unittest.TestCase):
    def test_value(self):
        self.assertEqual(TimeFreq.ms.value, 1)
        self.assertEqual(TimeFreq.s.value, 1000)
        self.assertEqual(TimeFreq.m.value, 60000)
        self.assertEqual(TimeFreq.h.value, 3600000)
        self.assertEqual(TimeFreq.d.value, 86400000)
        self.assertEqual(TimeFreq.W.value, 604800000)
        self.assertEqual(TimeFreq.Y.value, 86400000 * 365)

    def test_truediv(self):
        # div int
        self.assertEqual(TimeFreq.s / 2, TimeFreq.ms.value * 500)
        self.assertEqual(TimeFreq.m / 2, TimeFreq.ms.value * 30000)
        self.assertEqual(TimeFreq.h / 2, TimeFreq.ms.value * 1800000)
        self.assertEqual(TimeFreq.d / 2, TimeFreq.ms.value * 43200000)
        self.assertEqual(TimeFreq.W / 2, TimeFreq.ms.value * 302400000)
        self.assertEqual(TimeFreq.Y / 2, TimeFreq.ms.value * (86400000 * 365) // 2)

        # div TimeFreq
        self.assertEqual(TimeFreq.s / TimeFreq.ms, 1000)
        self.assertEqual(TimeFreq.m / TimeFreq.ms, 60000)
        self.assertEqual(TimeFreq.h / TimeFreq.ms, 3600000)

    def test_arithmetic_and_comparison(self):
        # Arithmetic
        self.assertEqual(TimeFreq.h + TimeFreq.m, 3_600_000 + 60_000)
        self.assertEqual(TimeFreq.h - TimeFreq.m, 3_600_000 - 60_000)
        self.assertEqual(TimeFreq.h * 2, 7_200_000)
        self.assertEqual(TimeFreq.h / TimeFreq.m, 60.0)
        self.assertEqual(TimeFreq.h // TimeFreq.m, 60)
        self.assertEqual(TimeFreq.h % TimeFreq.d, 3_600_000)
        self.assertEqual(divmod(TimeFreq.h, TimeFreq.m), (60.0, 0.0))

        # Comparisons
        self.assertTrue(TimeFreq.h > TimeFreq.m)
        self.assertTrue(TimeFreq.h <= 3_600_000)

    def test_faster_freq(self):
        self.assertEqual(TimeFreq.unknown.faster_freq(), TimeFreq.unknown)
        self.assertEqual(TimeFreq.ms.faster_freq(), TimeFreq.unknown)
        self.assertEqual(TimeFreq.s.faster_freq(), TimeFreq.ms)
        self.assertEqual(TimeFreq.m.faster_freq(), TimeFreq.s)
        self.assertEqual(TimeFreq.h.faster_freq(), TimeFreq.m)
        self.assertEqual(TimeFreq.d.faster_freq(), TimeFreq.h)
        self.assertEqual(TimeFreq.W.faster_freq(), TimeFreq.d)
        self.assertEqual(TimeFreq.Y.faster_freq(), TimeFreq.M)

    def test_slower_freq(self):
        self.assertEqual(TimeFreq.unknown.slower_freq(), TimeFreq.ms)
        self.assertEqual(TimeFreq.ms.slower_freq(), TimeFreq.s)
        self.assertEqual(TimeFreq.Y.slower_freq(), TimeFreq.Y)

    def test_min_max(self):
        self.assertEqual(min([TimeFreq.m, TimeFreq.d, TimeFreq.Y]), TimeFreq.m)
        self.assertEqual(max([TimeFreq.m, TimeFreq.d, TimeFreq.Y]), TimeFreq.Y)


class TestDatetimeUtils(unittest.TestCase):

    def test_normalize_time_str_converts_min_to_m(self):
        self.assertEqual(DatetimeUtils.normalize_time_str('min'), 'm')

    def test_normalize_time_str_keeps_other_values(self):
        self.assertEqual(DatetimeUtils.normalize_time_str('1s'), '1s')
        self.assertEqual(DatetimeUtils.normalize_time_str('1ms'), '1ms')
        print(DatetimeUtils.normalize_time_str('1s'), '1s')
        print(DatetimeUtils.normalize_time_str('1ms'), '1ms')

    def test_normalize_unix_converts_to_seconds(self):
        self.assertEqual(DatetimeUtils.normalize_unix(1609459200000, 's'), 1609459200)

    def test_normalize_unix_converts_to_milliseconds(self):
        self.assertEqual(DatetimeUtils.normalize_unix(1609459200, 'ms'), 1609459200000)

    def test_split_time_str_parses_correctly(self):
        self.assertEqual(DatetimeUtils.split_time_str('1m'), (1, TimeFreq.m))
        self.assertEqual(DatetimeUtils.split_time_str('1ms'), (1, TimeFreq.ms))
        print(DatetimeUtils.split_time_str('1m'), (1, TimeFreq.m))
        print(DatetimeUtils.split_time_str('1ms'), (1, TimeFreq.ms))
        self.assertEqual(TimeFreq.ms.factor_name, 'ms')
        self.assertEqual(TimeFreq.ms.value, 1)
        self.assertEqual(TimeFreq.s.factor_name, 's')
        self.assertEqual(TimeFreq.s.value, 1000)

    def test_str2freq_converts_correctly(self):
        self.assertEqual(DatetimeUtils.str2freq('1m', ret_unit=TimeFreq.ms), (60000, TimeFreq.ms))
        self.assertEqual(DatetimeUtils.str2freq('1s', ret_unit=TimeFreq.ms), (1000, TimeFreq.ms))
        self.assertEqual(DatetimeUtils.str2freq(time_str='1.5m', ret_unit=TimeFreq.ms), (90000, TimeFreq.ms))
        self.assertEqual(DatetimeUtils.str2freq(time_str='1.5s', ret_unit=TimeFreq.ms), (1500, TimeFreq.ms))
        self.assertRaises(ValueError, DatetimeUtils.str2freq, time_str='1.5ms', ret_unit=TimeFreq.s)

    def test_unix2datetime_converts_correctly(self):
        self.assertEqual(DatetimeUtils.unix2datetime(1609459200), datetime(2021, 1, 1))

    def test_datetime2unix_converts_correctly(self):
        self.assertEqual(DatetimeUtils.datetime2unix(datetime(2021, 1, 1)), 1609459200000)

    def test_unix2ymd_converts_correctly(self):
        self.assertEqual(DatetimeUtils.unix2ymd(1609459200), '2021-01-01')

    def test_unix2datetime_polars_converts_correctly(self):
        df = pl.DataFrame({'datetime': [1609459200]})
        result = DatetimeUtils.unix2datetime_polars(df)
        self.assertEqual(result['datetime'][0], datetime(2021, 1, 1))

    def test_datetime2unix_polars_converts_correctly(self):
        df = pl.DataFrame({'datetime': [datetime(2021, 1, 1)]})
        result = DatetimeUtils.datetime2unix_polars(df, 'datetime')
        self.assertEqual(result['datetime'][0], 1609459200000)

    def test_interval2freq(self):
        self.assertEqual(DatetimeUtils.interval2freq(Interval.MINUTE), TimeFreq.m)
        self.assertEqual(DatetimeUtils.interval2freq(Interval.HOUR), TimeFreq.h)
        self.assertEqual(DatetimeUtils.interval2freq(Interval.DAILY), TimeFreq.d)
        self.assertEqual(DatetimeUtils.interval2freq(Interval.WEEKLY), TimeFreq.W)

        self.assertTrue(DatetimeUtils.interval2freq(Interval.MINUTE) > TimeFreq.ms)
        self.assertTrue(DatetimeUtils.interval2freq(Interval.MINUTE) > TimeFreq.s)
        self.assertTrue(DatetimeUtils.interval2freq(Interval.MINUTE) == TimeFreq.m)
        self.assertTrue(DatetimeUtils.interval2freq(Interval.MINUTE) < TimeFreq.d)

    def test_interval2unix(self):
        self.assertEqual(DatetimeUtils.interval2unix(Interval.MINUTE, ret_unit=TimeFreq.ms), 60000)
        self.assertEqual(DatetimeUtils.interval2unix(Interval.HOUR, ret_unit=TimeFreq.ms), 3600000)
        self.assertEqual(DatetimeUtils.interval2unix(Interval.DAILY, ret_unit=TimeFreq.ms), 86400000)
        self.assertEqual(DatetimeUtils.interval2unix(Interval.WEEKLY, ret_unit=TimeFreq.ms), 604800000)

        self.assertEqual(DatetimeUtils.interval2unix(Interval.MINUTE, ret_unit=TimeFreq.m), 1)


if __name__ == '__main__':
    unittest.main()
