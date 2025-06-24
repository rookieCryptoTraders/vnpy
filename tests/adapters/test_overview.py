# -*- coding=utf-8 -*-
from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from unittest.mock import patch, mock_open

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import BarOverview, TickOverview, FactorOverview, TimeRange, DataRange, OverviewHandler
from vnpy.trader.utility import extract_vt_symbol


class TestTimeRange(unittest.TestCase):
    """Test TimeRange class"""

    def setUp(self):
        self.start = datetime(2025, 1, 1)
        self.end = datetime(2025, 1, 2)
        self.interval = Interval.DAILY
        self.time_range = TimeRange(start=self.start, end=self.end, interval=self.interval)

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.time_range.start, self.start)
        self.assertEqual(self.time_range.end, self.end)

        # Test invalid range
        with self.assertRaises(ValueError):
            TimeRange(start=self.end, end=self.start, interval=self.interval)

    def test_overlaps(self):
        """Test overlap detection"""
        # Test overlapping ranges
        other = TimeRange(
            start=self.start + timedelta(hours=12),
            end=self.end + timedelta(hours=12)
            , interval=self.interval
        )
        self.assertTrue(self.time_range.overlaps(other))

        other = TimeRange(
            start=self.start + timedelta(days=1),
            end=self.end + timedelta(days=1)
            , interval=self.interval
        )
        self.assertTrue(self.time_range.overlaps(other))

        other = TimeRange(
            start=self.end + timedelta(days=1),  # end+1d
            end=self.end + timedelta(days=1), interval=self.interval
        )
        self.assertTrue(self.time_range.overlaps(other))

        # Test non-overlapping ranges
        other = TimeRange(
            start=self.end + timedelta(days=2),  # end+2d
            end=self.end + timedelta(days=2)
            , interval=self.interval
        )
        self.assertFalse(self.time_range.overlaps(other))

    def test_merge(self):
        """Test range merging"""
        other = TimeRange(
            start=self.start + timedelta(hours=12),
            end=self.end + timedelta(hours=12), interval=self.interval
        )

        merged = self.time_range.merge_if_continuous(
            other,
            max_gap_ms=timedelta(days=1)
        )
        self.assertIsNotNone(merged)

        # Test result has correct bounds
        self.assertEqual(merged.start, self.start)
        self.assertEqual(
            merged.end,
            self.end + timedelta(hours=12)
        )

        # Test merging with gap
        other = TimeRange(
            start=self.end + timedelta(hours=1),
            end=self.end + timedelta(days=1), interval=self.interval
        )
        merged = self.time_range.merge_if_continuous(
            other,
            max_gap_ms=timedelta(hours=2)
        )
        self.assertIsNotNone(merged)

        # Test not merging with large gap
        merged = self.time_range.merge_if_continuous(
            other,
            max_gap_ms=timedelta(minutes=30)
        )
        self.assertIsNone(merged)


class TestDataRange(unittest.TestCase):
    """Test DataRange class"""

    def setUp(self):
        self.data_range = DataRange(interval=Interval.MINUTE)
        self.start = datetime(2025, 1, 1)
        self.end = datetime(2025, 1, 2)

    def test_add_range(self):
        """Test adding ranges"""
        # Add initial range
        gaps = self.data_range.add_range(start=self.start, end=self.end)
        # self.assertEqual(len(gaps), 0)
        self.assertEqual(len(self.data_range.ranges), 1)

        # Add overlapping range
        gaps = self.data_range.add_range(
            start=self.start + timedelta(hours=12),
            end=self.end + timedelta(hours=12)
        )
        # self.assertEqual(len(gaps), 0)
        self.assertEqual(len(self.data_range.ranges), 1)

        # Add range with gap
        gaps = self.data_range.add_range(
            start=self.end + timedelta(days=2),
            end=self.end + timedelta(days=2)
        )
        # self.assertEqual(len(gaps), 1)
        self.assertEqual(len(self.data_range.ranges), 2)

    def test_bounds(self):
        """Test start/end properties"""
        self.assertIsNone(self.data_range.start)
        self.assertIsNone(self.data_range.end)

        self.data_range.add_range(start=self.start, end=self.end)
        self.assertEqual(self.data_range.start, self.start)
        self.assertEqual(self.data_range.end, self.end)


class TestOverviewHandler(unittest.TestCase):
    """Test OverviewHandler class"""

    def setUp(self):
        self.handler = OverviewHandler()
        self.start = datetime(2025, 1, 1)
        self.end = datetime(2025, 1, 2)

    def test_add_bar_data(self):
        """Test adding bar data"""
        self.handler.add_data(
            type_="bar",
            vt_symbol="btcusdt.BINANCE",
            exchange=Exchange.BINANCE,
            start=self.start,
            end=self.end,
            interval=Interval.MINUTE
        )

        # Add data with gap
        self.handler.add_data(
            type_="bar",
            vt_symbol="btcusdt.BINANCE",
            exchange=Exchange.BINANCE,
            start=self.end + timedelta(hours=2),
            end=self.end + timedelta(days=1),
            interval=Interval.MINUTE
        )

    def test_missing_data(self):
        """Test missing data detection"""
        self.handler.add_data(
            type_="bar",
            vt_symbol="btcusdt.BINANCE",
            exchange=Exchange.BINANCE,
            start=self.start,
            end=self.end,
            interval=Interval.MINUTE
        )

        # no gaps
        end_time = self.end + timedelta(minutes=1)
        requests = self.handler.get_missing_data_requests(end_time=end_time)
        self.assertEqual(len(requests), 0)
        print(requests)

        # 1h
        end_time = self.end + timedelta(hours=1)
        requests = self.handler.get_missing_data_requests(end_time=end_time)
        self.assertEqual(len(requests), 1)
        print(requests)


        # has gaps and end time
        self.handler.data_ranges["bar"]["btcusdt.BINANCE"].add_range(
            start=self.end + timedelta(hours=2),
            end=self.end + timedelta(days=1)
        )
        end_time = self.end + timedelta(days=2)
        requests = self.handler.get_missing_data_requests(end_time=end_time)
        self.assertEqual(len(requests), 2)
        print(requests)

        # has gaps, start time and end time
        self.handler.data_ranges["bar"]["btcusdt.BINANCE"].add_range(
            start=self.end + timedelta(hours=2),
            end=self.end + timedelta(days=1)
        )
        start_time= self.start - timedelta(days=1)
        end_time = self.end + timedelta(days=2)
        requests = self.handler.get_missing_data_requests(end_time=end_time,start_time=start_time)
        self.assertEqual(len(requests), 3)
        print(requests)


def save_overview(filepath: str, overview_dict: dict):
    """Save overview dict to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(overview_dict, f, default=lambda x: x.__dict__)


def load_overview(filepath: str, cls):
    """Load overview dict from JSON file"""
    with open(filepath) as f:
        data = json.load(f)
        overview_dict = {}
        for key, value in data.items():
            overview = cls()
            for k, v in value.items():
                setattr(overview, k, v)
            overview_dict[key] = overview
        return overview_dict


if __name__ == '__main__':
    unittest.main()
