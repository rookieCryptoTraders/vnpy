# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/vnpy/adapters
# @File     : overview.py
# @Time     : 2025/1/13 16:10
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
from __future__ import annotations

import atexit
import datetime
import json
import os
from typing import Dict, List, Union

from vnpy.config import VTSYMBOL_KLINE
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import BarOverview
from vnpy.trader.database import TV_BaseOverview
from vnpy.trader.object import HistoryRequest
from vnpy.trader.utility import load_json, save_json


class OverviewEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (Exchange, Interval)):
            return o.value
        elif isinstance(o, datetime.datetime):
            return o.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, datetime.date):
            return o.strftime("%Y-%m-%d")
        elif hasattr(o, '__dict__'):
            return o.__dict__
        else:
            return super().default(o)


class OverviewDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.dict_to_object, *args, **kwargs)

    def dict_to_object(self, d):
        if 'exchange' in d:
            d['exchange'] = Exchange(d['exchange'])
        if 'interval' in d:
            d['interval'] = Interval(d['interval'])
        for dt_field in ['start', 'end']:
            if dt_field in d:
                try:
                    d[dt_field] = datetime.datetime.strptime(d[dt_field], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    d[dt_field] = datetime.datetime.strptime(d[dt_field], '%Y-%m-%d')
                except TypeError:
                    print("TypeError", d[dt_field])
        return d


def save_overview(filename: str, overview_data: Dict[str, TV_BaseOverview]) -> None:
    # with open(path, 'w', encoding='utf-8') as f:
    #     json.dump(overview_data, f, cls=OverviewEncoder)

    # convert overview_data to dict
    overview_data_dict = {k: v.__dict__ for k, v in overview_data.items()}  # v is TV_BaseOverview

    # use vnpy save json
    save_json(filename, overview_data_dict, cls=OverviewEncoder, mode='w')


def load_overview(filename: str, overview_cls: TV_BaseOverview.__class__) -> Dict[str, TV_BaseOverview]:
    # if not os.path.exists(file_path):
    #     return {}
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     dic = json.load(f)
    #     for k, v in dic.items():
    #         dic[k] = cls(**v)
    #     return dic

    # use vnpy load json
    overviews: Dict[str, TV_BaseOverview] = {}
    overview_dict = load_json(filename=filename, cls=OverviewDecoder)
    for k, v in overview_dict.items():
        overviews[k] = overview_cls(**v)
    return overviews


"""def update_bar_overview(symbol: str,
                        exchange: Exchange,
                        interval: Interval,
                        bars: Union[List, pl.DataFrame],
                        file_path: str,
                        stream: bool = False) -> None:
    if isinstance(bars, list) and isinstance(bars[0], BarData):
        raise NotImplementedError()
    elif isinstance(bars, list) and isinstance(bars[0], (list, tuple)):
        overview_dict = load_overview(file_path, cls=BarOverview)
        # 新增overview
        vt_symbol: str = VTSYMBOL_KLINE.format(interval=interval.value, symbol=bars[0][1], exchange=exchange.name)
        if vt_symbol not in overview_dict:
            overview = BarOverview(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start=bars[0][0],  # the first value is datetime
                end=bars[-1][0],  # the first value is datetime
                count=len(bars)
            )
        else:
            # 读取已存储的overview
            overview = overview_dict[vt_symbol]

            if stream:
                overview.end = bars[-1][0]
                overview.count += len(bars)
            else:
                overview.start = min(bars[0][0], overview.start)
                overview.end = max(bars[-1][0], overview.end)
                overview.count = len(bars)  # 假设bars包含所有K线数据

        overview_dict[vt_symbol] = overview

        save_overview(file_path, overview_dict)
    elif isinstance(bars, pl.DataFrame):
        # find the first and last datetime
        start = bars['datetime'].min()
        end = bars['datetime'].max()
        count = len(bars)"""


def get_timedelta(interval: Interval) -> datetime.timedelta:
    """
    Get the timedelta for a given interval.

    Parameters:
        interval (Interval): Candlestick interval (e.g., 1m, 1h, 1d).

    Returns:
        timedelta: Time delta corresponding to the interval.
    """
    interval_map = {
        Interval.MINUTE: datetime.timedelta(minutes=1),
        Interval.HOUR: datetime.timedelta(hours=1),
        Interval.DAILY: datetime.timedelta(days=1),
        Interval.WEEKLY: datetime.timedelta(weeks=1)
    }
    return interval_map.get(interval, datetime.timedelta(minutes=1))  # Default to 1 minute if unknown


class OverviewHandler:
    """
    Handles the overview metadata for market bars in memory,
    loads data on startup, updates dynamically, and saves on exit.
    """

    def __init__(self, path: str):
        """
        Initialize OverviewHandler by loading existing overview data.

        Parameters:
            path (str): Path to the JSON file where overview data is stored.
        """
        self.filename = path
        self.overview_dict: Dict[str, BarOverview] = {}  # Stores bar metadata in memory
        self.load_overview()

        # Register the save function to execute when the program exits
        atexit.register(self.save_overview)

    def load_overview(self):
        """
        Load overview data from the JSON file into memory using OverviewDecoder.
        """
        if os.path.exists(self.filename):
            overview_data = load_json(self.filename, cls=OverviewDecoder)
            self.overview_dict = {k: BarOverview(**v) for k, v in overview_data.items()}
            print(f"OverviewHandler: Loaded {len(self.overview_dict)} overview records from {self.filename}.")
        else:
            print("OverviewHandler: No existing overview file found. Starting fresh.")

    def update_bar_overview(self, symbol: str, exchange: Exchange, interval: Interval, bars: List[tuple]):
        """
        Update the in-memory overview data when new bars arrive.

        Parameters:
            symbol (str): Trading symbol (e.g., BTCUSDT).
            exchange (Exchange): Exchange (e.g., Binance, CME).
            interval (Interval): Candlestick interval (e.g., 1m, 1h, 1d).
            bars (List[tuple]): List of bar data in tuple format (datetime, price, volume, etc.).
        """
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

    def save_overview(self):
        """
        Save the in-memory overview data to a JSON file using OverviewEncoder.
        """
        overview_data_dict = {k: v.__dict__ for k, v in self.overview_dict.items()}
        save_json(self.filename, overview_data_dict, cls=OverviewEncoder, mode="w")
        print(f"OverviewHandler: Saved {len(self.overview_dict)} overview records to {self.filename}.")

    def check_missing_data(self) -> List[HistoryRequest]:
        """
        Scan all overview records and detect missing historical data.
        Generates a list of HistoryRequest objects for any missing data.

        Returns:
            List[HistoryRequest]: List of missing data requests.
        """
        missing_requests = []
        current_time = datetime.datetime.now(tz=datetime.UTC)

        for vt_symbol, overview in self.overview_dict.items():
            expected_end = overview.end + get_timedelta(overview.interval)

            # If current time is significantly ahead, request missing data
            if current_time > expected_end:
                print(f"OverviewVT: Missing data detected for {vt_symbol}. Expected end: {expected_end}, Current time: {current_time}")

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
