# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/vnpy/adapters
# @File     : overview.py
# @Time     : 2025/1/13 16:10
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
from __future__ import annotations

import json
import os
from typing import Dict, List, Union

import polars as pl

from vnpy.config import VTSYMBOL_KLINE
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import BarOverview, FactorOverview
from vnpy.trader.object import BarData
from vnpy.trader.utility import load_json, save_json


class OverviewEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (Exchange, Interval)):
            return o.value
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
        return d


def save_overview(filename: str, overview_data: Dict[str, Dict]) -> None:
    # with open(filename, 'w', encoding='utf-8') as f:
    #     json.dump(overview_data, f, cls=OverviewEncoder)

    # use vnpy save json
    save_json(filename, overview_data, cls=OverviewEncoder, mode='w')


def load_overview(filename: str) -> Dict[str, Dict]:
    # if not os.path.exists(file_path):
    #     return {}
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     dic = json.load(f)
    #     for k, v in dic.items():
    #         dic[k] = cls(**v)
    #     return dic

    # use vnpy load json
    return load_json(filename=filename, cls=OverviewDecoder)


def update_bar_overview(symbol: str,
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
        count = len(bars)
