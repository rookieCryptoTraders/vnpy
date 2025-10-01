# -*- coding=utf-8 -*-
# @Project  : crypto_backtrader
# @FilePath : adapters
# @File     : dbs.py
# @Time     : 2024/3/13 13:42
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: 用于定义功能性的函数，即实现原子化的操作，比如创建db、读取db文件，写入db文件等。

import datetime
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import *
import pandas as pd
import polars as pl

from vnpy.utils.datetimes import TimeFreq
from .base import BaseDataAdapter, DataSource, BaseSchema


#
# class DbOverview(object):
#     """
#     Overview of bar data stored in database.
#     """
#
#     symbol: str = ""
#     exchange: Exchange = None
#     interval: Interval = None
#     count: int = 0
#     start: float = None
#     end: float = None
#
#     def __init__(self, symbol: str,
#                  exchange: Exchange,
#                  interval: Interval,
#                  start: float,
#                  end: float,
#                  count: int = 0, **kwargs):
#         self.symbol = symbol
#         if isinstance(exchange, Exchange):
#             self.exchange = exchange
#         elif isinstance(exchange, str):
#             self.exchange = Exchange(exchange)
#         else:
#             raise NotImplementedError()
#         if isinstance(interval, Interval):
#             self.interval = interval
#         elif isinstance(interval, str):
#             self.interval = Interval(interval)
#         else:
#             raise NotImplementedError()
#         self.start = start
#         self.end = end
#         self.count = count
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#
#     @abstractmethod
#     def to_vnpy_overview(self):
#         pass
#
#
# class DbBarOverview(DbOverview):
#     """
#     BarOverview of bar data stored in database.
#     """
#
#     symbol: str = ""
#     exchange: Exchange = None
#     interval: Interval = None
#     count: int = 0
#     start: float = None
#     end: float = None
#
#     def __init__(self, symbol: str,
#                  exchange: Exchange,
#                  interval: Interval,
#                  start: float,
#                  end: float,
#                  count: int = 0):
#         super().__init__(symbol=symbol, exchange=exchange, interval=interval, start=start, end=end, count=count)
#
#     def to_vnpy_overview(self):
#         return BarOverview(
#             symbol=self.symbol,
#             exchange=self.exchange,
#             interval=self.interval,
#             start=unix2datetime(self.start),
#             end=unix2datetime(self.end),
#             count=self.count,
#         )
#
#
# class DbFactorOverview(DbOverview):
#     """
#     BarOverview of bar data stored in database.
#     """
#
#     symbol: str = ""
#     name: str = ""
#     exchange: Exchange = None
#     interval: Interval = None
#     count: int = 0
#     start: float = None
#     end: float = None
#
#     def __init__(self, symbol: str,
#                  name: str,
#                  exchange: Exchange,
#                  interval: Interval,
#                  start: float,
#                  end: float,
#                  count: int = 0):
#         super().__init__(symbol=symbol, name=name, exchange=exchange, interval=interval, start=start, end=end,
#                          count=count)
#
#     def to_vnpy_overview(self):
#         return FactorOverview(
#             symbol=self.symbol,
#             name=self.name,
#             exchange=self.exchange,
#             interval=self.interval,
#             start=unix2datetime(self.start),
#             end=unix2datetime(self.end),
#             count=self.count,
#         )
#
#

class DBController(ABC):
    """
    用于数据库的初始化、创建、删除、alter table等具体的原子操作
    """
    database = None
    __conn__ = None  # 需要通过connect函数去初始化

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def connect(self, database='', *args, **kwargs):
        pass

    @abstractmethod
    def commit(self, *args, **kwargs):
        pass

    @abstractmethod
    def close(self, *args, **kwargs):
        """close connection"""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def create_table(self, *args, **kwargs):
        raise NotImplementedError("未来该函数将作为完全参数化的api")

    # @abstractmethod
    # def create_table_symbol(self, table_name='symbols', db_path=None):
    #     pass
    #
    # @abstractmethod
    # def create_table_kline(self, freq: Union[str, TimeFreq] = TimeFreq.unknown, db_path=None):
    #     pass
    #
    # @abstractmethod
    # def create_table_factor(self, factor_name: str, freq: Union[str, TimeFreq], db_path=None):
    #     pass
    #
    # @abstractmethod
    # def insert_or_get_symbol(self, symbol_name, db_path=None) -> int:
    #     pass
    #
    # @abstractmethod
    # def insert_kline(self, candle, freq: str, db_path=None):
    #     pass


class BaseDBAdapter(BaseDataAdapter):
    """base database adapter. so it has creating table, altering table, dropping table, selecting data, inserting data, etc.

    Parameters
    ----------
    table_prefix : str
        used to distinguish different tables, e.g., `bar_` or `factor_`

    """

    def __init__(self, from_: Optional[DataSource] = None, to_: Optional[DataSource] = None,
                 table_prefix: str = "",
                 *args, **kwargs):
        super().__init__(from_=from_, to_=to_)
        self.set_tz('UTC')
        self.table_prefix = table_prefix

    def set_tz(self, tz='UTC'):
        os.environ['TZ'] = tz


class KlineDBAdapter(BaseDBAdapter):
    """database adapter. so it has creating table, altering table, dropping table, selecting data, inserting data, etc."""
    __col_mapper__ = {'open': 'open_', 'close': 'close_'}  # 主要为了解决sqlite中的关键字问题
    __col_mapper_reversed__ = {v: k for k, v in __col_mapper__.items()}

    def __init__(self, from_: Optional[DataSource] = None, to_: Optional[DataSource] = None,
                 table_prefix: str = "",
                 *args, **kwargs):
        super().__init__(from_=from_, to_=to_, table_prefix=table_prefix)
        self.set_tz('UTC')

    def set_tz(self, tz='UTC'):
        os.environ['TZ'] = tz

    @abstractmethod
    def create_table(self, freq, *args, **kwargs):
        pass

    @abstractmethod
    def alter_table(self, freq, column_name, column_type, comment='', *args, **kwargs):
        pass

    @abstractmethod
    def drop_table(self, freq, *args, **kwargs):
        pass

    @abstractmethod
    def select(self, *args, **kwargs):
        pass

    @abstractmethod
    def insert(self, data, freq, *args, **kwargs):
        pass

    @abstractmethod
    def check_schema(self, freq: str, exp_schema: BaseSchema) -> Optional[dict[str, str]]:
        """
        Check if the table schema matches the given schema.
        Because comment is not supposed to appear in quant trading databases, so we ignore it.

        Parameters
        ----------
        freq : str
            The freq of the table to check.
        exp_schema : BaseSchema
            The schema to compare against.

        Returns
        -------
        Optional[dict[str, str]]
            A dictionary of column names and their types if the schema matches, otherwise None.

        """
        pass


class FactorDBAdapter(BaseDBAdapter):
    """database adapter. so it has creating table, altering table, dropping table, selecting data, inserting data, etc."""

    __primary_cols__ = None

    def __init__(self, from_: Optional[DataSource] = None, to_: Optional[DataSource] = None,
                 table_prefix: str = "",
                 *args, **kwargs):
        super().__init__(from_=from_, to_=to_, table_prefix=table_prefix)
        self.set_tz('UTC')

    def set_tz(self, tz='UTC'):
        os.environ['TZ'] = tz

    @abstractmethod
    def create_table(self, freq, *args, **kwargs):
        pass

    @abstractmethod
    def alter_table(self, freq, column_name, column_type, comment='', *args, **kwargs):
        pass

    @abstractmethod
    def drop_table(self, freq, *args, **kwargs):
        pass

    @abstractmethod
    def select(self, *args, **kwargs):
        pass

    @abstractmethod
    def insert(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_schema(self, freq: str, exp_schema: BaseSchema) -> Optional[dict[str, str]]:
        """
        Check if the table schema matches the given schema.
        Because comment is not supposed to appear in quant trading databases, so we ignore it.

        Parameters
        ----------
        freq : str
            The freq of the table to check.
        exp_schema : BaseSchema
            The schema to compare against.

        Returns
        -------
        Optional[dict[str, str]]
            A dictionary of column names and their types if the schema matches, otherwise None.

        """
        pass


# ====================================================
# binance schema definition (without schema type)
# ====================================================
class BinanceKlineSchema(BaseSchema):
    """
    binance data has similar structure, but different databases have different datatypes. so we need to decouple them.
    """

    def __init__(self, default_schema_type: Dict[str, str] = None, additional_columns: Dict[str, str] = None):
        super().__init__()
        self.datetime: str = ""
        self.ticker: str = ""
        self.open: str = ""
        self.high: str = ""
        self.low: str = ""
        self.close: str = ""
        self.volume: str = ""
        self.quote_asset_volume: str = ""
        self.number_of_trades: str = ""
        self.taker_buy_base_asset_volume: str = ""
        self.taker_buy_quote_asset_volume: str = ""
        additional_columns = {} if not additional_columns else additional_columns
        default_schema_type.update(additional_columns)
        self.assign_schema_type(schema_type=default_schema_type)


class BinanceFactorSchema(BaseSchema):
    """
    binance data has similar structure, but different databases have different datatypes. so we need to decouple them.
    """

    def __init__(self, default_schema_type: Dict[str, str] = None, additional_columns: Dict[str, str] = None):
        super().__init__()
        self.datetime: str = ""
        self.ticker: str = ""
        additional_columns = {} if not additional_columns else additional_columns
        default_schema_type.update(additional_columns)
        self.assign_schema_type(schema_type=default_schema_type)
