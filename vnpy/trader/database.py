from abc import ABC, abstractmethod
from datetime import datetime
from types import ModuleType
from typing import List, TypeVar
from dataclasses import dataclass, field
from importlib import import_module

from .constant import Interval, Exchange
from .object import BarData, TickData
from .setting import SETTINGS
from .utility import ZoneInfo
from .locale import _

from vnpy.config import VTSYMBOL_KLINE, VTSYMBOL_FACTORDATA

DB_TZ = ZoneInfo(SETTINGS["database.timezone"])


def convert_tz(dt: datetime) -> datetime:
    """
    Convert timezone of datetime object to DB_TZ.
    """
    dt: datetime = dt.astimezone(DB_TZ)
    return dt.replace(tzinfo=None)


@dataclass
class BaseOverview:
    symbol: str = ""
    exchange: Exchange = None
    interval: Interval = None
    count: int = 0
    start: datetime = None
    end: datetime = None

    vt_symbol: str = ""
    VTSYMBOL_TEMPLATE: str = field(default=None, init=False)


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
    def get_bar_overview(self) -> List[BarOverview]:
        """
        Return bar data avaible in database.
        """
        pass

    @abstractmethod
    def get_tick_overview(self) -> List[TickOverview]:
        """
        Return tick data avaible in database.
        """
        pass


database: BaseDatabase = None
TV_BaseOverview = TypeVar('TV_BaseOverview', bound=BaseOverview)  # TV means TypeVar


def get_database() -> BaseDatabase:
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
        print(_("找不到数据库驱动{}，使用默认的SQLite数据库").format(module_name))
        module: ModuleType = import_module("vnpy_sqlite")

    # Create database object from module
    database = module.Database()
    return database
