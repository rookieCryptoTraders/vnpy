""""""
from collections import defaultdict
from threading import Thread
from queue import Queue, Empty
from copy import copy
from typing import Literal
from logging import Logger, CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET

import polars as pl

from vnpy.event import Event, EventEngine
from vnpy.event.events import EVENT_RECORDER_UPDATE, EVENT_RECORDER_EXCEPTION, EVENT_RECORDER_RECORD
from vnpy.trader.engine import BaseEngine, MainEngine, OmsEngine
from vnpy.trader.object import (
    SubscribeRequest,
    TickData,
    BarData,
    FactorData,
    ContractData,
    LogData
)
from vnpy.trader.event import EVENT_TICK, EVENT_CONTRACT, EVENT_BAR
from vnpy.trader.utility import load_json, save_json, BarGenerator

from vnpy_clickhouse.clickhouse_database import ClickhouseDatabase

APP_NAME = "DataRecorder"


class RecorderEngine(BaseEngine):
    """用于将从binance等exchange拿到的数据入库, 不负责计算因子"""

    def __init__(self,
                 main_engine: MainEngine,
                 event_engine: EventEngine):
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

        self.queue = Queue()
        self.queue_pylist = []
        self.thread = Thread(target=self.run)
        self.active = False

        self.tick_recordings = {}  # list of symbols to record tick data
        self.bar_recordings = {}  # list of symbols to record bar data
        self.bar_generators = {}

        # self.load_setting()
        self.register_event()
        self.start()
        self.put_event()

        # 用clickhouse数据库
        self.database_manager = ClickhouseDatabase()
        self.buffer_bar = defaultdict(list)
        self.buffer_factor = defaultdict(list)
        self.buffer_size = 4  # todo: 调大该数字

    # def load_setting(self):
    #     """"""
    #     setting = load_json(self.setting_filename)
    #     self.tick_recordings = setting.get("tick", {})
    #     self.bar_recordings = setting.get("bar", {})

    # def save_setting(self):
    #     """"""
    #     setting = {
    #         "tick": self.tick_recordings,
    #         "bar": self.bar_recordings
    #     }
    #     save_json(self.setting_filename, setting)

    def save_data(self,
                  task_type: Literal["tick", "bar", "factor", None] = None,
                  data=None,
                  force_save: bool = False):
        """The actual implementation of the core functions that put the data into the database
        
        Parameters
        ----------
        task_type :
        data :
        force_save : bool
            Ignore buffer_size and force all current data to be saved

        Returns
        -------

        """
        if task_type == "tick":
            self.database_manager.save_tick_data([data])
        elif task_type == "bar":
            assert isinstance(data, BarData)
            self.buffer_bar[data.vt_symbol].append(data)
            # if data.volume < 1000:
            #     print(f"data_recorder.RecorderEngine.{self.save_data.__name__}: {data.__dict__}")
            #     raise ValueError(f"data_recorder.RecorderEngine.save_data: {data.__dict__}")
            to_remove = []  # 保存完数据后, 将其从buffer中删除
            for k, v in self.buffer_bar.items():
                if len(v) >= self.buffer_size or force_save:
                    to_remove.append(k)
                    self.database_manager.save_bar_data(v)
            for k in to_remove:
                self.buffer_bar[k] = []
        elif task_type == 'factor':
            if isinstance(data, FactorData):
                self.buffer_factor[data.vt_symbol].append(data)  # todo: 这里用vt_symbol可以吗???
                to_remove = []
                for k, v in self.buffer_factor.items():
                    if len(v) >= self.buffer_size or force_save:
                        to_remove.append(k)
                        self.database_manager.save_factor_data(name=data.factor_name,
                                                               data=v)
                for k in to_remove:
                    self.buffer_factor[k] = []
            elif isinstance(data, pl.DataFrame):
                self.database_manager.save_factor_data(name=data.columns[-1], data=data)
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")

        elif task_type is None and data is None:
            # 强制保存当前所有的数据
            for k, v in self.buffer_bar.items():
                if len(v) == 0: continue
                self.database_manager.save_bar_data(v)
            for k, v in self.buffer_factor.items():
                if len(v) == 0: continue
                self.database_manager.save_factor_data(name=v[0].factor_name, data=v)
            self.buffer_bar = defaultdict(list)
            self.buffer_factor = defaultdict(list)

    def run(self):
        """"""
        while self.active:
            try:
                task = self.queue.get(timeout=1)
                task_type, data = task
                self.save_data(task_type, data)

            except Empty:
                continue

    def close(self):
        """"""
        self.active = False

        self.save_data(None, None)  # 保存所有buffer中残留的数据

        if self.thread.isAlive():
            self.thread.join()

    def start(self):
        """"""
        self.write_log("启动数据拉取引擎")
        self.active = True
        self.thread.start()

    def add_bar_recording(self, vt_symbol: str):
        """add a symbol to the bar recording list, which means that the bar data of this symbol will be recorded"""
        if vt_symbol in self.bar_recordings:
            self.write_log(f"已在K线记录列表中：{vt_symbol}", level=NOTSET)
            return

        contract = self.main_engine.get_contract(vt_symbol)
        if not contract:
            self.write_log(f"找不到合约：{vt_symbol}", level=ERROR)
            return

        self.bar_recordings[vt_symbol] = {
            "symbol": contract.symbol,
            "exchange": contract.exchange.value,
            "gateway_name": contract.gateway_name
        }

        self.subscribe(contract)
        # self.save_setting()
        self.put_event()

        self.write_log(f"添加K线记录成功：{vt_symbol}", level=DEBUG)

    def add_tick_recording(self, vt_symbol: str):
        """add a symbol to the tick recording list, which means that the tick data of this symbol will be recorded"""
        if vt_symbol in self.tick_recordings:
            self.write_log(f"已在Tick记录列表中：{vt_symbol}", level=NOTSET)
            return

        contract = self.main_engine.get_contract(vt_symbol)
        if not contract:
            self.write_log(f"找不到合约：{vt_symbol}", level=ERROR)
            return

        self.tick_recordings[vt_symbol] = {
            "symbol": contract.symbol,
            "exchange": contract.exchange.value,
            "gateway_name": contract.gateway_name
        }

        self.subscribe(contract)
        # self.save_setting()
        self.put_event()

        self.write_log(f"添加Tick记录成功：{vt_symbol}", level=DEBUG)

    def remove_bar_recording(self, vt_symbol: str):
        """remove a symbol from the bar recording list"""
        if vt_symbol not in self.bar_recordings:
            self.write_log(f"不在K线记录列表中：{vt_symbol}", level=DEBUG)
            return

        self.bar_recordings.pop(vt_symbol)
        # self.save_setting()
        self.put_event()

        self.write_log(f"移除K线记录成功：{vt_symbol}")

    def remove_tick_recording(self, vt_symbol: str):
        """remove a symbol from the tick recording list"""
        if vt_symbol not in self.tick_recordings:
            self.write_log(f"不在Tick记录列表中：{vt_symbol}", level=DEBUG)
            return

        self.tick_recordings.pop(vt_symbol)
        # self.save_setting()
        self.put_event()

        self.write_log(f"移除Tick记录成功：{vt_symbol}")

    def register_event(self):
        """"""
        # self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_BAR, self.process_bar_event)
        # self.event_engine.register(EVENT_FACTOR, self.process_factor_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)

        # self.main_engine.register_log_event(EVENT_RECORDER_LOG)

    def process_bar_event(self, event: Event):
        """"""
        bar = event.data
        self.add_bar_recording(vt_symbol=bar.vt_symbol)
        if bar.vt_symbol in self.bar_recordings:
            self.record_bar(bar)

    def process_tick_event(self, event: Event):
        """"""
        tick = event.data

        if tick.vt_symbol in self.tick_recordings:
            self.record_tick(tick)

        if tick.vt_symbol in self.bar_recordings:
            bg = self.get_bar_generator(tick.vt_symbol)
            bg.update_tick(tick)

    def process_contract_event(self, event: Event):
        """"""
        contract = event.data
        vt_symbol = contract.vt_symbol

        if (vt_symbol in self.tick_recordings or vt_symbol in self.bar_recordings):
            self.subscribe(contract)

    def write_log(self, msg: str, level=INFO) -> None:
        """输出日志"""
        self.main_engine.write_log(msg, source=APP_NAME, level=level)

    def put_event(self):
        """"""
        tick_symbols = list(self.tick_recordings.keys())
        tick_symbols.sort()

        bar_symbols = list(self.bar_recordings.keys())
        bar_symbols.sort()

        data = {
            "tick": tick_symbols,
            "bar": bar_symbols
        }

        event = Event(
            EVENT_RECORDER_UPDATE,
            data
        )
        self.event_engine.put(event)

    def record_tick(self, tick: TickData):
        """"""
        task = ("tick", copy(tick))
        self.queue.put(task)

    def record_bar(self, bar: BarData):
        """"""
        task = ("bar", copy(bar))
        self.queue.put(task)

    def get_bar_generator(self, vt_symbol: str):
        """"""
        bg = self.bar_generators.get(vt_symbol, None)

        if not bg:
            bg = BarGenerator(self.record_bar)
            self.bar_generators[vt_symbol] = bg

        return bg

    def subscribe(self, contract: ContractData):
        """"""
        req = SubscribeRequest(
            symbol=contract.symbol,
            exchange=contract.exchange
        )
        self.main_engine.subscribe(req, contract.gateway_name)
