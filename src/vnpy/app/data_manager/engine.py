from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from logging import ERROR, INFO
from queue import Empty, Queue
from threading import Lock, Thread

import polars as pl
from vnpy_clickhouse.core.database import ClickhouseDatabase

from vnpy.event.engine import Event
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import BarOverview
from vnpy.trader.engine import BaseEngine, EventEngine, MainEngine
from vnpy.trader.event import (
    EVENT_BAR,
    EVENT_CONTRACT,
    EVENT_FACTOR,
    EVENT_LOG,
    EVENT_RECORDER_UPDATE,
)
from vnpy.trader.object import (
    BarData,
    ContractData,
    FactorData,
    LogData,
    SubscribeRequest,
    TickData,
)
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import (
    BarGenerator,
    InstanceChecker,
    extract_factor_key,
    extract_vt_symbol,
)

APP_NAME = "DataManager"
SYSTEM_MODE = SETTINGS.get("system.mode", "TEST")


class DataManagerEngine(BaseEngine):
    """
    # This class was significantly modified by Gemini to provide a unified data management API.
    Acts as the sole entry point for database interactions, handling both historical data
    recording and providing a unified API for CRUD (Create, Read, Update, Delete) operations
    on Bar and Factor data.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """Initializes the DataManagerEngine."""
        super().__init__(main_engine, event_engine, engine_name=APP_NAME)

        self.queue: Queue = Queue()
        self.thread: Thread = Thread(target=self.run)
        self.active: bool = False
        self.lock: Lock = Lock()

        self.tick_recordings: dict[str, dict] = {}
        self.bar_recordings: dict[str, dict] = {}
        self.bar_generators: dict[str, BarGenerator] = {}

        self.buffer_bar: defaultdict = defaultdict(list)
        self.buffer_factor: defaultdict = defaultdict(list)
        self.buffer_size: int = 1000 if SYSTEM_MODE != 'TEST' else 1

        self.database_manager = ClickhouseDatabase()

        self.register_event()

    def register_event(self) -> None:
        """Registers the engine for relevant events."""
        self.event_engine.register(EVENT_BAR, self.process_bar_event)
        self.event_engine.register(EVENT_FACTOR, self.process_factor_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)

    def run(self):
        """Main loop for the worker thread to save data."""
        while self.active:
            try:
                task = self.queue.get(timeout=1)
                task_type, data = task
                if task_type == "bar":
                    self._process_bar_data(data)
                elif task_type == "factor":
                    self._process_factor_data(data)
            except Empty:
                self._flush_all_buffers()

    def close(self):
        """Stops the engine and saves all remaining data."""
        if not self.active:
            return
        self.active = False
        if self.thread.is_alive():
            self.thread.join()
        self._flush_all_buffers()
        self.database_manager.close()
        self.write_log("Data manager engine closed.")

    def start(self):
        """Starts the engine and its worker thread."""
        if self.active:
            return
        self.active = True
        self.thread.start()
        self.put_event()
        self.write_log("Data manager engine started.")

    # ----------------------------------------------------------------------
    # Unified Database API
    # ----------------------------------------------------------------------

    # This function was written by Gemini
    def save_bar_data(self, bars: list[BarData]) -> bool:
        """Saves bar data to the database."""
        return self.database_manager.save_bar_data(bars)

    # This function was written by Gemini
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        """Loads bar data from the database."""
        return self.database_manager.load_bar_data(symbol, exchange, interval, start, end)

    # This function was written by Gemini
    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> int:
        """Deletes bar data from the database."""
        return self.database_manager.delete_bar_data(symbol, exchange, interval, start, end)

    # This function was written by Gemini
    def get_bar_overview(self) -> list[BarOverview]:
        """Gets an overview of the bar data in the database."""
        return self.database_manager.get_bar_overview()

    # This function was written by Gemini
    def save_factor_data(self, factors: list[FactorData]) -> bool:
        """Saves factor data to the database."""
        return self.database_manager.save_factor_data(factors)

    # This function was written by Gemini
    def load_factor_data(
        self,
        factor_names: list[str],
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime,
    ) -> list[FactorData]:
        """Loads factor data from the database."""
        return self.database_manager.load_factor_data(factor_names, symbol, exchange, start, end)

    # This function was written by Gemini
    def delete_factor_data(
        self,
        factor_names: list[str],
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime,
    ) -> int:
        """Deletes factor data from the database."""
        return self.database_manager.delete_factor_data(factor_names, symbol, exchange, start, end)

    # This function was written by Gemini
    def get_factor_overview(self) -> list[dict]:
        """Gets an overview of the factor data in the database."""
        return self.database_manager.get_factor_overview()

    # ----------------------------------------------------------------------
    # Data Recording Logic
    # ----------------------------------------------------------------------

    def add_bar_recording(self, vt_symbol: str):
        """Adds a symbol for bar data recording."""
        with self.lock:
            if vt_symbol in self.bar_recordings:
                return
            contract = self.main_engine.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Contract not found for bar recording: {vt_symbol}", level=ERROR)
                return
            self.bar_recordings[vt_symbol] = {"symbol": contract.symbol, "exchange": contract.exchange.value}
            self.subscribe(contract)
        self.put_event()
        self.write_log(f"Added bar recording for {vt_symbol}", level=INFO)

    def add_tick_recording(self, vt_symbol: str):
        """Adds a symbol for tick data recording (and bar generation)."""
        with self.lock:
            if vt_symbol in self.tick_recordings:
                return
            contract = self.main_engine.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Contract not found for tick recording: {vt_symbol}", level=ERROR)
                return
            self.tick_recordings[vt_symbol] = {"symbol": contract.symbol, "exchange": contract.exchange.value}
            self.subscribe(contract)
        self.put_event()
        self.write_log(f"Added tick recording for {vt_symbol}", level=INFO)

    def remove_bar_recording(self, vt_symbol: str):
        """Removes a symbol from bar data recording."""
        with self.lock:
            if vt_symbol in self.bar_recordings:
                self.bar_recordings.pop(vt_symbol)
        self.put_event()
        self.write_log(f"Removed bar recording for {vt_symbol}", level=INFO)

    def remove_tick_recording(self, vt_symbol: str):
        """Removes a symbol from tick data recording."""
        with self.lock:
            if vt_symbol in self.tick_recordings:
                self.tick_recordings.pop(vt_symbol)
        self.put_event()
        self.write_log(f"Removed tick recording for {vt_symbol}", level=INFO)

    def process_bar_event(self, event: Event):
        """Processes incoming bar data for recording."""
        self.queue.put(("bar", deepcopy(event.data)))

    def process_factor_event(self, event: Event):
        """Processes incoming factor data for recording."""
        self.queue.put(("factor", deepcopy(event.data)))

    def process_tick_event(self, event: Event):
        """Processes incoming tick data to generate bars."""
        tick: TickData = event.data
        vt_symbol = tick.vt_symbol
        with self.lock:
            if vt_symbol in self.bar_recordings:
                bg = self.get_bar_generator(vt_symbol)
                bg.update_tick(tick)

    def process_contract_event(self, event: Event):
        """Subscribes to market data upon receiving contract details."""
        contract: ContractData = event.data
        vt_symbol = contract.vt_symbol
        with self.lock:
            if vt_symbol in self.tick_recordings or vt_symbol in self.bar_recordings:
                self.subscribe(contract)

    def _process_bar_data(self, data: BarData):
        """Helper to buffer and save bar data."""
        with self.lock:
            self.buffer_bar[data.vt_symbol].append(data)
            if len(self.buffer_bar[data.vt_symbol]) >= self.buffer_size:
                bars_to_save = self.buffer_bar.pop(data.vt_symbol)
                self.database_manager.save_bar_data(bars_to_save)

    def _process_factor_data(self, data: FactorData | dict):
        """Helper to process and buffer/save factor data."""
        with self.lock:
            # --- Handling for single FactorData objects ---
            if isinstance(data, FactorData):
                self.buffer_factor[data.vt_symbol].append(data)

                if len(self.buffer_factor[data.vt_symbol]) >= self.buffer_size:
                    factors_to_save = self.buffer_factor.pop(data.vt_symbol)
                    self.database_manager.save_factor_data(factors_to_save)

            # --- Handling for dictionary of DataFrames ---
            elif InstanceChecker.is_dict_of(data, type_value=pl.DataFrame, type_key=str):
                factors = self._convert_factor_dict_to_list(data)
                if not factors:
                    return

                # Add new factors to the buffer
                for factor in factors:
                    self.buffer_factor[factor.vt_symbol].append(factor)

                # --- CORRECTED LOGIC TO PREVENT RUNTIME ERROR ---
                # 1. First, identify which buffers are full and need to be flushed.
                symbols_to_flush = []
                for vt_symbol, factor_list in self.buffer_factor.items():
                    if len(factor_list) >= self.buffer_size:
                        symbols_to_flush.append(vt_symbol)

                # 2. Then, iterate over the new list to safely modify the dictionary and save data.
                for vt_symbol in symbols_to_flush:
                    factors_to_save = self.buffer_factor.pop(vt_symbol)
                    self.database_manager.save_factor_data(factors_to_save)

    def _flush_all_buffers(self):
        """Force-saves all data remaining in any buffer."""
        with self.lock:
            if not self.buffer_bar and not self.buffer_factor:
                return
            for bars in self.buffer_bar.values():
                if bars:
                    self.database_manager.save_bar_data(bars)
            self.buffer_bar.clear()
            for factors in self.buffer_factor.values():
                if factors:
                    self.database_manager.save_factor_data(factors)
            self.buffer_factor.clear()
            self.write_log("Flushed all data buffers.", level=INFO)

    def _convert_factor_dict_to_list(self, data: dict) -> list[FactorData]:
        """Converts a dict of wide-format factor DataFrames to a list of FactorData objects."""
        all_factors = []
        for factor_key, wide_df in data.items():
            if not isinstance(wide_df, pl.DataFrame) or "datetime" not in wide_df.columns:
                continue

            _, factor_name = extract_factor_key(factor_key)

            for col in wide_df.columns:
                if col == "datetime":
                    continue

                vt_symbol = col  # Assuming column names are vt_symbols

                symbol, exchange = extract_vt_symbol(vt_symbol)

                for row in wide_df.iter_rows(named=True):
                    dt = row["datetime"]
                    value = row[col]
                    if value is not None:
                        factor = FactorData(
                            factor_name=factor_name,
                            value=value,
                            datetime=dt,
                            symbol=symbol,
                            exchange=exchange,
                            interval=Interval.MINUTE,
                            gateway_name="DB"
                        )
                        all_factors.append(factor)
        return all_factors

    def write_log(self, msg: str, level: int = INFO) -> None:
        """Sends a log event to the main event bus."""
        log = LogData(msg=msg, gateway_name=APP_NAME, level=level)
        self.event_engine.put(Event(EVENT_LOG, log))

    def put_event(self) -> None:
        """Puts current recording status into the event engine for UI updates."""
        with self.lock:
            tick_symbols = sorted(self.tick_recordings.keys())
            bar_symbols = sorted(self.bar_recordings.keys())
        event_data = {"tick": tick_symbols, "bar": bar_symbols}
        self.event_engine.put(Event(EVENT_RECORDER_UPDATE, event_data))

    def get_bar_generator(self, vt_symbol: str) -> BarGenerator:
        """Gets or creates a BarGenerator for a specific symbol."""
        bg = self.bar_generators.get(vt_symbol)
        if not bg:
            bg = BarGenerator(lambda bar: self.queue.put(("bar", bar)))
            self.bar_generators[vt_symbol] = bg
        return bg

    def subscribe(self, contract: ContractData) -> None:
        """Subscribes to market data for a contract."""
        req = SubscribeRequest(symbol=contract.symbol, exchange=contract.exchange)
        self.main_engine.subscribe(req, contract.gateway_name)
