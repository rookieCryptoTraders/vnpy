import importlib
import sys
import traceback
from collections import defaultdict
from copy import deepcopy
from logging import ERROR, INFO, NOTSET
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Literal

import polars as pl
from vnpy_clickhouse.clickhouse_database import ClickhouseDatabase

from vnpy.event.engine import Event
from vnpy.trader.constant import Exchange, Interval

from vnpy.trader.engine import BaseEngine, EventEngine, MainEngine
from vnpy.trader.event import (
    EVENT_BAR,
    EVENT_CONTRACT,
    EVENT_FACTOR,
    EVENT_LOG,
    EVENT_RECORDER_UPDATE, EVENT_BAR_FILLING, EVENT_FACTOR_FILLING
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
from vnpy.trader.utility import BarGenerator, extract_factor_key, InstanceChecker

APP_NAME = "DataRecorder"
SYSTEM_MODE = SETTINGS.get("system.mode", "LIVE")


class RecorderEngine(BaseEngine):
    """
    For storing tick, bar, and factor data from market data streams into a database.
    This engine is designed to be thread-safe and handles data in batches for efficiency.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """Initializes the RecorderEngine."""
        super().__init__(main_engine, event_engine, engine_name=APP_NAME)

        self.queue: Queue = Queue()
        self.thread: Thread = Thread(target=self.run)
        self.active: bool = False
        self.lock: Lock = Lock()  # Lock for thread-safe access to shared resources

        # Dictionaries to track which symbols are being recorded
        self.tick_recordings: dict[str, dict] = {}
        self.bar_recordings: dict[str, dict] = {}
        self.factor_recordings: dict[str, dict] = {}

        # Bar generators for creating bars from ticks
        self.bar_generators: dict[str, BarGenerator] = {}

        self.register_event()

        # --- Database Settings ---
        # Buffers for batch database writes
        self.buffer_bar: defaultdict = defaultdict(list)
        self.buffer_factor: defaultdict = defaultdict(list)
        self.buffer_size: int = 1000 if SYSTEM_MODE != 'TEST' else 1  # Number of records to buffer before writing

        # Database manager instance
        self.database_manager = ClickhouseDatabase(event_engine=event_engine)

        # Overview handler for data consistency checks (optional)
        # self.overview_handler_for_result_check = OverviewHandler()

    def update_schema(
            self,
            database_name: str,
            exchanges: list[Exchange],
            intervals: Interval | list[Interval],
            factor_keys: list[str] | None = None,
    ):
        """Dynamically updates the database schema based on external info."""
        # This function remains as is, assuming it works in your environment.
        database_name = database_name.lower()
        try:
            # Try importing the submodule first (for namespace packages)
            module = importlib.import_module(f"vnpy_{database_name}.vnpy_{database_name}")
        except ImportError:
            # Fallback to top-level module
            module = importlib.import_module(f"vnpy_{database_name}")
        convertor = module.outer_str2console_str
        dtype_mapper = module.DTYPE_MAPPER
        database_name_camelcase = database_name.capitalize()

        for exchange in exchanges:
            exchange_val = exchange.value
            exchange_camelcase = exchange_val.capitalize()
            factor_schema_class = getattr(
                module, f"{database_name_camelcase}{exchange_camelcase}FactorSchema"
            )
            factor_keys_dict = (
                {convertor(key): dtype_mapper["float64"] for key in factor_keys}
                if factor_keys
                else {}
            )
            schema_factor = factor_schema_class(additional_columns=factor_keys_dict)
            self.database_manager.update_all_schema(
                intervals=intervals, schema_factor=schema_factor
            )

    def register_event(self) -> None:
        """Registers the engine for relevant events."""
        self.event_engine.register(EVENT_BAR, self.process_bar_event)
        self.event_engine.register(EVENT_FACTOR, self.process_factor_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)
        self.event_engine.register(EVENT_BAR_FILLING, self.process_bar_filling_event)
        self.event_engine.register(EVENT_FACTOR_FILLING, self.process_factor_filling_event)

    def run(self):
        """Main loop for the worker thread."""
        task = None
        while self.active:
            try:
                # Get a task from the queue, with a timeout to allow periodic buffer checks
                task = self.queue.get(timeout=1)
                task_type, data, event_type = task
                self.save_data(task_type=task_type.replace("_flush", ""), data=data, force_save="_flush" in task_type,
                               event_type=event_type)
            except Empty:
                # If the queue is empty, flush any partially filled buffers
                self.write_log(f"the queue is empty, flush any partially filled buffers", level=NOTSET)
                # self.save_data(force_save=True)
            except Exception as e:
                self.write_log(f"Error in recorder worker thread: {e}", level=ERROR)
                self.write_log(f"Error event data: {task}", level=ERROR)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)

    def close(self):
        """Stops the engine and saves all remaining data."""
        if not self.active:
            return

        self.write_log("Closing data recorder engine.")
        self.active = False
        if self.thread.is_alive():
            self.thread.join()

        # Final save of any data left in buffers after the thread stops
        self._flush_all_buffers()
        self.write_log("Data recorder engine closed.")

    def start(self):
        """Starts the engine and its worker thread."""
        if self.active:
            return

        self.write_log("Starting data recorder engine.")
        self.active = True
        self.thread.start()
        self.put_event()

    # ----------------------------------------------------------------------
    # Public API for adding/removing recordings
    # ----------------------------------------------------------------------

    def add_bar_recording(self, vt_symbol: str):
        """Adds a symbol for bar data recording."""
        with self.lock:
            if vt_symbol in self.bar_recordings:
                self.write_log(f"Already in K-line recording list: {vt_symbol}", level=NOTSET)
                return

            contract = self.main_engine.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Cannot find contract for bar recording: {vt_symbol}", level=ERROR)
                return

            self.bar_recordings[vt_symbol] = {
                "symbol": contract.symbol,
                "exchange": contract.exchange.value,
                "gateway_name": contract.gateway_name,
            }
            self.subscribe(contract)

        self.put_event()
        self.write_log(f"Added K-line recording: {vt_symbol}", level=INFO)

    def add_tick_recording(self, vt_symbol: str):
        """Adds a symbol for tick data recording."""
        with self.lock:
            if vt_symbol in self.tick_recordings:
                self.write_log(f"Already in Tick recording list: {vt_symbol}", level=NOTSET)
                return

            contract = self.main_engine.get_contract(vt_symbol)
            if not contract:
                self.write_log(f"Cannot find contract for tick recording: {vt_symbol}", level=ERROR)
                return

            self.tick_recordings[vt_symbol] = {
                "symbol": contract.symbol,
                "exchange": contract.exchange.value,
                "gateway_name": contract.gateway_name,
            }
            self.subscribe(contract)

        self.put_event()
        self.write_log(f"Added Tick recording: {vt_symbol}", level=INFO)

    def remove_bar_recording(self, vt_symbol: str):
        """Removes a symbol from bar data recording."""
        with self.lock:
            if vt_symbol not in self.bar_recordings:
                return
            self.bar_recordings.pop(vt_symbol)

        self.put_event()
        self.write_log(f"Removed K-line recording: {vt_symbol}", level=INFO)

    def remove_tick_recording(self, vt_symbol: str):
        """Removes a symbol from tick data recording."""
        with self.lock:
            if vt_symbol not in self.tick_recordings:
                return
            self.tick_recordings.pop(vt_symbol)

        self.put_event()
        self.write_log(f"Removed Tick recording: {vt_symbol}", level=INFO)

    # ----------------------------------------------------------------------
    # Event processing methods (called by EventEngine)
    # ----------------------------------------------------------------------

    def process_bar_event(self, event: Event):
        """Processes incoming bar data."""
        self.record_bar(event.data, event_type=event.type)

    def process_factor_event(self, event: Event):
        """Processes incoming factor data."""
        self.record_factor(event.data, event_type=event.type)

    def process_tick_event(self, event: Event):
        """Processes incoming tick data."""
        tick: TickData = event.data
        vt_symbol = tick.vt_symbol

        with self.lock:
            if vt_symbol in self.tick_recordings:
                self.record_tick(tick)

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

    def process_bar_filling_event(self, event: Event):
        """Processes incoming bar data."""
        self.record_bar(event.data, flush=True, event_type=event.type)

    def process_factor_filling_event(self, event: Event):
        """Processes incoming factor data."""
        self.record_factor(event.data, flush=True, event_type=event.type)

    def process_tick_filling_event(self, event: Event):
        """Processes incoming tick data."""
        tick: TickData = event.data
        vt_symbol = tick.vt_symbol

        with self.lock:
            if vt_symbol in self.tick_recordings:
                self.record_tick(tick, flush=True, event_type=event.type)

            if vt_symbol in self.bar_recordings:
                bg = self.get_bar_generator(vt_symbol)
                bg.update_tick(tick, flush=True)

    # ----------------------------------------------------------------------
    # Data recording methods (putting data into the queue)
    # ----------------------------------------------------------------------

    def record_tick(self, tick: TickData, flush=False, event_type: str = None):
        """Puts tick data into the processing queue."""
        type_ = "tick_flush" if flush else "tick"
        event_type = "" if not event_type else event_type
        task = (type_, deepcopy(tick), event_type)
        self.queue.put(task)

    def record_bar(self, bar: BarData, flush=False, event_type: str = None):
        """Puts bar data into the processing queue."""
        type_ = "bar_flush" if flush else "bar"
        event_type = "" if not event_type else event_type
        task = (type_, deepcopy(bar), event_type)
        self.queue.put(task)

    def record_factor(self, factor: FactorData | dict, flush=False, event_type: str = None):
        """
        Puts factor data into the queue.
        DataFrame-based factors MUST be in a dict: {factor_key: pl.DataFrame}.
        """
        type_ = "factor_flush" if flush else "factor"
        event_type = "" if not event_type else event_type
        task = (type_, deepcopy(factor), event_type)
        self.queue.put(task)

    # ----------------------------------------------------------------------
    # Core data saving logic (called by worker thread)
    # ----------------------------------------------------------------------

    def save_data(
            self,
            task_type: Literal["bar", "factor", "tick"] | None = None,
            data=None,
            force_save: bool = False,
            event_type: str = None,
    ):
        """Routes data from the queue to the appropriate processing helper."""
        event_type = "not specified" if not event_type else event_type
        self.write_log(
            f"saving data: {task_type}, force_save={force_save}, event_type={event_type}",
            level=INFO if task_type else NOTSET,
        )
        if task_type == "tick":
            self.database_manager.save_tick_data([data])
        elif task_type == "bar":
            self._process_bar_data(data=data, force_save=force_save)
        elif task_type == "factor":
            self._process_factor_data(data=data, force_save=force_save)
        elif force_save:
            # This case is for flushing buffers when the queue is empty
            self._flush_all_buffers()

    def _process_bar_data(self, data: BarData, force_save: bool):
        """Helper to process and buffer bar data."""
        with self.lock:
            self.buffer_bar[data.vt_symbol].append(data)
            to_remove = []
            for k, v in self.buffer_bar.items():
                if len(v) >= self.buffer_size or (force_save and v):
                    self._save_bar_buffer(v, stream=False)
                    to_remove.append(k)
            for k in to_remove:
                self.buffer_bar.pop(k, None)

    def _process_factor_data(self, data: FactorData | dict, force_save: bool):
        """Helper to process and buffer/save factor data."""
        with self.lock:
            if isinstance(data, FactorData):
                self.write_log(f"Processing FactorData: {data.factor_name} = {data.value} for {data.vt_symbol}", level=INFO)
                self.buffer_factor[data.vt_symbol].append(data)
                to_remove = []
                for k, v in self.buffer_factor.items():
                    if len(v) >= self.buffer_size or (force_save and v):
                        self.write_log(f"Saving {len(v)} factor records for {k}, factor_name: {v[0].factor_name}", level=INFO)
                        try:
                            self.database_manager.save_factor_data(name=v[0].factor_name, data=v)
                            self.write_log(f"Successfully saved factor data for {k}", level=INFO)
                        except Exception as e:
                            self.write_log(f"Error saving factor data for {k}: {e}", level=ERROR)
                        to_remove.append(k)
                for k in to_remove:
                    self.buffer_factor.pop(k, None)
            elif InstanceChecker.is_dict_of(data, type_value=pl.DataFrame, type_key=str):
                self._process_factor_dict(data)
            else:
                self.write_log(
                    f"Unsupported data type for factor task: {type(data)}. ",
                    level=ERROR,
                )

    def _flush_all_buffers(self):
        """Force-saves all data remaining in any buffer."""
        with self.lock:
            if not self.buffer_bar and not self.buffer_factor:
                return  # Nothing to flush
            self.write_log("Flushing all data buffers...")
            for v in self.buffer_bar.values():
                if v:
                    self._save_bar_buffer(v, stream=False)
            self.buffer_bar.clear()
            for v in self.buffer_factor.values():
                if v:
                    self.database_manager.save_factor_data(name=v[0].factor_name, data=v)
            self.buffer_factor.clear()
            self.write_log("Buffers flushed.")

    def _save_bar_buffer(self, bar_list: list[BarData], stream=True):
        """Saves a list of bars from the buffer to the database."""
        sample_data = bar_list[0]
        self.database_manager.save_bar_data(
            bar_list,
            interval=sample_data.interval,
            exchange=sample_data.exchange,
            stream=stream
        )

    def _process_factor_dict(self, data: dict):
        """
        Converts a dict of wide-format factor DataFrames to a single
        long-format DataFrame and saves it.
        """
        long_format_dfs = []
        checked_interval = None
        for factor_key, wide_df in data.items():
            if not isinstance(wide_df, pl.DataFrame):
                self.write_log(f"Value for '{factor_key}' is not a DataFrame. Skipping.", level=ERROR)
                continue
            interval, _ = extract_factor_key(factor_key)
            if checked_interval is None:
                checked_interval = interval
            else:
                assert interval == checked_interval, "All factors in dict must have same interval."

            df_long = wide_df.melt(
                id_vars=["datetime"],
                value_vars=[col for col in wide_df.columns if col != "datetime"],
                variable_name="ticker",
                value_name=factor_key,
            )
            long_format_dfs.append(df_long)
        if not long_format_dfs:
            return
        final_df = long_format_dfs[0]
        for i in range(1, len(long_format_dfs)):
            final_df = final_df.join(
                long_format_dfs[i], on=["datetime", "ticker"], how="outer_coalesce"
            )
        self.database_manager.save_factor_data(data=final_df, interval=checked_interval)

    # ----------------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------------

    def write_log(self, msg: str, level: int = INFO) -> None:
        """Sends a log event to the main event bus."""
        log = LogData(msg=msg, gateway_name=APP_NAME, level=level)
        event = Event(EVENT_LOG, log)
        self.event_engine.put(event)

    def put_event(self) -> None:
        """Puts current recording status into the event engine for UI updates."""
        with self.lock:
            tick_symbols = sorted(self.tick_recordings.keys())
            bar_symbols = sorted(self.bar_recordings.keys())
        data = {"tick": tick_symbols, "bar": bar_symbols}
        event = Event(EVENT_RECORDER_UPDATE, data)
        self.event_engine.put(event)

    def get_bar_generator(self, vt_symbol: str) -> BarGenerator:
        """Gets or creates a BarGenerator for a specific symbol."""
        # This method is called from within a locked block, so access is safe.
        bg = self.bar_generators.get(vt_symbol)
        if not bg:
            bg = BarGenerator(self.record_bar)
            self.bar_generators[vt_symbol] = bg
        return bg

    def subscribe(self, contract: ContractData) -> None:
        """Subscribes to market data for a contract."""
        # This method is called from within a locked block.
        req = SubscribeRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            interval=getattr(contract, 'interval', None)
        )
        self.main_engine.subscribe(req, contract.gateway_name)
