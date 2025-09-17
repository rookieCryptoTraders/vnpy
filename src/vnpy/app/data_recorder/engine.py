from collections import defaultdict
from copy import deepcopy
from logging import ERROR, INFO
from queue import Empty, Queue
from threading import Lock, Thread

from vnpy.event.engine import Event
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.engine import BaseEngine, EventEngine, MainEngine
from vnpy.trader.event import (
    EVENT_BAR,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_RECORDER_UPDATE,
    EVENT_TICK,
)
from vnpy.trader.object import (
    BarData,
    ContractData,
    LogData,
    SubscribeRequest,
    TickData,
)
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import BarGenerator

APP_NAME = "DataRecorder"
SYSTEM_MODE = SETTINGS.get("system.mode", "TEST")


class DataRecorderEngine(BaseEngine):
    """
    Data recording engine for capturing and storing market data.

    Handles real-time recording of tick and bar data from market feeds,
    with buffering and batch processing for efficient database operations.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """Initializes the DataRecorderEngine."""
        super().__init__(main_engine, event_engine, engine_name=APP_NAME)

        self.queue: Queue = Queue()
        self.thread: Thread = Thread(target=self.run)
        self.active: bool = False
        self.lock: Lock = Lock()

        self.tick_recordings: dict[str, dict] = {}
        self.bar_recordings: dict[str, dict] = {}
        self.bar_generators: dict[str, BarGenerator] = {}

        self.buffer_bar: defaultdict = defaultdict(list)
        self.buffer_size: int = 1000 if SYSTEM_MODE != "TEST" else 1

        self.database = get_database()

        self.register_event()

    def register_event(self) -> None:
        """Registers the engine for relevant events."""
        self.event_engine.register(EVENT_BAR, self.process_bar_event)
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)

    def run(self):
        """Main loop for the worker thread to save data."""
        while self.active:
            try:
                task = self.queue.get(timeout=1)
                task_type, data = task
                if task_type == "bar":
                    self._process_bar_data(data)
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
        self.write_log("Data recorder engine closed.")

    def start(self):
        """Starts the engine and its worker thread."""
        if self.active:
            return
        self.active = True
        self.thread.start()
        self.put_event()
        self.write_log("Data recorder engine started.")

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
                self.write_log(
                    f"Contract not found for bar recording: {vt_symbol}", level=ERROR
                )
                return
            self.bar_recordings[vt_symbol] = {
                "symbol": contract.symbol,
                "exchange": contract.exchange.value,
            }
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
                self.write_log(
                    f"Contract not found for tick recording: {vt_symbol}", level=ERROR
                )
                return
            self.tick_recordings[vt_symbol] = {
                "symbol": contract.symbol,
                "exchange": contract.exchange.value,
            }
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



    def process_tick_event(self, event: Event):
        """Processes incoming tick data to generate bars."""
        tick: TickData = event.data
        vt_symbol = tick.vt_symbol
        with self.lock:
            if vt_symbol in self.tick_recordings:
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
                self.database.save_bar_data(bars_to_save)



    def _flush_all_buffers(self):
        """Force-saves all data remaining in any buffer."""
        with self.lock:
            if not self.buffer_bar:
                return
            for bars in self.buffer_bar.values():
                if bars:
                    self.database.save_bar_data(bars)
            self.buffer_bar.clear()
            self.write_log("Flushed all data buffers.", level=INFO)



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
