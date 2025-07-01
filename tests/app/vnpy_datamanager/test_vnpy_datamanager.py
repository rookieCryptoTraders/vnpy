from datetime import datetime
from unittest import TestCase
from unittest.mock import patch, MagicMock

from vnpy.app.vnpy_datamanager import DataManagerEngine
from vnpy.event import EventEngine
from vnpy.trader.constant import Exchange
from vnpy.trader.constant import Interval
from vnpy.trader.engine import MainEngine

import multiprocessing
from time import sleep
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from vnpy.trader.constant import Exchange, Interval
from vnpy.factor.engine import FactorEngine
from vnpy.factor import FactorMakerApp
from vnpy.event import EventEngine
from vnpy.gateway.mimicgateway.mimicgateway import MimicGateway
from vnpy.trader.engine import MainEngine
from vnpy.app.data_recorder import DataRecorderApp
from vnpy.trader.object import BarData
from vnpy.app.vnpy_datamanager import DataManagerEngine,DataManagerApp


class TestDataManagerEngine(TestCase):
    def setUp(self):
        event_engine = EventEngine()
        self.main_engine = MainEngine(event_engine)
        self.main_engine.write_log("Main engine created successfully")

        gateway_settings = {
            "symbols": [],
            "simulation_interval_seconds": 4.0,  # Bars every second for each symbol
            "open_price_range_min": 100,
            "open_price_range_max": 105,
            "price_change_range_min": -1,
            "price_change_range_max": 1,
            "volume_range_min": 50,
            "volume_range_max": 200
        }

        # start factor engine
        self.factor_maker_engine: FactorEngine = self.main_engine.add_app(FactorMakerApp)
        self.factor_maker_engine.init_engine(fake=True)
        self.main_engine.write_log(f"Started [{self.factor_maker_engine.__class__.__name__}]")

        # start data recorder
        self.data_recorder_engine = self.main_engine.add_app(DataRecorderApp)
        self.data_recorder_engine.update_schema(database_name=self.data_recorder_engine.database_manager.database_name,
                                           exchanges=self.main_engine.exchanges,
                                           intervals=self.main_engine.intervals,
                                           factor_keys=[key for key in self.factor_maker_engine.flattened_factors.keys()])
        self.data_recorder_engine.start()
        self.main_engine.write_log(f"Started [{self.data_recorder_engine.__class__.__name__}]")

        # Start engines after data is backfilled
        self.factor_maker_engine = self.main_engine.add_app(FactorMakerApp)
        self.main_engine.write_log(f"Started [{self.factor_maker_engine.__class__.__name__}]")

        self.data_manager = DataManagerEngine(self.main_engine, self.main_engine.event_engine)
        
    def test_get_bar_overview(self):
        self.setUp()
        result = self.data_manager.get_bar_overview()
        print(result)

    def test_get_tick_overview(self):
        self.setUp()
        result = self.data_manager.get_tick_overview()
        print(result)

    def test_get_factor_overview(self):
        self.setUp()
        result = self.data_manager.get_factor_overview()
        print(result)

    def test_download_bar_data(self):
        self.setUp()
        self.data_recorder_engine.database_manager.overview_handler.get
        # symbol = "BTCUSDT"
        # exchange = Exchange.BINANCE
        # interval = Interval.MINUTE
        # start = datetime(2023, 7, 20, 12, 0, 0)
        # end = datetime(2023, 7, 20, 12, 59, 59)
        # result:list[dict] = self.data_manager.download_bar_data(
        #     symbol, exchange, interval.value, start, end,save=False
        # )
        # print(result)

    def download_bar_data_20250629(self):
        self.setUp()
        requests = [
            {
                "symbol": "BTCUSDT",
                "exchange": Exchange.BINANCE,
                "interval": Interval.MINUTE.value,
                "start": datetime(2023, 7, 20, 12, 0, 0),
                "end": datetime(2023, 7, 20, 12, 59, 59)
            }
        ]
        result = self.data_manager.download_bar_data_20250629(requests)
        print(result)


    @patch("vnpy.app.vnpy_datamanager.engine.get_database")
    @patch("vnpy.app.vnpy_datamanager.engine.get_datafeed")
    def test_import_data_from_csv(self, mock_get_datafeed, mock_get_database):
        mock_datafeed = MagicMock()
        mock_get_datafeed.return_value = mock_datafeed

        mock_database = MagicMock()
        mock_get_database.return_value = mock_database

        file_path = "test.csv"
        symbol = "BTCUSDT"
        exchange = Exchange.BINANCE
        interval = Interval.MINUTE
        tz_name = "Asia/Shanghai"
        datetime_head = "datetime"
        open_head = "open"
        high_head = "high"
        low_head = "low"
        close_head = "close"
        volume_head = "volume"
        turnover_head = "turnover"
        open_interest_head = "open_interest"
        datetime_format = "%Y-%m-%d %H:%M:%S"

        result = self.data_manager.import_data_from_csv(
            file_path, symbol, exchange, interval, tz_name,
            datetime_head, open_head, high_head, low_head,
            close_head, volume_head, turnover_head,
            open_interest_head, datetime_format
        )

        self.assertIsInstance(result, tuple)

    @patch("vnpy.app.vnpy_datamanager.engine.get_database")
    def test_output_data_to_csv(self, mock_get_database):
        mock_database = MagicMock()
        mock_get_database.return_value = mock_database

        # Mock bar data
        mock_bar_data = [
            MagicMock(
                symbol="BTCUSDT",
                exchange=Exchange.BINANCE,
                datetime=datetime(2023, 7, 20, 12, 0, 0),
                open_price=30000.0,
                high_price=31000.0,
                low_price=29500.0,
                close_price=30500.0,
                volume=100.0,
                turnover=3000000.0,
                open_interest=0.0
            )
        ]
        mock_database.load_bar_data.return_value = mock_bar_data

        file_path = "output_test.csv"
        symbol = "BTCUSDT"
        exchange = Exchange.BINANCE
        interval = Interval.MINUTE
        start = datetime(2023, 7, 20, 12, 0, 0)
        end = datetime(2023, 7, 20, 12, 59, 59)

        result = self.data_manager.output_data_to_csv(
            file_path, symbol, exchange, interval, start, end
        )

        self.assertTrue(result)
        mock_database.load_bar_data.assert_called_once_with(
            symbol, exchange, interval, start, end
        )

        # Verify CSV file content
        with open(file_path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 data row
            self.assertIn("symbol,exchange,datetime,open,high,low,close,volume,turnover,open_interest", lines[0])
            self.assertIn("BTCUSDT,BINANCE,2023-07-20 12:00:00,30000.0,31000.0,29500.0,30500.0,100.0,3000000.0,0.0", lines[1])

# if __name__ == "__main__":
#     unittest.main()