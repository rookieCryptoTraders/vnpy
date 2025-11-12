# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/tests
# @File     : real_test.py
# @Time     : 2025/1/21 19:25
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
import datetime
import multiprocessing
import time
import traceback
from datetime import timedelta
from time import sleep
from logging import WARNING
import polars as pl

from vnpy.app.data_recorder import DataRecorderApp, DataRecorderEngine
from vnpy.app.vnpy_datamanager import DataManagerApp, DataManagerEngine
from vnpy.config import match_format_string
from vnpy.event import EventEngine
from vnpy_clickhouse.exceptions import InsertError
from vnpy_factor import FactorMakerApp
from vnpy_factor.factor_engine import FactorEngine
from vnpy_factor.factor_registry import FactorRegistry
from vnpy.gateway.mimicgateway.mimicgateway import MimicGateway
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import BAR_OVERVIEW_KEY
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import BarData

import os
import glob

__PROJECT_SETTING_DIR__ = ''


def cleanup_temp_files(directory, suffix=".tmp"):
    """
    Removes all temp files ending with the given suffix in the specified directory.
    """
    pattern = os.path.join(directory, f"*{suffix}")
    temp_files = glob.glob(pattern)
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Removed temp file: {temp_file}")
        except Exception as e:
            print(f"Could not remove {temp_file}: {e}")


# Usage example:

def run_child():
    """
    1. start gateway
    2. feed data to factor engine
    3. push bar and factors into database
    4. backfill missing bar data
    """
    global __PROJECT_SETTING_DIR__

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.write_log("Main engine created successfully")
    __PROJECT_SETTING_DIR__ = main_engine.TEMP_DIR
    print(__PROJECT_SETTING_DIR__)

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
    factor_maker_engine: FactorEngine = main_engine.add_app(FactorMakerApp, registry=FactorRegistry())
    factor_maker_engine.init_engine(use_talib=False)
    main_engine.write_log(f"Started [{factor_maker_engine.__class__.__name__}]")

    # start data recorder
    data_recorder_engine: DataRecorderEngine = main_engine.add_app(DataRecorderApp)
    data_recorder_engine.update_schema(database_name=data_recorder_engine.database_manager.database_name,
                                       exchanges=main_engine.exchanges,
                                       intervals=main_engine.intervals,
                                       factor_keys=[key for key in factor_maker_engine.factor_data.keys()])
    data_recorder_engine.start()
    main_engine.write_log(f"Started [{data_recorder_engine.__class__.__name__}]")

    # init gateway
    gateway = main_engine.add_gateway(MimicGateway, "MIMIC")

    # download data using vnpy_datamanager if data missed
    data_manager_engine: DataManagerEngine = main_engine.add_app(DataManagerApp,
                                                                 database=data_recorder_engine.database_manager)
    data_manager_engine.init_engine()
    main_engine.write_log(f"Started [{data_manager_engine.__class__.__name__}]")

    # makeup missing data (download data, save bars, calculate factors, save factors)
    for i in range(3):  # allow 3 attempts to download data
        if i > 0:
            data_manager_engine.write_log(f"Retrying data gap filling, attempt {i + 1}/3...", level=WARNING)
        # gaps to requests
        gap_dict = data_recorder_engine.database_manager.get_gaps(end_time=datetime.datetime.now(),
                                                                  start_time=datetime.datetime(2025, 11, 12, 5, 30))
        # no gap, break
        if all(len(gap) == 0 for gap in gap_dict.values()):
            break

        # have gaps, download data and fill bar gaps, insert bars into database
        gap_data_dict = data_manager_engine.download_bar_data_gaps(gap_dict)

        # insert bar data using dataframes
        for overview_key, data_dict in gap_data_dict.items():
            for period_start, data in data_dict.items():
                data_frame = pl.DataFrame(data['data'])
                data_frame = data_frame.with_columns(
                    pl.lit(data['symbol']).alias("symbol"),
                    pl.lit(data['exchange']).alias("exchange"),
                    pl.lit(data['interval']).alias("interval"),
                )
                data_frame = data_frame.rename({
                    "open":"open_price",
                    "high":"high_price",
                    "low":"low_price",
                    "close":"close_price",
                })
                gateway.on_bar_filling(data_frame)

        for i in range(60):  # wait for all data to be processed
            data_recorder_engine.write_log(f"processing bar data queues... {data_recorder_engine.queue.qsize()} left")
            time.sleep(3)
            if event_engine.queue.empty() and data_recorder_engine.queue.empty():
                break
            if i == 19:
                raise InsertError("Slow insertion. Not all datas have been saved")

        fill_period = factor_maker_engine.fill_historical_factors(request_data=gap_dict)
        for i in range(60):  # wait for all data to be processed
            data_recorder_engine.write_log(
                f"processing factor data queues... {data_recorder_engine.queue.qsize()} left")
            time.sleep(3)
            if event_engine.queue.empty() and data_recorder_engine.queue.empty():
                break
            if i == 19:
                raise InsertError("Slow insertion. Not all datas have been saved")
        if len(fill_period) > 0:
            if fill_period['bar_end'].replace(microsecond=0) + timedelta(minutes=1) <= datetime.datetime.now().replace(
                    microsecond=0):
                data_manager_engine.write_log(
                    f"Factor recalculation for missing bars {fill_period['bar_start']} - {fill_period['bar_end']} completed. but new bar comes in, wait for next round...")
                continue
            else:
                data_manager_engine.write_log(
                    f"Factor recalculation for missing bars {fill_period['bar_start']} - {fill_period['bar_end']} completed.")
                break

    # Start live data subscription
    main_engine.connect(gateway_settings, "MIMIC")
    main_engine.subscribe_all(gateway_name='MIMIC')


def run_parent():
    """
    Running in the parent process.
    """
    print("Starting parent process")

    # Crypto markets trade 24/7
    child_process = None

    try:
        if child_process is None:
            print("Starting child process")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("Child process started successfully")

            # Keep the parent process running
            while True:
                sleep(5)
                if not child_process.is_alive():
                    print("Child process unexpectedly exited, restarting...")
                    child_process.join()
                    child_process = multiprocessing.Process(target=run_child)
                    child_process.start()

    except KeyboardInterrupt:
        if child_process is not None:
            print("Shutting down child process")
            child_process.terminate()
            child_process.join()
            child_process = None
            print("Child process shutdown successful")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_tb(e.__traceback__)
        if child_process is not None:
            child_process.terminate()
            child_process.join()
            child_process = None
            print("Child process terminated due to error")


if __name__ == '__main__':
    run_parent()
    # cleanup_temp_files(__PROJECT_SETTING_DIR__)
