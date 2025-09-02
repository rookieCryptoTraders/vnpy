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
from time import sleep

from tqdm import tqdm

from vnpy.app.data_recorder import DataRecorderApp
from vnpy.app.vnpy_datamanager import DataManagerApp, DataManagerEngine
from vnpy.config import match_format_string
from vnpy.event import EventEngine
from vnpy.factor import FactorMakerApp
from vnpy.factor.engine import FactorEngine
from vnpy.gateway.mimicgateway.mimicgateway import MimicGateway
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import VTSYMBOL_OVERVIEW, DataRange, TimeRange
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import BarData
from vnpy.utils.datetimes import DatetimeUtils, TimeFreq

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
    factor_maker_engine: FactorEngine = main_engine.add_app(FactorMakerApp)
    factor_maker_engine.init_engine(fake=False)
    main_engine.write_log(f"Started [{factor_maker_engine.__class__.__name__}]")

    # start data recorder
    data_recorder_engine = main_engine.add_app(DataRecorderApp)
    data_recorder_engine.update_schema(database_name=data_recorder_engine.database_manager.database_name,
                                       exchanges=main_engine.exchanges,
                                       intervals=main_engine.intervals,
                                       factor_keys=[key for key in factor_maker_engine.flattened_factors.keys()])
    data_recorder_engine.start()
    main_engine.write_log(f"Started [{data_recorder_engine.__class__.__name__}]")

    # init gateway
    gateway = main_engine.add_gateway(MimicGateway, "MIMIC")
    # main_engine.connect(gateway_settings, "MIMIC")

    # download data using vnpy_datamanager if data missed
    data_manager_engine: DataManagerEngine = main_engine.add_app(DataManagerApp,
                                                                 database=data_recorder_engine.database_manager)
    data_manager_engine.init_engine()
    main_engine.write_log(f"Started [{data_manager_engine.__class__.__name__}]")

    # makeup missing data (download data, save bars, calculate factors, save factors)
    for i in range(3):  # allow 3 attempts to download data
        # gaps to requests
        gap_dict = data_recorder_engine.database_manager.get_gaps(end_time=datetime.datetime.now(),
                                                                  start_time=datetime.datetime(2025, 8, 30, 12, 30))
        # no gap, break
        if all(len(gap) == 0 for gap in gap_dict.values()):
            break

        # align gap_dict between overview keys. which means even if one of the vt_symbol bar data if full, we have to download and re-calculate them
        gap_keys = list(gap_dict.keys())
        universal_gaps = DataRange(interval=DatetimeUtils.freq2interval(main_engine.minimum_freq))
        for gaps in gap_dict.values():
            universal_gaps.add_ranges(gaps, inplace=True, method='union')

        # to simplify logic, we download all data from the very beginning of the universal gaps to the end of the universal gaps
        gap_dict = {
            key: [TimeRange(interval=main_engine.minimum_interval, start=universal_gaps.start, end=universal_gaps.end)]
            for key in gap_keys}

        # have gaps, download data and fill bar gaps, insert bars into database
        gap_data_dict = data_manager_engine.download_bar_data_gaps(gap_dict)

        # re-insert all data into database. this step won't trigger factor calculation
        for overview_key, data_list in gap_data_dict.items():
            # print(f"main Processing overview key: {overview_key}, data count: {len(data_list)}")
            info = match_format_string(VTSYMBOL_OVERVIEW, overview_key)
            bar_list = [BarData(
                gateway_name=gateway.gateway_name,
                symbol=info['symbol'],
                exchange=Exchange(info['exchange']),
                interval=Interval(info['interval']),
                datetime=d['datetime'],
                volume=d['volume'],
                open_price=d['open'],
                high_price=d['high'],
                low_price=d['low'],
                close_price=d['close'],
                quote_asset_volume=d['quote_asset_volume'],
                number_of_trades=d['number_of_trades'],
                taker_buy_base_asset_volume=d['taker_buy_base_asset_volume'],
                taker_buy_quote_asset_volume=d['taker_buy_quote_asset_volume'],
            ) for d in data_list]
            gateway.on_bar_filling(bar_list)  # todo: change belonging to data_manager_engine
        time.sleep(5)

        # after bar data is inserted into database, we need to re-calculate factors for all data
        # because factor calculation may depend on previous bars, we have to re-calculate all bars
        # todo:
        #  1. load <window>-1 bars: init bar memory
        #  2. request <window> bars to calculate <window> factors: mock live data by calling one by one gateway.on_bar to trigger non-factor-dependent factor calculation. re-insert these factors into database and factor memory
        #  3. request <missing> bars to calculate <missing> factors: mock live data by calling one by one gateway.on_bar to trigger non-factor-dependent and factor-dependent factor calculation
        # 1. load <window>-1 bars: init bar memory
        start_dt = datetime.datetime.now()
        for overview_key, time_ranges in gap_dict.items():
            tmp = min([time_range.start for time_range in time_ranges])
            if tmp < start_dt:
                start_dt = tmp
        params = {
            "begin": start_dt - datetime.timedelta(minutes=1),  # start_dt is the start of filling gap factors
            "end": start_dt - datetime.timedelta(minutes=1),  # start_dt is the start of filling gap factors
            "interval": Interval.MINUTE,
            "ticker_list": main_engine.vt_symbols,
        }
        factor_maker_engine.send_load_bar_request(type_='period', param=params)
        for i in range(5):  # wait for bar memory to be ready
            time.sleep(1)

        #  2. request <window> bars to calculate <window> factors:
        #       mock live data by calling one by one gateway.on_bar to trigger non-factor-dependent factor calculation.
        #       re-insert these factors into database and factor memory
        #       future implementation should be `factor_maker_engine.send_load_factor_request(type_='gap', param=gap_dict)`
        data_lens_list = [len(data_list) for data_list in gap_data_dict.values()]
        assert min(data_lens_list) == max(data_lens_list), "data count mismatch between symbols"
        data_length = data_lens_list[0]
        # for i in tqdm(range(data_length), desc="Recalculating factors for initial bars"):
        for i in range(data_length):
            for overview_key, data_list in gap_data_dict.items():
                # print(f"main Processing overview key: {overview_key}, data count: {len(data_list)}")
                info = match_format_string(VTSYMBOL_OVERVIEW, overview_key)
                d = data_list[i]
                bar = BarData(
                    gateway_name=gateway.gateway_name,
                    symbol=info['symbol'],
                    exchange=Exchange(info['exchange']),
                    interval=Interval(info['interval']),
                    datetime=d['datetime'],
                    volume=d['volume'],
                    open_price=d['open'],
                    high_price=d['high'],
                    low_price=d['low'],
                    close_price=d['close'],
                    quote_asset_volume=d['quote_asset_volume'],
                    number_of_trades=d['number_of_trades'],
                    taker_buy_base_asset_volume=d['taker_buy_base_asset_volume'],
                    taker_buy_quote_asset_volume=d['taker_buy_quote_asset_volume'],
                )
                gateway.on_bar(bar)  # todo: change belonging to data_manager_engine
            a = 1

        # filling missing factor
        # gateway.on_factor_filling(gap_dict)
        break

    # Start live data subscription
    # main_engine.connect(gateway_settings, "MIMIC")
    # main_engine.subscribe_all(gateway_name='MIMIC')


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
