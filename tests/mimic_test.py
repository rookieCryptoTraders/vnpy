# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/tests
# @File     : real_test.py
# @Time     : 2025/1/21 19:25
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import multiprocessing
from time import sleep
import time

from vnpy.trader.constant import Exchange

from vnpy.factor.engine import FactorEngine
from vnpy.factor import FactorMakerApp
from vnpy.event import EventEngine
from vnpy.gateway.mimicgateway.mimicgateway import MimicGateway
from vnpy.trader.engine import MainEngine
from vnpy.app.data_recorder import DataRecorderApp


# from vnpy.strategy.examples.test_strategy_template import TestStrategyTemplate
# 1. 落实data_recorder对于overview的记录
# 2. data_manager在识别到数据缺失后:
# 	1. 下载bar(需要往前多下载max_window大小的数据)
# 	2. put bar event

def run_child():
    """
    1. start gateway
    2. feed data to factor engine
    3. push bar and factors into database
    """

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.write_log("Main engine created successfully")

    gateway_settings = {
        "symbols": [],
        "simulation_interval_seconds": 2.0,  # Bars every second for each symbol
        "open_price_range_min": 100,
        "open_price_range_max": 105,
        "price_change_range_min": -1,
        "price_change_range_max": 1,
        "volume_range_min": 50,
        "volume_range_max": 200
    }

    # connect to exchange
    main_engine.add_gateway(MimicGateway, "MIMIC")
    # main_engine.subscribe_all(gateway_name='MIMIC')

    main_engine.connect(gateway_settings, "MIMIC")
    main_engine.write_log("Connected to MIMIC interface")
    main_engine.subscribe_all(gateway_name='MIMIC')

    # start factor engine
    factor_maker_engine: FactorEngine = main_engine.add_app(FactorMakerApp)
    factor_maker_engine.init_engine(fake=True)
    main_engine.write_log(f"启动[{factor_maker_engine.__class__.__name__}]")

    # start data recorder
    data_recorder_engine = main_engine.add_app(DataRecorderApp)
    data_recorder_engine.update_schema(database_name=data_recorder_engine.database_manager.database_name,
                                       exchanges=main_engine.exchanges,
                                       intervals=main_engine.intervals,
                                       factor_keys=[key for key in factor_maker_engine.flattened_factors.keys()])
    data_recorder_engine.start()
    main_engine.write_log(f"启动[{data_recorder_engine.__class__.__name__}]")


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


if __name__ == '__main__':
    run_parent()
