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

from vnpy.factor.engine import FactorEngine
# from vnpy.app.factor_maker import FactorMakerApp
from vnpy.event import EventEngine
from vnpy.gateway.mimicgateway.mimicgateway import MimicGateway
from vnpy.trader.engine import MainEngine
from vnpy.app.data_recorder import DataRecorderApp
from vnpy.strategy.engine import StrategyEngine # Added import


# from vnpy.strategy.examples.test_strategy_template import TestStrategyTemplate


def run_child():
    """
    1. start gateway
    2. feed data to factor engine
    3. push bar and factors into database
    """

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.write_log("Main engine created successfully")

    # start factor engine
    factor_maker_engine: FactorEngine = main_engine.add_engine(FactorEngine)
    factor_maker_engine.init_engine(fake=False)
    # Log active factors
    active_factor_keys = list(factor_maker_engine.flattened_factors.keys())
    main_engine.write_log(f"FactorEngine initialized with {len(active_factor_keys)} active factors. Keys: {active_factor_keys[:5]}")
    # Log first 5 keys for brevity

    # Add and initialize StrategyEngine
    main_engine.write_log("Adding Strategy Engine...")
    strategy_engine = main_engine.add_engine(StrategyEngine)
    strategy_engine.init_engine() # This loads strategies from strategy_config.json
    main_engine.write_log("Strategy Engine added and initialized.")

    # Start strategies
    for strategy_name in strategy_engine.strategies:
        strategy = strategy_engine.strategies[strategy_name]
        if hasattr(strategy, 'on_start') and callable(strategy.on_start):
            main_engine.write_log(f"Starting strategy: {strategy_name}")
            strategy.on_start() # Directly call on_start for testing
        else:
            main_engine.write_log(f"Strategy {strategy_name} does not have on_start or it's not callable.")

    # Log active strategies and their parameters
    active_strategy_names = list(strategy_engine.strategies.keys())
    main_engine.write_log(f"StrategyEngine initialized with {len(active_strategy_names)} active strategies: {active_strategy_names}")
    for strategy_name in active_strategy_names:
        strategy_instance = strategy_engine.strategies[strategy_name]
        main_engine.write_log(f"Strategy '{strategy_name}' params: {strategy_instance.get_parameters()}")

    gateway_settings = {
        "symbols": ["SYM1.LOCAL", "SYM2.LOCAL", "SYM3.LOCAL", "SYM4.LOCAL"], # Example symbols
        "simulation_interval_seconds": 2,  # Bars every second for each symbol
        "open_price_range_min": 100,
        "open_price_range_max": 105,
        "price_change_range_min": -1,
        "price_change_range_max": 1,
        "volume_range_min": 50,
        "volume_range_max": 200
    }

    # connect to exchange
    main_engine.add_gateway(MimicGateway, "MIMIC")
    main_engine.subscribe_all(gateway_name='MIMIC')

    main_engine.connect(gateway_settings, "MIMIC")
    main_engine.write_log("Connected to MIMIC interface")
    main_engine.subscribe_all(gateway_name='MIMIC')

    # start data recorder
    data_recorder_engine = main_engine.add_app(DataRecorderApp)
    main_engine.write_log(f"启动[{data_recorder_engine.__class__.__name__}]")
    data_recorder_engine.update_schema(database_name=data_recorder_engine.database_manager.database_name,
                                       exchanges=main_engine.exchanges,
                                       intervals=main_engine.intervals,
                                       factor_keys=[key for key in factor_maker_engine.flattened_factors.keys()])


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
