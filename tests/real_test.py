# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/tests
# @File     : real_test.py
# @Time     : 2025/1/21 19:25
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import multiprocessing
from datetime import datetime, time
from logging import DEBUG
from time import sleep

from vnpy.app.data_recorder import DataRecorderApp
from vnpy.app.factor_maker import FactorEngine
from vnpy.app.factor_maker import FactorMakerApp
from vnpy.event import EventEngine
from vnpy.gateway.binance import BinanceSpotGateway
from vnpy.trader.engine import MainEngine
from vnpy.trader.setting import SETTINGS
from vnpy.strategy.engine import BaseStrategyEngine
from vnpy.strategy.examples.test_strategy_template import TestStrategyTemplate


def run_child():
    """
    1. start gateway
    2. feed data to factor engine
    3. push bar and factors into database
    """

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.write_log("主引擎创建成功")

    # connect to exchange
    main_engine.add_gateway(BinanceSpotGateway, "BINANCE_SPOT")
    binance_gateway_setting = {
        "key": SETTINGS.get("gateway.api_key", ""),
        "secret": SETTINGS.get("gateway.api_secret", ""),
        "server": "REAL"
    }
    main_engine.connect(binance_gateway_setting, "BINANCE_SPOT")
    main_engine.write_log("连接币安接口")
    main_engine.subscribe_all(gateway_name='BINANCE_SPOT')

    # start data recorder
    data_recorder_engine = main_engine.add_app(DataRecorderApp)
    main_engine.write_log(f"启动[{data_recorder_engine.__class__.__name__}]")

    factor_maker_engine: FactorEngine = main_engine.add_app(FactorMakerApp)
    factor_maker_engine.init_engine(fake=True)
    main_engine.write_log(f"启动[{factor_maker_engine.__class__.__name__}]")

    # # test engine
    strategy_engine: BaseStrategyEngine = main_engine.add_engine(BaseStrategyEngine)

    # init strategy template
    template_strategy = TestStrategyTemplate(strategy_engine=strategy_engine,
                                             strategy_name=TestStrategyTemplate.strategy_name,
                                             vt_symbols=['btcusdt.BINANCE'], setting={})

    strategy_engine.init_engine(strategies_path="vnpy/tests/strategy/strategies",
                                strategies={template_strategy.strategy_name: template_strategy})
    main_engine.write_log(f"启动[{strategy_engine.__class__.__name__}]")

    # log_engine = main_engine.get_engine("log")
    # event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    # main_engine.write_log("注册日志事件监听")
    #
    # main_engine.connect(ctp_setting, "CTP")
    # main_engine.write_log("连接CTP接口")
    #
    # sleep(10)
    #
    # cta_engine.init_engine()
    # main_engine.write_log("CTA策略初始化完成")
    #
    # cta_engine.init_all_strategies()
    # sleep(60)   # Leave enough time to complete strategy initialization
    # main_engine.write_log("CTA策略全部初始化")
    #
    # cta_engine.start_all_strategies()
    # main_engine.write_log("CTA策略全部启动")
    counter = 0
    while True:
        if counter == 5:
            main_engine.write_log("running", level=DEBUG)
            counter = 0
        counter += 1
        sleep(1)


def run_parent():
    """
    Running in the parent process.
    """
    print("启动父进程")

    # Chinese futures market trading period (day/night)
    DAY_START = time(8, 45)
    DAY_END = time(15, 30)

    NIGHT_START = time(20, 45)
    NIGHT_END = time(2, 45)

    child_process = None

    while True:
        current_time = datetime.now().time()
        trading = False

        # Check whether in trading period
        if (
                (current_time >= DAY_START and current_time <= DAY_END)
                or (current_time >= NIGHT_START)
                or (current_time <= NIGHT_END)
        ) or True:
            trading = True

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            print("关闭子进程")
            child_process.terminate()
            child_process.join()
            child_process = None
            print("子进程关闭成功")

        sleep(5)


if __name__ == '__main__':
    run_parent()
