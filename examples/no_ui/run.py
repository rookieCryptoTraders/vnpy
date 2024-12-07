import multiprocessing
from time import sleep
from datetime import datetime, time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy.trader.object import SubscribeRequest, BarData
from vnpy.trader.constant import Exchange

# from vnpy.gateway.ctp import CtpGateway
# from vnpy.app.cta_strategy import CtaStrategyApp
# from vnpy.app.cta_strategy.base import EVENT_CTA_LOG

from vnpy.gateway.binance import BinanceSpotGateway
from vnpy.app.data_recorder import DataRecorderApp
from vnpy.app.factor_maker import FactorMakerApp

SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True
SETTINGS["log.file"] = True


def run_child():
    """
    Running in the child process.
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
    main_engine.subscribe(SubscribeRequest(symbol='btcusdt', exchange=Exchange.BINANCE), gateway_name='BINANCE_SPOT')

    # # start data recorder
    # data_recorder_engine=main_engine.add_app(DataRecorderApp)
    # main_engine.write_log("启动数据记录程序")

    factor_maker_engine = main_engine.add_app(FactorMakerApp)
    factor_maker_engine.init_engine()
    main_engine.write_log("启动因子计算程序")

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

    while True:
        # print(main_engine.event_engine._queue.get())
        # print(main_engine.event_engine._queue.get().type)
        # print(type(main_engine.event_engine._queue.get()))
        sleep(1)


def run_parent():
    """
    Running in the parent process.
    """
    print("启动CTA策略守护父进程")

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
        ) or True:  # fixme
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


if __name__ == "__main__":
    run_parent()
