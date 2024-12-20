# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : ${DIR_PATH}
# @File     : ${FILE_NAME}
# @Time     : 2024/9/28 21:00
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
from unittest import TestCase
import multiprocessing
from time import sleep
from datetime import datetime, time
from logging import INFO
import pprint

from vnpy.event import EventEngine,Event
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import SubscribeRequest, BarData
from vnpy.trader.constant import Exchange,Interval

from vnpy.gateway.binance import BinanceSpotGateway
from vnpy.app.data_recorder import DataRecorderApp
from vnpy.app.factor_maker import FactorEngine
from vnpy.app.factor_maker.factors import OPEN


class TestFactorEngine(TestCase):
    def setUp(self):
        event_engine = EventEngine()
        main_engine = MainEngine(event_engine)
        self.factor_engine = FactorEngine(main_engine, event_engine)
        self.factor_engine.init_engine()

    def test_pipeline(self):
        """测试因子从数据源到因子计算和最终入库的整个流程

        Returns
        -------

        """

        def run_child():
            """
            Running in the child process.
            """
            SETTINGS["log.file"] = True

            event_engine = EventEngine()
            main_engine = MainEngine(event_engine)
            # main_engine.add_gateway(CtpGateway)
            # cta_engine = main_engine.add_app(CtaStrategyApp)
            main_engine.write_log("主引擎创建成功")

            main_engine.add_gateway(BinanceSpotGateway, "BINANCE_SPOT")
            binance_gateway_setting = {
                "key": SETTINGS.get("gateway.api_key", ""),
                "secret": SETTINGS.get("gateway.api_secret", ""),
                "server": "REAL"
            }
            main_engine.connect(binance_gateway_setting, "BINANCE_SPOT")
            main_engine.write_log("连接币安接口")

            main_engine.subscribe(SubscribeRequest(symbol='btcusdt', exchange=Exchange.BINANCE),
                                  gateway_name='BINANCE_SPOT')

        def init_engine(self):
            fct = OPEN
            factor_class = str(fct).rsplit(sep='.', maxsplit=1)[1][:-2]  # 去掉最后的"'>"
            factor_symbol = fct.symbol
            factor_name = fct.factor_name
            # factor_class = factor_class[0]+'.'+factor_class[1].upper()
            self.factor_engine.add_factor(factor_class, factor_name, ticker='btcusdt',
                                          setting={'freq': Interval.MINUTE,
                                                   'symbol': factor_symbol,
                                                   'factor_name': factor_name,
                                                   'exchange': Exchange.BINANCE})  # self.freq.value, self.symbol, self.factor_name, self.exchange.value

        self.setUp()

        init_engine(self)
        self.factor_engine.start_all_factors()
        buffer=[]
        while not self.factor_engine.event_engine._queue.empty():

            res: Event=self.factor_engine.event_engine._queue.get(block=True, timeout=1)
            buffer.append(res)
            # sleep(10)
        print(buffer)

        # pprint.pprint(self.factor_engine.__dict__)

    def test_init_engine(self):
        self.fail()

    def test_get_all_factor_class_names(self):
        self.fail()

    def test_load_factor_class(self):
        self.fail()

    def test_load_factor_setting(self):
        self.fail()

    def test_save_factor_setting(self):
        self.fail()

    def test_load_factor_data(self):
        self.fail()

    def test_sync_factor_data(self):
        self.fail()

    def test_start_all_factors(self):
        self.fail()

    def test_start_factor(self):
        self.fail()

    def test_load_bar(self):
        self.fail()

    def test_update_memory(self):
        self.fail()
