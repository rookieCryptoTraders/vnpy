from pathlib import Path

from vnpy.trader.app import BaseApp

from .engine import DataManagerEngine, APP_NAME


class DataRecorderApp(BaseApp):
    """"""
    app_name = APP_NAME
    app_module = None
    app_path = Path(__file__).parent
    display_name = "行情记录"
    engine_class = DataManagerEngine
