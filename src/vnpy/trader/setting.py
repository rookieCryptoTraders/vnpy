"""
Global setting of the trading platform.
"""

from logging import DEBUG
from tzlocal import get_localzone_name

from .utility import load_json, TRADER_DIR, TEMP_DIR


_VT_SETTING_LOADED = False


SETTINGS: dict = {
    "font.family": "微软雅黑",
    "font.size": 12,

    "log.active": True,
    "log.level": DEBUG,
    "log.console": True,
    "log.file": True,

    "email.server": "smtp.qq.com",
    "email.port": 465,
    "email.username": "",
    "email.password": "",
    "email.sender": "",
    "email.receiver": "",

    "overview.jsonpath": "",

    "datafeed.name": "",
    "datafeed.username": "",
    "datafeed.password": "",

    "database.timezone": get_localzone_name(),
    # "database.name": "sqlite",
    # "database.database": "database.db",
    "database.name": "clickhouse",
    "database.database": "test",
    "database.host": "",
    "database.port": 0,
    "database.user": "",
    "database.password": "",

    # mycode
    "gateway.api_key": "",
    "gateway.api_secret": "",

    # trading
    "vt_symbols": [],

    "factor.settings_file_path": "factor_settings.json",
    "factor.definitions_file_path": "factor_defination_setting.json",
    "strategy.settings_file_path": "strategy_settings.json",
    "strategy.definitions_file_path": "strategy_template_definitions.json",
}

# Load global setting from json file.
SETTING_FILENAME: str = "vt_setting.json"

if not _VT_SETTING_LOADED:
    new_settings,setting_filepath = load_json(SETTING_FILENAME, return_filepath=True)
    SETTINGS.update(new_settings)
    _VT_SETTING_LOADED = True
    
# laoding sensitive info from .env file
import os
from dotenv import load_dotenv
load_dotenv()
SETTINGS.update({
    "gateway.api_key": os.getenv("BINANCE_API_KEY"),
    "gateway.api_secret": os.getenv("BINANCE_API_SECRET"),
})

print(f"[vnpy.trader.setting] Updated SETTINGS from {setting_filepath}")