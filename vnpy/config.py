# -*- coding=utf-8 -*-
# @Project  : crypto_backtrader
# @FilePath : 
# @File     : config.py
# @Time     : 2024/2/27 13:24
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import os
from pathlib import Path

# global settings
os.environ['TZ'] = 'UTC'

# path related
WORK_DIR = Path(os.path.dirname(__file__)).parent.parent
DATA_ROOT = os.path.join(WORK_DIR, 'data')
FACTOR_ROOT = os.path.join(DATA_ROOT, 'factors')
RES_ROOT = os.path.join(WORK_DIR, 'results')

# path formats
FILENAME_KLINE = "{symbol}_{interval}_{date}.csv"  # symbol, interval, date
FILENAME_KLINE_CONCAT = "{symbol}_{interval}_{start_date}_{end_date}.csv"  # symbol, interval, start_date, end_date
FILENAME_FACTOR = "{factorname}_{interval}_{date}.csv"  # factor, interval, date
FILENAME_FACTOR_CONCAT = "{factorname}_{interval}_{start_date}_{end_date}.csv"  # factor, interval, start_date, end_date

# vtsymbol templates.
# for keys. such as the first part (the str before @) of the factor key
VTSYMBOL_KLINE = "kline_{interval}_{symbol}.{exchange}"  # interval, symbol, exchange
VTSYMBOL_TICK = "tick_{interval}_{symbol}.{exchange}"  # interval, symbol, exchange
# for factor_key. all symbols and exchanges needs to be calculated, so we don't care if they will be displayed in the key.
# forms factor_key and is displayed as column names in database
VTSYMBOL_FACTOR = "factor_{interval}_{factorname}"
# for datas. vnpy regards it as the combination of `symbol` and `exchange`, and rsplit it by '.'.
VTSYMBOL_BARDATA = "{symbol}.{exchange}"
VTSYMBOL_TICKDATA = "{symbol}.{exchange}"
VTSYMBOL_FACTORDATA = "{interval}_{symbol}_{factorname}.{exchange}"  # displayed in ticker column of database

# data related
TRAIN_START_DATE = '2020-10-01'
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2024-04-01'
TRAIN_START_DATE_for_test = '2022-01-01'
TRAIN_END_DATE_for_test = '2022-01-31'
TEST_START_DATE_for_test = '2022-02-01'
TEST_END_DATE_for_test = '2022-02-28'

# data related
BAR_OVERVIEW_FILENAME = "overview_bar.json"
TICK_OVERVIEW_FILENAME = "overview_tick.json"
FACTOR_OVERVIEW_FILENAME = "overview_factor.json"
