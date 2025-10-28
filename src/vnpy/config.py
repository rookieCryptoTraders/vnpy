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
import re

# global settings
os.environ['TZ'] = 'UTC'

# path related
# Adjusted by Gemini to point to the current workspace directory.
WORK_DIR = Path(os.getcwd())
DATA_ROOT = os.path.join(WORK_DIR, 'data')
FACTOR_ROOT = os.path.join(DATA_ROOT, 'factors')
RES_ROOT = os.path.join(WORK_DIR, 'results')

# path formats
FILENAME_KLINE = "{symbol}_{interval}_{date}.csv"  # symbol, interval, date
FILENAME_KLINE_CONCAT = "{symbol}_{interval}_{start_date}_{end_date}.csv"  # symbol, interval, start_date, end_date
FILENAME_FACTOR = "{factorname}_{interval}_{date}.csv"  # factor, interval, date
FILENAME_FACTOR_CONCAT = "{factorname}_{interval}_{start_date}_{end_date}.csv"  # factor, interval, start_date, end_date

# vtsymbol templates.
VTSYMBOL = "{symbol}.{exchange}"  # symbol, exchange
# for keys. such as the first part (the str before @) of the factor key
VTSYMBOL_KLINE = "kline_{interval}_{symbol}.{exchange}"  # interval, symbol, exchange
VTSYMBOL_TICK = "tick_{interval}_{symbol}.{exchange}"  # interval, symbol, exchange
# for factor_key. all symbols and exchanges needs to be calculated, so we don't care if they will be displayed in the key.
# forms factor_key and is displayed as column names in database
VTSYMBOL_FACTOR = "factor_{interval}_{factorname}"
FACTOR_KEY_TEMPLATE = "{factorname}@{version}#{config_hash}"
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

# overview related
BAR_OVERVIEW_FILENAME = "overview_bar.json"
TICK_OVERVIEW_FILENAME = "overview_tick.json"
FACTOR_OVERVIEW_FILENAME = "overview_factor.json"
BAR_OVERVIEW_KEY = "overview_bar_{interval}_{symbol}.{exchange}"
TICK_OVERVIEW_KEY = "overview_tick_{interval}_{symbol}.{exchange}"
FACTOR_OVERVIEW_KEY = "overview_factor_{interval}_{symbol}.{exchange}|{factor_key}"


def match_format_string(format_str, s):
    """Match s against the given format string, return dict of matches.

    We assume all of the arguments in format string are named keyword arguments (i.e. no {} or
    {:0.2f}). We also assume that all chars are allowed in each keyword argument, so separators
    need to be present which aren't present in the keyword arguments (i.e. '{one}{two}' won't work
    reliably as a format string but '{one}-{two}' will if the hyphen isn't used in {one} or {two}).

    We raise if the format string does not match s.

    Example:
    fs = '{test}-{flight}-{go}'
    s = fs.format('first', 'second', 'third')
    match_format_string(fs, s) -> {'test': 'first', 'flight': 'second', 'go': 'third'}
    """

    # First split on any keyword arguments, note that the names of keyword arguments will be in the
    # 1st, 3rd, ... positions in this list
    tokens = re.split(r'\{(.*?)\}', format_str)
    keywords = tokens[1::2]

    # Now replace keyword arguments with named groups matching them. We also escape between keyword
    # arguments so we support meta-characters there. Re-join tokens to form our regexp pattern
    tokens[1::2] = list(map(u'(?P<{}>.*?)'.format, keywords[:-1]))+[u'(?P<{}>.*)'.format(keywords[-1])]
    tokens[0::2] = map(re.escape, tokens[0::2])
    pattern = ''.join(tokens)

    # Use our pattern to match the given string, raise if it doesn't match
    matches = re.match(pattern, s)
    if not matches:
        raise Exception("Format string did not match")

    # Return a dict with all of our keywords and their values
    return {x: matches.group(x) for x in keywords}
