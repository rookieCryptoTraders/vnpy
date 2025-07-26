"""
Event type string used in the trading platform.
"""

from vnpy.event import EVENT_TIMER  # noqa

EVENT_TICK = "eTick."
EVENT_TRADE = "eTrade."
EVENT_ORDER = "eOrder."
EVENT_POSITION = "ePosition."
EVENT_ACCOUNT = "eAccount."
EVENT_QUOTE = "eQuote."
EVENT_CONTRACT = "eContract."
EVENT_LOG = "eLog"
EVENT_TELEGRAM = "eTelegram."


# ================================= apps =================================
# name rule: EVENT_<APP_NAME>_<ACTION/EVENT_NAME>
# ========================================================================
# data recorder
# EVENT_RECORDER_LOG = "eRecorderLog"  # tell the LogEngine to log the message
EVENT_BAR = "eBar."  # bar is arrived, recording/calculating are needed
EVENT_BAR_FILLING = "eBarFilling."  # datamanager downloaded bar data, filling the bar data to the database

EVENT_RECORDER_UPDATE = "eRecorderUpdate"  # signal to indicate the recorder has been updated
EVENT_RECORDER_RECORD = "eRecorderRecord"  # signal to trigger the recorder to record data
EVENT_RECORDER_EXCEPTION = "eRecorderException"  # signal to indicate the recorder has an exception

EVENT_DATAMANAGER_LOAD_BAR = "eDataManagerLoadBar"  # DataManager simply load bar data from the database
EVENT_DATAMANAGER_LOAD_FACTOR = "eDataManagerLoadFactor"  # DataManager simply load factor data from the database
EVENT_DATAMANAGER_LOAD_BAR_RESPONSE = "eDataManagerLoadBarResponse"  # data receiver app receives responded bar data
EVENT_DATAMANAGER_LOAD_FACTOR_RESPONSE = "eDataManagerLoadFactorResponse"  # data receiver app receives responded factor data

# factor maker. fixme: comments are not specified
EVENT_BAR_FACTOR = "eBarFactor."
EVENT_FACTOR = "eFactor."
EVENT_FACTOR_CALCULATE = "eFactorCalculate"
EVENT_FACTOR_FILLING = "eFactorFilling"  # factor data is filled into the database, this will trigger the data recorder to record the factor data and factor maker calucation
EVENT_FACTOR_BAR_UPDATE = "eFactorBarUpdate"
EVENT_HISTORY_DATA_REQUEST = "eHistoryDataRequest"

# EVENT_FACTORMAKER_LOG = "eFactorMakerLog"   # tell the LogEngine to log the message
EVENT_FACTORMAKER = "eFactorMaker"
