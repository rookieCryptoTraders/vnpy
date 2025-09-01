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


# factor maker. fixme: comments are not specified
EVENT_BAR_FACTOR = "eBarFactor."
EVENT_FACTOR = "eFactor." # insert 1 line data into database
EVENT_FACTOR_FILLING = "eFactorFilling"  # insert multiline data. this will trigger the data recorder to record the factor data and factor maker calucation
EVENT_FACTORMAKER_CALCULATE = "eFactormakerCalculate"
EVENT_FACTORMAKER_CALCULATE_ALL = "eFactormakerCalculateAll"
EVENT_FACTORMAKER_BAR_READY = "eFactormakerBarReady"  # bar data is ready
EVENT_FACTORMAKER_FACTOR_READY = "eFactormakerFactorReady"  # factor data is ready
EVENT_FACTORMAKER_ALL_READY = "eFactormakerAllReady"  # all data is ready, this will trigger the factor maker to calculate factors
EVENT_FACTOR_BAR_UPDATE = "eFactorBarUpdate"

# data manager
EVENT_HISTORY_DATA_REQUEST = "eHistoryDataRequest"
EVENT_DATAMANAGER_LOAD_BAR_REQUEST = "eDataManagerLoadBarRequest"  # DataManager simply load bar data from the database
EVENT_DATAMANAGER_LOAD_BAR_RESPONSE = "eDataManagerLoadBarResponse"  # data receiver app receives responded bar data
EVENT_DATAMANAGER_LOAD_FACTOR_REQUEST = "eDataManagerLoadFactorRequest"  # DataManager simply load factor data from the database
EVENT_DATAMANAGER_LOAD_FACTOR_RESPONSE = "eDataManagerLoadFactorResponse"  # data receiver app receives responded factor data

# EVENT_FACTORMAKER_LOG = "eFactorMakerLog"   # tell the LogEngine to log the message
EVENT_FACTORMAKER = "eFactorMaker"
