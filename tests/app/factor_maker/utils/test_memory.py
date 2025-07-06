# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/tests/app/factor_maker/utils
# @File     : test_memory.py
# @Time     : 2025/7/6 22:21
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
import datetime
from unittest import TestCase
from vnpy.factor.memory import MemoryData
from vnpy.trader.constant import Interval
import datetime

# Jun 16 2024 21:36:06
sample_unix_s1 = 1718393273
sample_unix_s2 = 1718422073
sample_unix_s3 = 1718544966
sample_unix_ms1 = int(sample_unix_s1 * 1000)
sample_unix_ms2 = int(sample_unix_s2 * 1000)
sample_unix_ms3 = int(sample_unix_s3 * 1000)

sample_dt_s1 = datetime.datetime.fromtimestamp(sample_unix_s1)
sample_dt_s2 = datetime.datetime.fromtimestamp(sample_unix_s2)
sample_dt_s3 = datetime.datetime.fromtimestamp(sample_unix_s3)


class TestMemoryData(TestCase):
    def setUp(self):
        self.memory_data = MemoryData(data={'datetime': [sample_dt_s1,
                                                         sample_dt_s2,
                                                         sample_dt_s3], 'a': [4, 5, 6], 'b': [7, 8, 9]},
                                      interval=Interval.MINUTE)
        # print(type(self.memory_data))
        # print(self.memory_data)

    def test_append(self):
        new_data = MemoryData(data={'datetime': sample_dt_s3 + datetime.timedelta(minutes=1), 'a': 10, 'b': 11},
                              interval=Interval.MINUTE)
        self.memory_data.vstack_truncated(new_data)
        print(self.memory_data)

        self.assertEqual(len(self.memory_data['datetime']), 4)
        self.assertEqual(self.memory_data['a'][-1], 10)
        self.assertEqual(self.memory_data['b'][-1], 11)
