#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re

"""基础特征处理"""


def deal(input_path, output_path):
    f = open(input_path, mode="r", encoding="utf-8")
    out = open(output_path, mode="w", encoding="utf-8")
    while True:
        line = f.readline()
        if len(line) <= 0:
            break
        arr = re.split("\t", line)
        arr = arr[:len(arr)-1]
        # 处理connection_type index=6 [初中及以下 硕士 高中 NULL 中专/技校 博士及以上 本科 大专]
        if arr[6] == "WIFI":
            arr[6] = '1'
        elif arr[6] == "MOBILE":
            arr[6] = '2'
        else:
            arr[6] = '0'
        # 处理学历 education  index=15
        education = arr[15]
        if education == "初中及以下":
            arr[15] = '1'
        elif education == "中专/技校":
            arr[15] = '2'
        elif education == "高中":
            arr[15] = '3'
        elif education == "大专":
            arr[15] = '4'
        elif education == "本科":
            arr[15] = '5'
        elif education == "硕士":
            arr[15] = '6'
        elif education == "博士及以上":
            arr[15] = '7'
        else:
            arr[15] = '0'
        out.write(','.join(arr)+"\n")
        # 处理安装的app
        print(','.join(arr))
    f.close()
    out.close()


if __name__ == '__main__':
    import sys

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # input_path = "C:/Users/wanghb/Downloads/ad-5000.txt"
    # output_path = "C:/Users/wanghb/Downloads/ad-5000.csv"
    deal(input_path, output_path)

