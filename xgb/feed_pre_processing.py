#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

path = "../data/feed.data"
raw_data = pd.read_csv(path, names=range(358))
x_data = raw_data.drop(labels=[0, 1, 2], axis=1)
y_data = pd.DataFrame(raw_data[2][0:100])
start = time.time()
print("start save x_data:")
x_data.to_csv("../data/test.x.csv", index=False, header=False)
print("end sava x_data cost:", time.time()-start)
print("start save y_data")
start = time.time()
print(y_data.to_csv("../data/test.y.csv", index=False, header=False))
print("end save y_data cost:", time.time()-start)
