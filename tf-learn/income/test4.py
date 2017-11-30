#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import random

train_set = pd.DataFrame(data=np.random.rand(100, 4))
y = [random.randrange(2) for k in range(100)]
train_set[4] = pd.DataFrame(data=y)
print(train_set)
# 列切片
# train_x = train_set.ix[:, 0:3]
# print(train_x)
# train_y = train_set.ix[4]
# print(train_y)


def input_fn(df):
    continue_cols = {str(k): tf.constant(df[k].values, shape=[df[k].size, 1]) for k in range(4)}
    label = tf.constant(value=df[4].values)
    return dict(continue_cols), label


def main(_):
    # print("continue_cols:{}".format(continue_cols))
    cols = [tf.contrib.layers.real_valued_column(str(k)) for k in range(4)]
    # print("cols:{}".format(cols))
    m = tf.contrib.learn.LinearClassifier(model_dir="/home/work/test", feature_columns=cols)
    m.fit(input_fn=lambda: input_fn(train_set), steps=10)

    results = m.evaluate(input_fn=lambda: input_fn(train_set), steps=1)


    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

if __name__ == '__main__':
    tf.app.run()


