#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib import learn, layers
import pandas as pd
import numpy as np
import random

COLUMNS = ["x", "y"]
FEATURES = ["x"]
LABEL = "y"

train_set = pd.DataFrame(data=np.random.rand(100, 1), columns=FEATURES)
x = train_set
# y = [random.randrange(2) for k in range(100)]

train_set[LABEL] = train_set['x']*2+np.random.randn(100)*0.333 + 10
print("train_set:{}".format(train_set))
print("x:{}".format(x))
print("y:{}".format(train_set[0]*2+np.random.randn(100)*0.333 + 10))


def input_fn(df):
    feature_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in FEATURES}
    label = tf.constant(value=df[LABEL].values)
    return dict(feature_cols), label


def main(_):
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    m = tflearn.LinearRegressor(feature_columns=feature_cols, model_dir="/home/work/test")
    m.fit(input_fn=lambda: input_fn(train_set), steps=10)

    results = m.evaluate(input_fn=lambda: input_fn(train_set), steps=1)

    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

if __name__ == '__main__':
    tf.app.run()

