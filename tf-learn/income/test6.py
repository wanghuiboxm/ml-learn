#! /usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.contrib import learn, layers
import tensorflow as tf
import pandas as pd
import numpy as np

COLUMNS = ["x", "vy"]
FEATURES = ["x"]
LABEL = "vy"


def input_fn(df):
    feature_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in FEATURES}
    label = tf.constant(value=df[LABEL].values, shape=[df[LABEL].size, 1])
    return feature_cols, label


def main(_):
    train_set = pd.DataFrame(data=np.random.rand(100, 1), columns=FEATURES)
    train_set[LABEL] = train_set['x']*2+np.random.randn(100)*0.333 + 10
    print("train_set:{}".format(train_set))

    feature_cols = [layers.real_valued_column(k) for k in FEATURES]
    m = learn.LinearClassifier(feature_columns=feature_cols, model_dir="D:/home/work/test")
    train_result = m.fit(input_fn=lambda: input_fn(train_set), steps=10)
    print("***********train*************")
    # for k in train_result:
    #     print("{}: {}".format(k, train_result[k]))
    evaluate_result = m.evaluate(input_fn=lambda: input_fn(train_set), steps=1)
    print("************evaluate*************")
    for k in evaluate_result:
        print("{}: {}".format(k, evaluate_result[k]))
    print("************predict*************")
    # result = m.predict(input_fn=lambda: input_fn(train_set))
    result = m.predict(input_fn=lambda: input_fn(train_set))
    for k in result:
        print("{}".format(k))

if __name__ == '__main__':
    tf.app.run()