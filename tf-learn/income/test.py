#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]


def get_data():
    train_file_name = "D:/home/work/logistic_regression/census_income/census-income.data"
    test_file_name = "D:/home/work/logistic_regression/census_income/census-income.data"

    # 用 pandas 读入数据
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")
    print("df_train:{}, keys:{}, shape:{} \n".format(df_train.tail(10), df_train.keys(), df_train.shape))
    print("df_test:{}, keys:{} \n".format(df_test.tail(10), df_test.keys()))

    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                       keys=["female", "male"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket(
        "education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
        "relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
        "workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000)

    # 为连续的列元素设置一个实值列
    age = tf.contrib.layers.real_valued_column("age")
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    # 为了更好的学习规律，收入是与年龄阶段有关的，因此需要把连续的数值划分
    # 成一段一段的区间来表示收入
    age_buckets = tf.contrib.layers.bucketized_column(age,
                                                      boundaries=[
                                                          18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ])

    # 上面所说的模型，
    # 这个为 wide 模型
    wide_columns = [gender, native_country, education, occupation, workclass,
                    relationship, age_buckets,
                    tf.contrib.layers.crossed_column([education, occupation],
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column(
                        [age_buckets, education, occupation],
                        hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([native_country, occupation],
                                                     hash_bucket_size=int(1e4))]
    for k in wide_columns:
        print(k)

if __name__ == '__main__':
    get_data()