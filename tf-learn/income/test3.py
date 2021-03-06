#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tempfile
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir",
                    "D:/home/work/logistic_regression/census_income",
                    "Base directory for output models.")
flags.DEFINE_string("model_type", "wide", "valid model types:{'wide','deep', 'wide_n_deep'")
flags.DEFINE_integer("train_steps", 100, "Number of training steps.")
flags.DEFINE_string("train_data",
                    "D:/home/work/logistic_regression/census_income/census-income.data",
                    "Path to the training data.")
flags.DEFINE_string("test_data",
                    "D:/home/work/logistic_regression/census_income/census-income.test",
                    "path to the test data")

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]

CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


# build the estimator
def build_estimator(model_dir):
    # 离散分类别的
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["female", "male"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
    # 类别转换
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    wide_columns = [gender, native_country, education, occupation, workclass, relationship, age_buckets,
                    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4))]

    # embedding_column用来表示类别型的变量
    deep_columns = [tf.contrib.layers.embedding_column(workclass, dimension=8),
                    tf.contrib.layers.embedding_column(education, dimension=8),
                    tf.contrib.layers.embedding_column(gender, dimension=8),
                    tf.contrib.layers.embedding_column(relationship, dimension=8),
                    tf.contrib.layers.embedding_column(native_country, dimension=8),
                    tf.contrib.layers.embedding_column(occupation, dimension=8),
                    age, education_num, capital_gain, capital_loss, hours_per_week, ]

    if FLAGS.model_type == "wide":
        m = tflearn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)
    elif FLAGS.model_type == "deep":
        m = tflearn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100, 50])
    else:
        m = tflearn.DNNLinearCombinedClassifier(model_dir=model_dir, linear_feature_columns=wide_columns,
                                                         dnn_feature_columns=deep_columns, dnn_hidden_units=[100, 50])

    return m


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
    print("continuous:{}".format(continuous_cols))
    # CATEGORICAL_cols = {k: tf.SparseTensor(indices=)}
    categorical_cols = {
        k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
                           values=df[k].values,
                           dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    label = tf.constant(df[LABEL_COLUMN].values)

    return feature_cols, label


def train_and_eval():
    train_file_name = FLAGS.train_data
    test_file_name = FLAGS.test_data
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python"
    )
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python"
    )

    # drop Not a number elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    # convert >50 to 1
    df_train[LABEL_COLUMN] = (
        df_train["income_bracket"].apply(lambda x: ">50" in x).astype(int)
    )
    df_test[LABEL_COLUMN] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model dir = %s" % model_dir)

    m = build_estimator(model_dir)
    print(FLAGS.train_steps)
    m.fit(input_fn=lambda: input_fn(df_train),
          steps=FLAGS.train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    result = m.predict(input_fn=lambda: input_fn(df_test))
    for k in result:
        print("predict:{}".format(k))

def main(_):
    import time
    tic = time.time()
    train_and_eval()
    toc = time.time()
    print("model_type:{} cost:{}".format(FLAGS.model_type, 1000 * (toc - tic)))


if __name__ == "__main__":
    tf.app.run()
