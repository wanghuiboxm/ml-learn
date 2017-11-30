#! /usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.contrib import learn, layers
import tensorflow as tf
import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir",
                    "/home/work/workspace/wanghuibo/miui_ad/module",
                    "Base directory for output models.")
flags.DEFINE_string("model_type", "wide", "valid model types:{'wide','deep', 'wide_n_deep'")
flags.DEFINE_integer("train_steps", 100, "Number of training steps.")
flags.DEFINE_string("train_data",
                    "/home/work/workspace/wanghuibo/miui_ad/ad_test.csv",
                    "Path to the training data.")

COLUMNS = ["label", "instance_id", "click_time", "ad_id", "user_id", "position_id", "connection_type", "miui_version",
           "ip", "adroid_version", "advertiser_id", "compaign_id", "app_id", "age", "gender", "education", "province",
           "city", "device_info"]
FEATURES = ["instance_id", "click_time", "ad_id", "user_id", "position_id", "connection_type", "miui_version",
            "ip", "adroid_version", "advertiser_id", "compaign_id", "app_id", "age", "gender", "education", "province",
            "city", "device_info"]
LABEL = "label"


def input_fn(df):
    feature_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in FEATURES}
    label = tf.constant(value=df[LABEL].values, shape=[df[LABEL].size, 1])
    return feature_cols, label


def build_model():
    feature_cols = [layers.real_valued_column(k) for k in FEATURES]

    if FLAGS.model_type == "wide":
        m = learn.LinearRegressor(feature_columns=feature_cols, model_dir=FLAGS.model_dir)
        # m = learn.LogisticRegressor(model_dir=FLAGS.model_dir)
    elif FLAGS.model_type == "deep":
        m = learn.DNNLinearCombinedRegressor(model_dir=FLAGS.model_dir, feature_columns=feature_cols, hidden_units=[100, 50])
    else:
        m = learn.DNNLinearCombinedRegressor(model_dir=FLAGS.model_dir, linear_feature_columns=feature_cols,
                                             dnn_feature_columns=feature_cols, dnn_hidden_units=[100, 50])

    return m

def my_model(features, labels):
    logist = layers.fully_connected(inputs=features, num_outputs=1, activation_fn=tf.nn.sigmoid)
    loss = tf.losses.sigmoid_cross_entropy(labels, logist)
    train_op = layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer='sgd', learning_rate=0.01)

    return logist, loss, train_op

def main(_):
    data_set = pd.read_csv(FLAGS.train_data, skipinitialspace=True, names=COLUMNS)
    predict_set = data_set

    print("***********train*************")
    m = build_model()
    # m.fit(input_fn=lambda: input_fn(train_set), steps=FLAGS.train_steps)

    print("************evaluate*************")
    # evaluate_result = m.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    # for k in evaluate_result:
    #     print("{}: {}".format(k, evaluate_result[k]))

    print("************predict*************")
    result = m.predict_scores(input_fn=lambda: input_fn(predict_set))
    score = []
    import math
    for k in result:
        k = 1/(1+math.exp(-k))
        score.append(k)
        # print("predict:{}".format(k))
    # score.sort()
    # print(score)
    write("/home/work/workspace/wanghuibo/miui_ad/ad-pre.csv", predict_set, score)


def write(output_path, dataset, score):
    print("dateset:{}, score:{}".format(len(dataset), len(score)))
    f = open(output_path, mode='w', encoding='utf-8')
    dataset['score'] = score
    dataset = dataset.sort_values(by='instance_id')
    result = pd.DataFrame(dataset['instance_id'])
    result['score'] = dataset['score']
    print(dataset)
    for k in result.values:
        f.write(",".join([str(int(k[0])), str(k[1])])+"\n")
    f.close()

if __name__ == '__main__':
    tf.app.run()
