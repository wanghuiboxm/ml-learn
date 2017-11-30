#! /usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import numpy as np
import pandas as pd
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def input_fn(filepath):
    data_set = pd.read_csv(filepath, names=range(19))
    x = data_set.drop(axis=1, labels=[0])  # , 1, 4, 8, 9, 10, 11, 12
    x = np.array(x)
    y = data_set[0]
    y = np.array(y)
    instance_id = data_set[1]
    return x, y, instance_id


def train(model_file, train_file):
    print("*********read data set from:{}****************".format(train_file))
    train_x, train_y, instance_id = input_fn(train_file)

    # model = keras.models.Model(inputs=)
    input_all = keras.layers.Input(shape=(train_x.shape[1],), name='input_layer')
    fl = keras.layers.BatchNormalization()(input_all)
    fl = keras.layers.Dense(256)(fl)
    fl = keras.layers.PReLU()(fl)
    fl = keras.layers.Dropout(0.25)(fl)

    fl = keras.layers.BatchNormalization()(fl)
    fl = keras.layers.Dense(128)(fl)
    fl = keras.layers.PReLU()(fl)
    fl = keras.layers.Dropout(0.25)(fl)

    fl = keras.layers.BatchNormalization()(fl)
    fl = keras.layers.Dense(64)(fl)
    fl = keras.layers.LeakyReLU()(fl)
    fl = keras.layers.Dropout(0.25)(fl)

    fl = keras.layers.BatchNormalization()(fl)
    output_all = keras.layers.Dense(1, activation='sigmoid', name='output_layer')(fl)
    model = keras.models.Model(input_all, output_all)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    hist = model.fit(train_x, train_y, 10240, 200, shuffle=True, validation_split=0.1)
    print("***********end train, save model:{}***************".format(model_file))
    model.save(model_file, overwrite=True)
    print("***********save model success************")


def predict(model_file, test_file, out_file):
    print("************load model*************")
    # model = keras.models.Sequential()
    model = keras.models.load_model(model_file)
    test_x, test_y, instance_id = input_fn(test_file)
    score = model.predict(test_x)

    print("**************save predict result:{}***************".format(out_file))
    n_score = []
    for k in score:
        n_score.append(k[0])
    result = pd.DataFrame(instance_id.values, columns=['instance_id'])
    result['proba'] = n_score
    # result['label'] = test_y
    result = result.sort_values(by='instance_id')
    print(result)
    result.to_csv(out_file, index=False)

if __name__ == '__main__':
    model_file = '/home/work/test/keras.model'
    train_file = 'C:/Users/wanghb/Downloads/ad-5000.csv'
    test_file = 'C:/Users/wanghb/Downloads/ad-5000.csv'
    save_predict_file = '/home/work/test/ad-predict-result.csv'

    # model_file = '/home/work/workspace/wanghuibo/miui_ad/v2/model'
    # train_file = '/home/work/workspace/wanghuibo/miui_ad/ad_train.csv'
    # test_file = '/home/work/workspace/wanghuibo/miui_ad/ad_test.csv'
    # save_predict_file = '/home/work/workspace/wanghuibo/miui_ad/v2/pre-result.csv'

    train(model_file, train_file)
    predict(model_file, test_file, save_predict_file)
