#! /usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.contrib import keras
import numpy as np
import pandas as pd
import tensorflow as tf


def test1():
    x_train = np.linspace(-5, 5, 100)
    x_train = x_train.reshape(len(x_train), 1)
    y_train = np.sin(x_train)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=2, input_dim=1))
    # model.add(keras.layers.Dense(units=1, activation='softmax', input_dim=2))

    print(model.summary())

    # categorical_crossentropy
    # keras.losses.mean_squared_logarithmic_error()
    model.compile(loss='mean_squared_logarithmic_error', optimizer=keras.optimizers.SGD(0.01))
    hist = model.fit(x_train, y_train, epochs=100, shuffle=True)
    print("train:{}".format(hist))

    out = model.predict(x_train, batch_size=1)

    print("predict:{}".format(out))

def test2():
    train_set = pd.read_csv('C:/Users/wanghb/Downloads/ad-5000.csv', names=range(19))
    train_x = pd.DataFrame(train_set.ix[:, 1:18])
    train_x = np.array(train_x.values)
    # train_x = train_x.reshape(len(train_x), 18)
    train_y = train_set[0]
    train_y = np.array(train_y)
    # train_y = keras.utils.to_categorical(train_y, num_classes=2)
    # train_y = train_y.reshape(len(train_y), 1)
    test_set = pd.read_csv('C:/Users/wanghb/Downloads/ad-5000.csv', names=range(19))
    test_x = pd.DataFrame(test_set.ix[:, 1:18])
    test_x = np.array(test_x.values)
    test_y = np.array(test_set[0].values)
    # test_y = keras.utils.to_categorical(test_y, num_classes=2)

    print("x:{}".format(train_x))
    print("y:{}".format(train_y))
    print("x len:{}, y len:{}".format(len(train_x), len(train_y)))

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=1, activation='relu', input_dim=18))
    model.add(keras.layers.Dense(1, activation='sigmoid'))


    print(model.summary())
    # keras.losses.mean_squared_logarithmic_error()
    model.compile(loss='mean_squared_logarithmic_error', optimizer=keras.optimizers.SGD(0.01), metrics=['accuracy'])
    hist = model.fit(train_x, train_y, epochs=10, shuffle=True, batch_size=1000, validation_split=0.1)
    model.save(filepath='/home/work/test/keras.model', overwrite=True)
    print("train:{}".format(hist))
    # model = keras.models.load_model(filepath='/home/work/test/keras.model')
    out = model.predict(test_x, batch_size=1)

    for k in out:
        print("predict:{}".format(k))

def test3():
    data_set = pd.read_csv('C:/Users/wanghb/Downloads/ad-5000.csv', names=range(19))
    train_x = pd.DataFrame(data_set.ix[:, 1:18])
    train_x = np.array(train_x.values)
    train_x = train_x.reshape(len(train_x), 18)
    train_y = data_set[0]
    train_y = np.array(train_y)

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 18])
    y = tf.placeholder(tf.float32, [None, 2])

    # 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
    W = tf.Variable(tf.zeros([18, 2]))
    b = tf.Variable(tf.zeros([1]))
    prediction = tf.nn.softmax(tf.matmul(x, W)+b)

    # 二次代价函数
    # square是求平方
    # reduce_mean是求平均值
    loss = tf.reduce_mean(tf.square(y-prediction))

    # 使用梯度下降法来最小化loss，学习率是0.2
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast是进行数据格式转换，把布尔型转为float32类型

    batch_size = len(data_set)//100
    with tf.Session() as sess:
        # 执行初始化
        sess.run(init)
        # 迭代21个周期
        for epoch in range(21):
            avg_loss = 0
            # 每个周期迭代n_batch个batch，每个batch为100
            for batch in range(100):
                # 通过feed喂到模型中进行训练
                _, c = sess.run([train_step, loss], feed_dict={x: train_x, y: train_y})
                avg_loss += c
            print("Epoch:{} loss:{}".format(epoch, avg_loss/batch_size))
        # 计算准确率
        acc = sess.run(accuracy, feed_dict={x: train_x, y: train_y})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

if __name__ == '__main__':
    test3()