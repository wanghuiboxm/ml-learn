# coding: utf-8

import tensorflow as tf
import numpy as np

# Prepare train data
# train_X = np.linspace(-1, 1, 100)
# train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

train_X = np.random.randn(100)
train_Y = 2*train_X + np.random.randn(100)*0.333 + 10
print("train_x:{}".format(train_X))
print("train_Y:{}".format(train_Y))

# Define the model
X = tf.placeholder(tf.float32)
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X * w - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(1000):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X: x, Y: y})
        print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
        epoch += 1
