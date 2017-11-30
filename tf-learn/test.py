#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def sum():
    state = tf.Variable(0, name="counter")
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        print("init state value=", sess.run(state))
        for i in range(3):
            result = sess.run(update)
            print("update state value=", result)


if __name__ == '__main__':
    sum()
