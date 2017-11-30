#! /usr/bin/env python
# -*- coding: utf-8 -*-

from random import normalvariate
import numpy as np
from math import exp

from numpy import shape

# 随机梯度求解
from numpy.core.umath import multiply


def sigmodid(v):
    return 1.0/(1+exp(-v))


def stoc_grad_ascend(data_matrix, class_labels, k, iter):

    # m是样本数 n是每个样本的特征数
    m, n = shape(data_matrix)
    print("m=%d,n=%d" % (m, n))
    alpha = 0.01
    # 初始化参数
    w = np.zeros((n, 1))
    print("w=", w)
    w_0 = 0.
    v = normalvariate(0, 0.2) * np.ones((n, k))
    print("v=", v)

    for it in range(iter):
        print("sgd iter:%d" % it)
        for x in range(m):
            inter_1 = data_matrix[x] * v
            inter_2 = multiply(data_matrix[x], data_matrix[x]) * multiply(v, v)
            # 完成求交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2)/2.

            p = w_0 + data_matrix[x]*w + interaction

            loss = sigmodid(class_labels[x]*p[0, 0]) - 1
            print("loss=", loss)

            w_0 -= alpha*loss*class_labels[x]

            for i in range(n):
                if data_matrix[x, i] != 0:
                    w[i, 0] -= alpha*loss*class_labels[x]*data_matrix[x, i]
                    for j in range(k):
                        v[i, j] -= alpha*loss*class_labels[x]*(data_matrix[x, i]*inter_1[0, j]-v[i, j]*data_matrix[x,i])

        return w_0, w, v


if __name__ == '__main__':
    stoc_grad_ascend(np.random.rand(100, 10), np.ones((100, 1)), 10, 10)

