#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def get_data(path):
    data_set = pd.read_csv(path, header=None)
    print(data_set)
    x_data = data_set.drop(labels=[0, 1, 2], axis=1)
    y_data = data_set[2]
    # return np.array(x_data), np.array(y_data)
    return x_data, y_data

if __name__ == '__main__':
    x_data, y_data = get_data('D:/Workspaces/ml-learn/data/feed.data')
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_test, label=y_test)
    d_test = xgb.DMatrix(X_test)
    params = {
        'eta': 0.3,
        'max_depth': 3,
        'min_child_weight': 1,
        'gama': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'nthread': 20,
        'scale_pos_weight': 1,
        'lambda': 0,
        'seed': 27,
        'silent': 1,
        'eval_metric': 'logloss'
    }
    # xgboost自带接口
    model_bst = xgb.train(params, d_train, 200, [(d_train, 'train'), (d_valid, 'valid')], early_stopping_rounds=500, verbose_eval=10)
    assert isinstance(model_bst, xgb.Booster)

    y_bst = model_bst.predict(d_test)
    print("XGBoost自带接口 AUC Score: %f" % metrics.roc_auc_score(y_test, y_bst))


    d_train_new = model_bst.predict(d_train, pred_leaf=True)
    d_test_new = model_bst.predict(d_test, pred_leaf=True)

    model_xgboost_lr = LogisticRegression(max_iter=1000)
    model_xgboost_lr.fit(d_train_new, y_train)
    y_xgboost_lr = model_xgboost_lr.predict(d_test_new)

    # model_lr = LogisticRegression(max_iter=1000)
    # model_lr.fit(X_train, y_train)
    # y_lr = model_lr.predict(X_test)

    # print("LR AUC Score: %f" % metrics.roc_auc_score(y_test, y_lr))
    print("XGBoost+LR AUC Score: %f" % metrics.roc_auc_score(y_test, y_xgboost_lr))
