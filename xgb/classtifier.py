#! /usr/bin/env python
# -*- coding: utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn import datasets
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


'''
学习通过xgboost构造特征
'''
# 准备数据
X, y = datasets.make_hastie_10_2(random_state=0)
X = DataFrame(X)
y = DataFrame(y, columns=['label'])
y = y['label'].map({-1: 0, 1: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train)

# 定义模型
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
    'nthread': 2,
    'scale_pos_weight': 1,
    'lambda': 0,
    'seed': 27,
    'silent': 1,
    'eval_metric': 'logloss'
}
# xgboost自带接口
model_bst = xgb.train(params, d_train, 500, [(d_train, 'train'), (d_valid, 'valid')], early_stopping_rounds=500, verbose_eval=10)
assert isinstance(model_bst, xgb.Booster)

# sklearn接口
clf = xgb.XGBRegressor(
    max_depth=3,
    learning_rate=0.3,
    n_estimators=200,
    objective='binary:logistic',
    booster='gbtree',
    nthread=2,
    gamma=0.3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    seed=27
)
model_sklearn = clf.fit(X_train, y_train)

# 预测
y_bst = model_bst.predict(d_test)
y_sklean = model_sklearn.predict(X_test)
print("XGBoost自带接口 AUC Score: %f" % metrics.roc_auc_score(y_test, y_bst))
print("SKLearn接口 AUC Score: %f" % metrics.roc_auc_score(y_test, y_sklean))

# 生成两组新特征
# XGBoost
d_train_new = model_bst.predict(d_train, pred_leaf=True)
d_test_new = model_bst.predict(d_test, pred_leaf=True)

# sklearn
X_train_new = model_sklearn.apply(X_train)
X_test_new = model_sklearn.apply(X_test)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_lr = model_lr.predict(X_test)

model_xgboost_lr_raw = LogisticRegression(max_iter=1000)
model_xgboost_lr_raw.fit(np.concatenate((X_train, d_train_new), axis=1), y_train)
y_xgboost_lr_raw = model_xgboost_lr_raw.predict(np.concatenate((X_test, d_test_new), axis=1))

model_xgboost_lr = LogisticRegression(max_iter=1000)
model_xgboost_lr.fit(d_train_new, y_train)
y_xgboost_lr = model_xgboost_lr.predict(d_test_new)

model_sklearn_lr = LogisticRegression(max_iter=1000)
model_sklearn_lr.fit(X_train_new, y_train)
y_sklean_lr = model_sklearn_lr.predict(X_test_new)


print("LR AUC Score: %f" % metrics.roc_auc_score(y_test, y_lr))
print("sklearn+LR AUC Score: %f" % metrics.roc_auc_score(y_test, y_sklean_lr))
print("XGBoost+LR AUC Score: %f" % metrics.roc_auc_score(y_test, y_xgboost_lr))
print("XGBoost+LR+raw AUC Score: %f" % metrics.roc_auc_score(y_test, y_xgboost_lr_raw))
