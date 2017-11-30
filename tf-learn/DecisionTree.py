#! /usr/bin/env python
# -*- coding: utf-8 -*-
import operator

import xgboost as xgb

# dtrain = xgb.DMatrix("../data/agaricus.txt.train")
# dtest = xgb.DMatrix("../data/agaricus.txt.test")

dtrain = xgb.DMatrix("../data/a4a")
dtest = xgb.DMatrix("../data/a4a.t")


param = {
    'max_depth': 10,
    "eta": 0.01,
    'silent': 0,
    'objective': 'multi:softmax',
    'num_class': 2,
    'booster': 'gbtree','subsample':0.8,'colsample_bytree':0.8,'seed':27}
num_round = 20

bst = xgb.train(param, dtest, num_round)
assert isinstance(bst, xgb.Booster)
importace = bst.get_fscore()
importace = sorted(importace.items(), key=operator.itemgetter(1))
print(importace)

preds = bst.predict(dtrain)

# for v in preds:
#     print(v)
