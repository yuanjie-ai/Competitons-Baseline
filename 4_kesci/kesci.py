#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'kesci'
__author__ = 'JieYuan'
__mtime__ = '19-1-7'
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from tqdm import tqdm

nums = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cats = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'month', 'poutcome'
]

train = pd.read_csv('/home/yuanjie/desktop/train_set.csv')
test = pd.read_csv('/home/yuanjie/desktop/test_set.csv')

y = train.y
data = train.append(test).drop(['ID', 'y'], 1)

for feat in cats:
    data[feat] = LabelEncoder().fit_transform(data[feat])


def get_new_columns(name, aggs):
    l = []
    for k in aggs.keys():
        for agg in aggs[k]:
            l.append(name + '_' + k + '_' + agg)
    return l


########################################################
'''biubiu提供'''
for d in tqdm(cats):
    aggs = {}
    for s in cats:
        aggs[s] = ['count', 'nunique']
    for den in nums:
        aggs[den] = ['mean', 'max', 'min', 'std']
    _ = aggs.pop(d)
    temp = data.groupby(d).agg(aggs).reset_index()
    temp.columns = [d] + get_new_columns(d, aggs)
    data = pd.merge(data, temp, on=d, how='left')
########################################################


X, _X = data[:25317], data[25317:]

params = {'boosting_type': 'gbdt',
          'objective': 'binary',
          'max_depth': -1,
          'num_leaves': 127,
          'learning_rate': 0.01,
          'min_split_gain': 0.0,
          'min_child_weight': 0.001,
          'min_child_samples': 20,
          'subsample': 0.8,
          'subsample_freq': 8,
          'colsample_bytree': 0.8,
          'reg_alpha': 0.0,
          'reg_lambda': 0.0,
          'scale_pos_weight': 1,
          'random_state': None,
          'n_jobs': -1,
          'n_estimators': 419}

clf = LGBMClassifier(**params)

clf.fit(X, y)
rst = clf.predict_proba(_X)[:, 1]
test[['ID']].assign(pred=rst).to_csv('./baseline.csv', index=False)