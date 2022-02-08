import numpy as np
import pandas as pd
import sys
import platform
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes, load_boston, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import *
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier

#1. 데이터
import joblib
x_train = joblib.load(open('../_save/m30_x_train_save.dat', 'rb'))
x_test = joblib.load(open('../_save/m30_x_test_save.dat', 'rb'))
y_train = joblib.load(open('../_save/m30_y_train_save.dat', 'rb'))
y_test = joblib.load(open('../_save/m30_y_test_save.dat', 'rb'))


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
# scaler = PolynomialFeatures()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# lgbm_wrapper = LGBMClassifier(n_estimators=100000)

# evals = [(x_test, y_test)]
# lgbm_wrapper.fit(x_train, y_train, early_stopping_rounds=5, eval_metric="multi_logloss",     #one of [None, 'micro', 'macro', 'weighted'].
#                  eval_set=evals, verbose=True)

import pickle
lgbm_wrapper = pickle.load(open('../_save/m23_cov_save1.dat', 'rb'))

preds = lgbm_wrapper.predict(x_test)
pred_proba = lgbm_wrapper.predict_proba(x_test)[:, 1]


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

# def get_clf_eval(y_test, pred=preds, pred_proba=pred_proba):
#     confusion = confusion_matrix( y_test, pred)
#     accuracy = accuracy_score(y_test , pred)
#     precision = precision_score(y_test , pred, average="macro")
#     recall = recall_score(y_test , pred, average="macro")
#     f1 = f1_score(y_test,pred, average="macro")
#     # ROC-AUC 추가 
#     roc_auc = roc_auc_score(y_test, pred_proba)
#     print('오차 행렬')
#     print(confusion)
#     # ROC-AUC print 추가
#     print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
#     F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    
# get_clf_eval(y_test, preds, pred_proba)
f1 = f1_score(y_test,preds, average="macro")
print(f1)