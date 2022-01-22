import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM
import os
import sys
import platform
import random
import math
from typing import List, Dict,Tuple 
import sklearn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
from catboost import Pool, CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# print(f"-os:{platform.platform()}") #os:Windows-10-10.0.19043-SP0
# print(f"-python:{sys.version}") #python:3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
# print(f"-pandas:{pd.__version__}") #pandas:1.3.4
# print(f"-numpy:{np.__version__}") #numpy:1.20.3
# print(f"-sklearn: {sklearn.__version__}") #sklearn: 1.0.1

path = "../_data/dacon/jobcare_data/"

train_data=pd.read_csv(path+'train.csv')

test_data=pd.read_csv(path+'test.csv')

d_code=pd.read_csv(path+'속성_D_코드.csv')

h_code=pd.read_csv(path+'속성_H_코드.csv')

l_code=pd.read_csv(path+'속성_L_코드.csv')

submit_file = pd.read_csv(path + "sample_submission.csv")

# print(train_data.info)

# print(test_data.info)

# print(d_code.info)

# print(h_code.info)

# print(l_code.info)

from datetime import datetime

# print(train_data.shape) (501951, 35)
# print(test_data.shape) (46404, 34)

#print(train_data.head())

#    id  d_l_match_yn  d_m_match_yn  d_s_match_yn  h_l_match_yn  h_m_match_yn  ...  contents_attribute_e  contents_attribute_h  person_rn  contents_rn     contents_open_dt  target
# 0   0          True          True          True         False         False  ...                     4                   139     618822       354805  2020-01-17 12:09:36       1
# 1   1         False         False         False          True          True  ...                     4                   133     571659       346213  2020-06-18 17:48:52       0
# 2   2         False         False         False          True         False  ...                     4                    53     399816       206408  2020-07-08 20:00:10       0
# 3   3         False         False         False          True         False  ...                     3                    74     827967       572323  2020-01-13 18:09:34       0
# 4   4          True          True          True         False         False  ...                     4                    74     831614       573899  2020-03-09 20:39:22       0

#print(test_data.head())
# [5 rows x 35 columns]

#  id  d_l_match_yn  d_m_match_yn  d_s_match_yn  h_l_match_yn  h_m_match_yn  ...  contents_attribute_m  contents_attribute_e  contents_attribute_h  person_rn  contents_rn     contents_open_dt
# 0   0          True         False         False          True          True  ...                     1                     5                   263     393790       236865  2020-12-01 02:24:18   
# 1   1         False         False         False          True         False  ...                     1                     4                   263     394058       236572  2020-12-17 05:42:53   
# 2   2          True         False         False          True          True  ...                     3                     4                   177    1002061       704612  2020-12-10 23:33:41   
# 3   3          True         False         False          True          True  ...                     5                     3                   177    1000813       704652  2020-12-03 19:44:55   
# 4   4          True         False         False          True         False  ...                     1                     4                   177     111146       704413  2020-12-11 21:24:34   

# [5 rows x 34 columns]

train = train_data.drop(['id', 'contents_open_dt'], axis=1) 
test = test_data.drop(['id', 'contents_open_dt'], axis=1)

xgb_model=XGBClassifier(booster='gbtree', min_child_weight=10, max_depth=8,gamma =0,nthread =4, learning_rate=0.1, colsample_bytree=0.8,colsample_bylevel=0.9,n_estimators =400)

x=train.iloc[:,:-1]
y=train.iloc[:,-1]


xgb_model.fit(x,y, eval_metric='error')

preds=xgb_model.predict(test)

submit_file['target']=preds

print(preds)

submit_file.to_csv(path + 'sookook2.csv',index=False)








