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
from sklearn.metrics import accuracy_score, f1_score
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
from catboost import Pool, CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
import time
from datetime import datetime

warnings.filterwarnings('ignore')

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    
    print("1사분위 : ", quartile_1)
    print("q2: ", q2)
    print("3사분위: ", quartile_3)
    iqr = quartile_3-quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 +(iqr*1.5)
    return np.where((data_out>upper_bound)|  #또는
                    (data_out<lower_bound))


seed=66
# def NMAE(true, pred):
#     mae = np.mean(np.abs(true-pred))
#     score = mae / np.mean(np.abs(true))
#     return score

print(f"-os:{platform.platform()}") #os:Windows-10-10.0.19043-SP0
print(f"-python:{sys.version}") #python:3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
print(f"-pandas:{pd.__version__}") #pandas:1.3.4
print(f"-numpy:{np.__version__}") #numpy:1.20.3
print(f"-sklearn: {sklearn.__version__}") #sklearn: 1.0.1

path = "../_data/dacon/housing/"

datasets=pd.read_csv(path+'train.csv')

test=pd.read_csv(path+'test.csv')

submit_file = pd.read_csv(path+'sample_Submission.csv')

datasets.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

print(datasets.shape) #(1350, 14)
print(test.shape) #(1350, 13)
 
datasets.head()

print('=========================중복값 제거===========================')

print("제거 전 :", datasets.shape) #(1350, 14)
datasets=datasets.drop_duplicates()
print("제거 후 :", datasets.shape) #(1349, 14)

print('=========================중복값 제거 완료===========================')


print('=========================이상치 탐지와 제거===========================')

outliers_loc = outliers(datasets['Garage Yr Blt'])
print(outliers_loc)
print(datasets.loc[[255], 'Garage Yr Blt'])
datasets.drop(datasets[datasets['Garage Yr Blt']==2207].index, inplace=True)
print(datasets.shape) #(1348, 14)
# datasets.loc[254, 'Garage Yr Blt'] = 2007

print('=========================이상치 제거 완료===========================')

qual_cols=datasets.dtypes[datasets.dtypes == np.object].index
def label_encoder(df_, qual_cols):
    df= df_.copy()
    mapping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
    for col in qual_cols:
        df[col]=df[col].map(mapping)
    return df

datasets = label_encoder(datasets, qual_cols)
test = label_encoder(test,qual_cols)

# print(datasets.head())

#    Overall Qual  Gr Liv Area  Exter Qual  Garage Cars  Garage Area  Kitchen Qual  Total Bsmt SF  1st Flr SF  Bsmt Qual  Full Bath  Year Built  Year Remod/Add  Garage Yr Blt  target
# 0            10         2392           5            3          968             5           2392        2392          5          2        2003            2003           2003  386250
# 1             7         1352           4            2          466             4           1352        1352          5          2        2006            2007           2006  194000
# 2             5          900           3            1          288             3            864         900          3          1        1967            1967           1967  123000
# 3             5         1174           3            2          576             4            680         680          3          1        1900            2006           2000  135000
# 4             7         1958           4            3          936             4           1026        1026          4          2        2005            2005           2005  250000

x=datasets.drop('target',axis=1)
y=datasets['target']
test=test

print(x.head())
print(y.head())
print(test.head())

print('==================================결측치 탐지 시작(train)!!=======================================================')

for col1 in datasets.columns:
    n_nan1 = datasets[col1].isnull().sum()
    if n_nan1>0:
      msg1 = '{:^20}에서 결측치 개수: {}개'.format(col1,n_nan1)
      print(msg1)
    else:
        print('결측치가 없습니다.')

for col2 in datasets.columns:
    n_nan2 = datasets[col2].isnull().sum()
    if n_nan2>0:
        msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
        print(msg2)
    else:
      print('결측치가 없습니다.')
      
print('==================================결측치 탐지 완료(train)!!=======================================================')

print('==================================결측치 탐지 시작(test)!!========================================================')
for col1 in test.columns:
    n_nan1 = test[col1].isnull().sum()
    if n_nan1>0:
      msg1 = '{:^20}에서 결측치 개수: {}개'.format(col1,n_nan1)
      print(msg1)
    else:
        print('결측치가 없습니다.')

for col2 in test.columns:
    n_nan2 = test[col2].isnull().sum()
    if n_nan2>0:
        msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
        print(msg2)
    else:
      print('결측치가 없습니다.')
      
print('==================================결측치 탐지 완료(test)!!=======================================================')

x_train,x_test,y_train,y_test= train_test_split(x,y,shuffle=True, random_state=seed, train_size=0.7)

print(x_train)
print(y_train)

parameter = [
    {'xg__n_estimators':[100, 200, 300, 400, 500, 600, 700], 'xg__subsample':(0.5 , 1),
     'xg__learning_rate':[0.1, 0.3, 0.001, 0.01, 0.05], 'xg__min_child_weight':(0,3), 'xg__reg_lambda' : (0.001,10),
     'xg__max_depth':[4, 5, 6], 'xg__colsample_bytree':[0.3, 0.6, 0.9, 1], 'xg__eval_metric':['rmse']}
    ]

from sklearn.pipeline import Pipeline, make_pipeline

pipe = Pipeline([('mn', MinMaxScaler()), ('xg', XGBRegressor())])
model = GridSearchCV(pipe, parameter, cv=5, n_jobs=-1)

import time

start=time.time()

model.fit(x_train, y_train)

end=time.time()


y_pred = model.predict(x_test)

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
print(NMAE(y_test, y_pred))

nmae=NMAE(y_test, y_pred)
print("NMAE : ", np.round(nmae,6))

test_pred = model.predict(test)
submit_file['target'] = test_pred

print('==============================================')
print("걸린시간 :", end-start)
print('==============================================')

path_save_csv = '../Study/Hackarthon/dacon/housing/_save_csv/'

now1 = datetime.now()
now_date = now1.strftime("%m%d_%H%M")
submit_file.to_csv(path_save_csv + now_date + '_' + str(round(nmae, 4)) + '.csv')

with open(path_save_csv + now_date + '_' + str(round(nmae, 4)) + 'submit.txt', 'a') as file:
        file.write("\n=============================================")
        file.write('저장시간 : ' + now_date + '\n')
        file.write('colsample_bytree: ' + str('colsample_bytree') + '\n')
        file.write('learning_rate: ' + str('learning_rate') + '\n')
        file.write('max_depth: ' + str('max_depth') + '\n')
        file.write('min_child_weight: ' + str('min_child_weight') + '\n')
        file.write('n_estimators: ' + str('n_estimators') + '\n')
        file.write('reg_lambda: ' + str('reg_lambda') + '\n')
        file.write('subsample: ' + str('subsample') + '\n')

        file.write('걸린시간 : '+ str(round(end,4))+ '\n')
        file.write('NMAE : ' + str(round(nmae,6))+ '\n')


submit_file.to_csv(path+'sookook1200.csv', index=False)  

