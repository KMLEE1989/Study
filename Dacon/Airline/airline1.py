from cgi import test
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn import datasets
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

seed = 66

print(f"-os:{platform.platform()}") #os:Windows-10-10.0.19043-SP0
print(f"-python:{sys.version}") #python:3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
print(f"-pandas:{pd.__version__}") #pandas:1.3.4
print(f"-numpy:{np.__version__}") #numpy:1.20.3
print(f"-sklearn: {sklearn.__version__}") #sklearn: 1.0.1

path = "../_data/dacon/airline_dataset/"

datasets = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path+'sample_Submission.csv')

print(datasets.info())
#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0   id                                 3000 non-null   int64
#  1   Gender                             3000 non-null   object
#  2   Customer Type                      3000 non-null   object
#  3   Age                                3000 non-null   int64
#  4   Type of Travel                     3000 non-null   object
#  5   Class                              3000 non-null   object
#  6   Flight Distance                    3000 non-null   int64
#  7   Seat comfort                       3000 non-null   int64
#  8   Departure/Arrival time convenient  3000 non-null   int64
#  9   Food and drink                     3000 non-null   int64
#  10  Gate location                      3000 non-null   int64
#  11  Inflight wifi service              3000 non-null   int64
#  12  Inflight entertainment             3000 non-null   int64
#  13  Online support                     3000 non-null   int64
#  14  Ease of Online booking             3000 non-null   int64
#  15  On-board service                   3000 non-null   int64
#  16  Leg room service                   3000 non-null   int64
#  17  Baggage handling                   3000 non-null   int64
#  18  Checkin service                    3000 non-null   int64
#  19  Cleanliness                        3000 non-null   int64
#  20  Online boarding                    3000 non-null   int64
#  21  Departure Delay in Minutes         3000 non-null   int64
#  22  Arrival Delay in Minutes           3000 non-null   float64
#  23  target                             3000 non-null   int64

print(test.info())

#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0   id                                 2000 non-null   int64
#  1   Gender                             2000 non-null   object
#  2   Customer Type                      2000 non-null   object
#  3   Age                                2000 non-null   int64
#  4   Type of Travel                     2000 non-null   object
#  5   Class                              2000 non-null   object
#  6   Flight Distance                    2000 non-null   int64
#  7   Seat comfort                       2000 non-null   int64
#  8   Departure/Arrival time convenient  2000 non-null   int64
#  9   Food and drink                     2000 non-null   int64
#  10  Gate location                      2000 non-null   int64
#  11  Inflight wifi service              2000 non-null   int64
#  12  Inflight entertainment             2000 non-null   int64
#  13  Online support                     2000 non-null   int64
#  14  Ease of Online booking             2000 non-null   int64
#  15  On-board service                   2000 non-null   int64
#  16  Leg room service                   2000 non-null   int64
#  17  Baggage handling                   2000 non-null   int64
#  18  Checkin service                    2000 non-null   int64
#  19  Cleanliness                        2000 non-null   int64
#  20  Online boarding                    2000 non-null   int64
#  21  Departure Delay in Minutes         2000 non-null   int64
#  22  Arrival Delay in Minutes           2000 non-null   float64

print(submit_file.info())

#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   id      2000 non-null   int64
#  1   target  2000 non-null   int64

datasets.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

print(datasets.shape)
print(test.shape)