import os
from argon2 import Parameters
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import platform
import sys
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,  GridSearchCV

file_path = 'H:/Study/Hackarthon/dacon/shopping/dataset/dataset'

train = pd.read_csv('H:/Study/Hackarthon/dacon/shopping/dataset/dataset/train.csv')
test = pd.read_csv('H:/Study/Hackarthon/dacon/shopping/dataset/dataset/test.csv')
sample_submission = pd.read_csv('H:/Study/Hackarthon/dacon/shopping/dataset/dataset/sample_submission.csv')

print(f"-os:{platform.platform()}") 
print(f"-python:{sys.version}") 
print(f"-pandas:{pd.__version__}") 
print(f"-numpy:{np.__version__}") 
print(f"-sklearn: {sklearn.__version__}") 
print(f'-tensorflow: {tf.__version__}')

# print(train.shape) (6255, 13)
# print(test.shape) (180, 12)
# print(sample_submission.shape) (180, 2)

# print(train.info())

# #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   id            6255 non-null   int64
#  1   Store         6255 non-null   int64
#  2   Date          6255 non-null   object
#  3   Temperature   6255 non-null   float64
#  4   Fuel_Price    6255 non-null   float64
#  5   Promotion1    2102 non-null   float64
#  6   Promotion2    1592 non-null   float64
#  7   Promotion3    1885 non-null   float64
#  8   Promotion4    1819 non-null   float64
#  9   Promotion5    2115 non-null   float64
#  10  Unemployment  6255 non-null   float64
#  11  IsHoliday     6255 non-null   bool
#  12  Weekly_Sales  6255 non-null   float64
# dtypes: bool(1), float64(9), int64(2), object(1)
# memory usage: 592.6+ KB
# None

train = train.fillna(0)

# print(train.shape) (6255, 13)

def date_encoder(date):
    day, month, year = map(int, date.split('/'))
    return day, month, year

train['Day'] = train['Date'].apply(lambda x: date_encoder(x)[0])
train['Month'] = train['Date'].apply(lambda x: date_encoder(x)[1])
train['Year'] = train['Date'].apply(lambda x: date_encoder(x)[2])

train = train.drop(columns=['Day', 'Date'])

scaler = StandardScaler()

scaler.fit(train[['Promotion1', 'Promotion2','Promotion3', 'Promotion4', 'Promotion5']])

scaled = scaler.transform(train[['Promotion1','Promotion2','Promotion3','Promotion4','Promotion5']])

train[['Scaled_Promotion1','Scaled_Promotion2',
        'Scaled_Promotion3','Scaled_promotion4',
        'Scaled_Promotion5']] = scaled

train = train.drop(columns=['Promotion1','Promotion2','Promotion3','Promotion4','Promotion5'])

test = test.fillna(0)

test['Month']=test['Date'].apply(lambda x: date_encoder(x)[1])
test['Year'] = test['Date'].apply(lambda x:date_encoder(x)[2])
test = test.drop(columns=['Date'])

scaled = scaler.transform(test[['Promotion1','Promotion2','Promotion3','Promotion4','Promotion5']])

test[['Scaled_Promotion1','Scaled_Promotion2','Scaled_Promotion3','Scaled_promotion4','Scaled_Promotion5']] = scaled

test = test.drop(columns=['Promotion1','Promotion2','Promotion3','Promotion4','Promotion5'])


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# model = make_pipeline(MinMaxScaler(), XGBRegressor())

# model = RandomForestRegressor()


train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


x_train = train.drop(columns=['Weekly_Sales'])
y_train = train['Weekly_Sales']

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

Parameter = [
    {'xg__n_estimators':[100, 200, 300, 400, 500, 600], 'xg__subsample':(0.5 , 1),
     'xg__learning_rate':[0.1, 0.3, 0.001, 0.01, 0.05], 'xg__min_child_weight':(0,3), 'xg__reg_lambda' : (0.001,10),
     'xg__max_depth':[4, 5, 6, 7], 'xg__colsample_bytree':[0.3, 0.6, 0.9, 1], 'xg__eval_metric':['rmse']}
    ]

model = GridSearchCV(XGBRegressor(), Parameter, cv = kfold, verbose=1, 
                     refit=True, n_jobs=-1)  # cv = cross validation은 kfold로 / # 해당 모델에 맞는 parameter로 써주기!
                                           
 

model.fit(x_train, y_train)

prediction = model.predict(test)
print('---------------------------예측된 데이터의 상위 10개의 값 확인 --------------------------------- \n')
print(prediction[:10])

sample_submission['Weekly_Sales'] = prediction

# sample_submission.head()

sample_submission.to_csv('H:/Study/Hackarthon/dacon/shopping/dataset/dataset/sookook8.csv', index = False)



