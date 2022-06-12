import os
import sys
import platform
import random
import math
from typing import List, Dict,Tuple 
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost.sklearn import XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  
from tensorflow.keras. optimizers import SGD
from keras import optimizers
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.layers.core import Activation
from sklearn import metrics
from tensorflow.keras.models import Model,load_model
import numpy as np
from sklearn.metrics import mean_squared_error

def nmae(true, pred):
    
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    
    return score

print(f"-os:{platform.platform()}") 
print(f"-python:{sys.version}") 
print(f"-pandas:{pd.__version__}") 
print(f"-numpy:{np.__version__}") 
print(f"-sklearn: {sklearn.__version__}") 
print(f'-tensorflow: {tf.__version__}')

# -os:Windows-10-10.0.19043-SP0
# -python:3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
# -pandas:1.3.4
# -numpy:1.20.3
# -sklearn: 1.1.1
# -tensorflow: 2.7.0

train = pd.read_csv('H:/Dacon/used/train.csv')
test = pd.read_csv('H:/Dacon/used/test.csv')
submit_file = pd.read_csv('H:/Dacon/used/sample_submission.csv')

print(train.shape) #(1015, 11)
print(test.shape) #(436, 10)
print(submit_file.shape) #(436, 2)

print(train.describe)

#         id                          title  odometer location    isimported          engine transmission   
#  fuel   paint  year    target
# 0        0                   Toyota RAV 4     18277   Lagos   Foreign Used  4-cylinder(I4)    automatic  petrol     Red  2016  13665000     
# 1        1            Toyota Land Cruiser        10    Lagos          New   4-cylinder(I4)    automatic  petrol   Black  2019  33015000     
# 2        2  Land Rover Range Rover Evoque     83091    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol     Red  2012   9915000     
# 3        3                   Lexus ES 350     91524    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol    Gray  2007   3815000     
# 4        4                   Toyota Venza     94177    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol     Red  2010   7385000     
# ...    ...                            ...       ...      ...           ...             ...          ...     ...     ...   ...       ...     
# 1010  1010                 Toyota Corolla     46768    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol   Black  2014   5415000     
# 1011  1011                   Toyota Camry     31600    Abuja  Foreign Used  4-cylinder(I4)    automatic  petrol  Silver  2011   3615000     
# 1012  1012                   Toyota Camry     96802    Abuja  Foreign Used  4-cylinder(I4)    automatic  petrol   Black  2011   3415000     
# 1013  1013                   Lexus GX 460    146275    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol    Gold  2013  14315000     
# 1014  1014                         DAF CF         0    Lagos  Locally used  6-cylinder(V6)       manual  diesel   white  1998  10015000     

# [1015 rows x 11 columns]>

print(test.describe)

# id                  title  odometer location    isimported          engine transmission    fuel     
#   paint  year
# 0      0    Mercedes-Benz C 300      1234    Abuja          New   4-cylinder(I4)    automatic  petrol       White  2017
# 1      1           Honda Accord     29938    Abuja  Foreign Used  4-cylinder(I4)    automatic  petrol       White  2013
# 2      2    Mercedes-Benz S 550     87501    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol       Black  2012
# 3      3          Toyota Sienna    180894    Lagos  Locally used  6-cylinder(V6)    automatic  petrol   Dark Grey  2001
# 4      4           Toyota Hiace    104814    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol       White  2000
# ..   ...                    ...       ...      ...           ...             ...          ...     ...         ...   ...
# 431  431  Mercedes-Benz GLK 350     78175    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol   Dark Blue  2014
# 432  432        Honda Crosstour    129223    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol         Red  2011
# 433  433   Mercedes-Benz ML 350    100943    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol       Black  2013
# 434  434           Lexus GX 470     81463    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol  Mint green  2003
# 435  435          Toyota Sienna       646    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol      Silver  2006

# [436 rows x 10 columns]>

print(submit_file.describe)

# <bound method NDFrame.describe of       id  target
# 0      0       0
# 1      1       0
# 2      2       0
# 3      3       0
# 4      4       0
# ..   ...     ...
# 431  431       0
# 432  432       0
# 433  433       0
# 434  434       0
# 435  435       0

# [436 rows x 2 columns]>

print(train.columns)
# Index(['id', 'title', 'odometer', 'location', 'isimported', 'engine',
#        'transmission', 'fuel', 'paint', 'year', 'target'],
#       dtype='object')

print(test.columns)
# Index(['id', 'title', 'odometer', 'location', 'isimported', 'engine',
#        'transmission', 'fuel', 'paint', 'year'],
#       dtype='object')

print(submit_file.columns)
# Index(['id', 'target'], dtype='object')

# 각 파일의 feature 를 파악해 봅니다. 
####################################################################################################################

train['brand'] = train['title'].apply(lambda x : x.split(" ")[0])
test['brand'] = test['title'].apply(lambda x: x.split(" ")[0])

print(train.brand.head())
print(test.brand.head())

print(set(train.brand) | set(test.brand))

# 이상값 수정
train['brand'] = train['brand'].replace({'Mercedes-Benz/52' : 'Mercedes-Benz'})

brand_label = {_brand : label for label, _brand in enumerate(set(set(train.brand)|set(test.brand)))}
print(brand_label)

train['brand'] = train['brand'].replace(brand_label)
test['brand'] = test['brand'].replace(brand_label)

# train, test 데이터에 있는 모든 car title 합치기
car_title = pd.concat([train.title, test.title], ignore_index = True)
car_title = set(car_title)
print(car_title)

car_title_label = {car_title : label for label, car_title in enumerate(car_title)}
print(car_title_label)

# 생성된 dictionary를 이용하여 car title labeling
train['title'] = train['title'].replace(car_title_label)
test['title'] = test['title'].replace(car_title_label)

print(train.head(1))
print(test.head(1))

print(train.location.unique())
print(test.location.unique())

train['location'] = train['location'].replace({
    'Abia State' : 'Abia',
    'Abuja ' : 'Abuja',
    'Lagos ' : 'Lagos',
    'Lagos State' : 'Lagos',
    'Ogun State' : 'Ogun'
    })

test['location'] = test['location'].replace({
    'Abuja ' : 'Abuja',
    'Lagos ' : 'Lagos',
    'Lagos State' : 'Lagos',
    'Ogun State' : 'Ogun',
    'Arepo ogun state ' : 'Ogun'
    # Arepo is a populated place located in Ogun State, Nigeria. 출처. 위키백과
})

# 합집합( '|' ) - 교집합( '&' )
set(set(train.location.unique()) | set(test.location.unique())) - set(set(train.location.unique()) & set(test.location.unique()))

train['location'] = train['location'].replace({
    'Accra' : 'other',
    'Adamawa ' : 'other',
    'FCT' : 'other',
    'Mushin' : 'other'
})

print(train.location.unique())

test['location'] = test['location'].replace({
    'Accra' : 'other',
    'Adamawa ' : 'other',
    'FCT' : 'other',
    'Mushin' : 'other'
})

print(test.location.unique())

location_label = {location : label for label, location in enumerate(train.location.unique())}
print(location_label)

train['location'] = train['location'].replace(location_label)
test['location'] = test['location'].replace(location_label)

import re 

def clean_text(texts): 
    corpus = [] 
    for i in range(0, len(texts)): 
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>\<]', '',texts[i]) #@%*=()/+ 와 같은 문장부호 제거
        review = re.sub(r'\d+','',review)#숫자 제거
        review = review.lower() #소문자 변환
        review = re.sub(r'\s+', ' ', review) #extra space 제거
        review = re.sub(r'<[^>]+>','',review) #Html tags 제거
        review = re.sub(r'\s+', ' ', review) #spaces 제거
        review = re.sub(r"^\s+", '', review) #space from start 제거
        review = re.sub(r'\s+$', '', review) #space from the end 제거
        review = re.sub(r'_', ' ', review) #space from the end 제거
        #review = re.sub(r'l', '', review)
        corpus.append(review) 
        
    return corpus

train_paint = clean_text(train['paint']) #메소드 적용
train['paint'] = train_paint
print('train data에서 paint의 unique 카테고리 개수 : ', len(train['paint'].unique()))

train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)

print(train.paint.unique())

test_paint = clean_text(test['paint'])
test['paint'] = test_paint
print('test data에서 paint의 unique 카테고리 개수 : ', len(test['paint'].unique()))

test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'black' if x.find('blac') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'gold' if x.find('golf') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)

print(test.paint.unique())

paint_label = {_paint : label for label, _paint in enumerate(set(pd.concat([train.paint, test.paint])))}

print(paint_label)

train['paint'] = train['paint'].replace(paint_label)
test['paint'] = test['paint'].replace(paint_label)

set(set(train.engine.unique()) | set(test.engine.unique())) - set(set(train.engine.unique()) & set(test.engine.unique()))

print(train[train.engine == '4-cylinder(H4)'])

train = train.replace({'4-cylinder(H4)' : '4-cylinder(I4)'})

train.iloc[[327, 830]]

test[test.engine == '12-cylinder(V12)']

test = test.replace({'12-cylinder(V12)' : '8-cylinder(V8)'})

test.iloc[[142]]

engine_label = {_engine : label for label, _engine in enumerate(set(pd.concat([train.engine, test.engine])))}
print(engine_label)

train = train.replace(engine_label)
test = test.replace(engine_label)

train[(train.year == 1217) | (train.year == 1218)]

test[(test.year == 1324) | (test.year == 1726) | (test.year == 2626)]

train['year'] = train['year'].replace([1218, 1217], [2010, 2010])
test['year'] = test['year'].replace([1324, 1726, 2626], [2010, 2010, 2020])

train.iloc[[415, 827]]

test.iloc[[304, 406, 411]]

isimported_label = {'Foreign Used': 0, 'Locally used' : 1, 'New ' : 2}
transmission_label = {'automatic' : 0, 'manual' : 1}
fuel_label = {'petrol' : 0, 'diesel' : 1}

train['isimported'] = train['isimported'].replace(isimported_label)
test['isimported'] = test['isimported'].replace(isimported_label)

train['transmission'] = train['transmission'].replace(transmission_label)
test['transmission'] = test['transmission'].replace(transmission_label)

train['fuel'] = train['fuel'].replace(fuel_label)
test['fuel'] = test['fuel'].replace(fuel_label)

print(train.head())

print(test.head())

from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

#model = RandomForestRegressor() 
# model = CatBoostRegressor()

model = make_pipeline(MinMaxScaler(), CatBoostRegressor())

x_train = train.drop(['id', 'target'], axis = 1)
y_train = train.target

x_test = test.drop(['id'], axis = 1)

model.fit(x_train, y_train)

pred = model.predict(x_test)

submission = pd.read_csv('H:/Dacon/used/sample_submission.csv')
submission.head()

submission['target'] = pred
print(pred[0:5])

submission.to_csv('H:/Dacon/used/sookook555.csv', index=False)


