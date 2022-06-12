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
# -sklearn: 0.24.2
# -tensorflow: 2.7.0

train = pd.read_csv('H:/Dacon/used/train.csv')
test = pd.read_csv('H:/Dacon/used/test.csv')

print(f'train data set은 {train.shape[1]} 개의 feature를 가진 {train.shape[0]} 개의 데이터 샘플로 이루어져 있습니다.')


# 데이터의 최상단 5 줄을 표시합니다.
print(train.head())

#    id                          title  odometer location    isimported          engine transmission    fuel  paint  year    target
# 0   0                   Toyota RAV 4     18277   Lagos   Foreign Used  4-cylinder(I4)    automatic  petrol    Red  2016  13665000
# 1   1            Toyota Land Cruiser        10    Lagos          New   4-cylinder(I4)    automatic  petrol  Black  2019  33015000
# 2   2  Land Rover Range Rover Evoque     83091    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol    Red  2012   9915000
# 3   3                   Lexus ES 350     91524    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol   Gray  2007   3815000
# 4   4                   Toyota Venza     94177    Lagos  Foreign Used  6-cylinder(V6)    automatic  petrol    Red  2010   7385000

print(train.shape, test.shape)

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

x = train.drop(['id','target'], axis=1)  
y = train['target']

test = test.drop(['id'], axis = 1)

x = x.to_numpy()
y = y.to_numpy()
test = test.to_numpy()

x = x.astype(float)
y = y.astype(float)
test = test.astype(float)

print(test.shape) #(436, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

input1=Input(shape=(10,))
dense1=Dense(32, activation='relu')(input1)
drop1=Dropout(0.25)(dense1)
dense2=Dense(64, activation='relu')(drop1)
dense3=Dense(64, activation='relu')(dense2)
dense4=Dense(64, activation='relu')(dense3)
dense5=Dense(128, activation='relu')(dense4)
dense6=Dense(128, activation='relu')(dense5)
dense7=Dense(64, activation='relu')(dense6)
drop8=Dropout(0.25)(dense7)
dense9=Dense(32, activation='relu')(drop8)
dense10=Dense(32, activation='relu')(dense9)
output1=Dense(1, activation='sigmoid')(dense10)


model=Model(inputs=input1, outputs=output1)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


############################################################################################################
import datetime
date=datetime.datetime.now()   
datetime = date.strftime("%m%d_%H%M")   #1206_0456
#print(datetime)

filepath='H:\_ModelCheckPoint'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'  # 2500-0.3724.hdf
model_path = "".join([filepath, 'k26_', datetime, '_', filename])
            #./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf
##############################################################################################################

es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_accuracy', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.3, callbacks=[es, mcp])

scaler = MinMaxScaler()
#scaler=StandardScaler()
# #scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test=scaler.transform(test)

def nmae(true, pred):
    
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    
    return score

y_predict = model.predict(x_test) # y예측
y_predict=y_predict.round(0).astype(float)
print(f'모델 NMAE: {nmae(y_test,y_predict)}')

y_predict[0:5]

submission = pd.read_csv('H:/Dacon/used/sample_submission.csv')
submission.head()
submission.to_csv('H:/Dacon/used/sookook15.csv', index=False)

