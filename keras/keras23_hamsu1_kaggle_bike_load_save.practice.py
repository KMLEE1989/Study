#log 사용 후의 결과 값과  로그 사용후 relu사용 값을 비교 해 봤을때 커다란 차이는 없었다. 게다가 모든 데이터 스케일링이 성공 적이지 못했다. 파라미터 튜닝의 잘못됨이 크므로
#재조정이 필요하다. 

import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score,mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model 
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

path = "../_data/kaggle/bike/"

train = pd.read_csv(path+'train.csv')
#print(train)  #(10886, 12)
print(train.shape)
#                   datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count
# 0      2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000       3          13     16
# 1      2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000       8          32     40
# 2      2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000       5          27     32
# 3      2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000       3          10     13
# 4      2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000       0           1      1
# ...                    ...     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...    ...
# 10881  2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329    336
# 10882  2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231    241
# 10883  2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164    168
# 10884  2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117    129
# 10885  2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84     88

#[10886 rows x 12 columns]

test_file = pd.read_csv(path+'test.csv')
#print(test_file) #(6493,9)
print(test_file.shape)
#                  datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed
# 0     2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027
# 1     2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000
# 2     2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000
# 3     2011-01-20 03:00:00       1        0           1        1  10.66  12.880        56    11.0014
# 4     2011-01-20 04:00:00       1        0           1        1  10.66  12.880        56    11.0014
# ...                   ...     ...      ...         ...      ...    ...     ...       ...        ...
# 6488  2012-12-31 19:00:00       1        0           1        2  10.66  12.880        60    11.0014
# 6489  2012-12-31 20:00:00       1        0           1        2  10.66  12.880        60    11.0014
# 6490  2012-12-31 21:00:00       1        0           1        1  10.66  12.880        60    11.0014
# 6491  2012-12-31 22:00:00       1        0           1        1  10.66  13.635        56     8.9981
# 6492  2012-12-31 23:00:00       1        0           1        1  10.66  13.635        65     8.9981

# [6493 rows x 9 columns]

submit_file = pd.read_csv(path+'sampleSubmission.csv')
print(submit_file) #(6493,2)
print(submit_file.columns)   #['datetime', 'count']

print(type(train))     #class 'pandas.core.frame.DataFrame'
print(train.info())     #중의값과 평균값의 비교 분석 50%는 중의값 mean은 평균값   *log 변환할때 0이 나오면 안대  그래서 항상 1을 더해준다(log1p)  로그와 지수 공부  
print(train.describe())
#             season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000

print(train.columns)
#Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
#        dtype='object')
print(train.head(3))
print(train.tail())

x = train.drop(['datetime', 'casual' , 'registered', 'count'], axis=1)  #컬럼 삭제할때는 드랍에 액시스 1 준다   
test_file = test_file.drop(['datetime'], axis=1)

print(x.columns)
print(x.shape)   #(10886, 8)
y=train['count']   #count는 회귀 모델 
print(y)
print(y.shape)  #(10886,)


#로그변환
y=np.log1p(y) 
#로그는 0의 값을 취할 수 없기에 fare에 작은 수 1을 전체적으로 더한 후 로그를 취해보자.

#plt.plot(y)
#plt.show()


x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)
#test_file=scaler.transform(test_file)

#2. 모델
# input1=Input(shape=(8,))
# dense1=Dense(100)(input1)
# dense2=Dense(100)(dense1)
# dense3=Dense(110)(dense2)
# dense4=Dense(110, activation='relu')(dense3)
# dense5=Dense(120)(dense4)
# dense6=Dense(120)(dense5)
# dense7=Dense(100,activation='relu')(dense6)
# dense8=Dense(110)(dense7)
# dense9=Dense(100)(dense8)
# dense10=Dense(100)(dense9)
# output1=Dense(1, activation='softmax')(dense10)
# model=Model(inputs=input1, outputs=output1)

#model.save("./_save/keras23_hamsu1_kaggle_bike_load_save.practice.h5")
#model=load_model("./_save/keras23_hamsu1_kaggle_bike_load_save.practice.h5")

#3. compile
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#model.save("./_save/keras23_hamsu1_kaggle_bike_load_save.practice.h5")
model=load_model("./_save/keras23_hamsu1_kaggle_bike_load_save.practice.h5")


#4.평가, 예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_pred= model.predict(x_test)

r2=r2_score(y_test, y_pred)
print('r2스코어:', r2)

rmse=RMSE(y_test, y_pred)
print('RMSE: ', rmse)

results=model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

submit_file.to_csv(path + "final.csv", index=False)

#Result
#loss :  14.812507629394531
#r2스코어: -6.559119706695873
#RMSE:  3.848701944240732



