#과제1. 중위값과 평균값의 의미 차이 
# 평균(mean)은 데이터를 모두 더한 후 데이터의 갯수로 나눈 값이다. 중앙값(median)은 전체 데이터 중 가운데에 있는 수이다. 데이터의 수가 짝수인 경우는 가장 가운데에 있는 두 수의 평균이 중앙값이다. 
# 이처럼 극단적인 값이 있는 경우 중앙값이 평균값보다 유용하다.
#통계에서 평균은 주어진 값 또는 수량 세트의 단순 평균으로 정의됩니다. 중간 값은 정렬 된 값 목록의 가운데 숫자라고한다.
#mean은 산술 평균이고, median은 위치 평균이며, 본질적으로 데이터 세트의 위치는 중앙값의 값을 결정한다.
#Mean은 데이터 세트의 중심을 설명하고 중간 값은 데이터 세트의 가장 중간 값을 강조 표시한다.
#평균은 정상적으로 분산 된 데이터에 적합합니다. 반면 데이터 분포가 왜곡 될 때 중앙값이 가장 좋습니다.
#주어진 데이터 세트의 평균 및 중간 값을 찾는다면.
#58, 26, 65, 34, 78, 44, 96  
# 평균=57.28 데이터의 합계를/n   
# 중앙값(올림차순으로 배열) 26,34,44,58,65,78,96  median=58 짝수인 경우 가운데 2개의 값을 2로 나눈다.

#과제2. 로그와 지수 
# 로그는 밑 a>0,a!=1  진수조건 x>0,y>0
# log a 1 =0 ,   log a a=1 
# logax +logay=loaxy
#logax-logay=loga(x/y)
#logax^n=nlogax

#로그와 지수변환
#logax=m  => x=a^m     loga1=0, logaa=1
#1=a^0 이므로 loga1=0 

#과제 3. 로그변환 했을때와 아닐때를 비교 하고 어떤 차이가 있는지 설명해 보거라  #데이터들이 쏠려있다 로그를 사용하면 정규분포화 할 수 있다 
# #로그변환전 
# loss :  23716.802734375                 
# r2스코어: 0.2496518256649335
# RMSE:  154.00260953094792

# 로그변환후
# loss :  1.4485578536987305
# r2스코어: 0.26077180726217164
# RMSE:  1.203560507111917  

import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score,mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
 

#1. 데이터

#.은 현재  ..이면 이전단계 
path = "./_data/bike/"

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
#y=np.log1p(y) 
#로그는 0의 값을 취할 수 없기에 fare에 작은 수 1을 전체적으로 더한 후 로그를 취해보자.

plt.plot(y)
plt.show()

'''
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

#2. 모델
model = Sequential() 
model.add(Dense(100, input_dim=8))
model.add(Dense(120))
model.add(Dense(130))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(130))
model.add(Dense(120))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4.평가, 예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_pred= model.predict(x_test)

r2=r2_score(y_test, y_pred)
print('r2스코어:', r2)

rmse=RMSE(y_test, y_pred)
print('RMSE: ', rmse)

loss :  23784.728515625
r2스코어: 0.24750282070952145
RMSE:  154.2229843904531

# #로그변환전                         #데이터들이 쏠려있다 로그를 사용하면 정규분포화 할 수 있다 
# loss :  23716.802734375
# r2스코어: 0.2496518256649335
# RMSE:  154.00260953094792

# 로그변환후
# loss :  1.4663631916046143
# r2스코어: 0.25168545240424756
# RMSE:  1.2109348047566921




############################################# 제출용 제작 #####################################################
results=model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

submit_file.to_csv(path + "자두꽃.csv", index=False)
'''


'''
# import matplotlib.pyplot as plt 

# #plt.scatter(x, y)
# plt.figure(figsize=(9,5))
# plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()
'''
