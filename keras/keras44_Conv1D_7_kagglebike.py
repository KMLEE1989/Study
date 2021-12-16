
import numpy as np
from numpy.core.records import array
import pandas as pd 
from sklearn.metrics import r2_score,mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Flatten, Conv1D
import time

#1. 데이터

#.은 현재  ..이면 이전단계 
path = "../_data/kaggle/bike/"

train = pd.read_csv(path+'train.csv')
#print(train)  #(10886, 12)
print(train.shape)


test_file = pd.read_csv(path+'test.csv')
#print(test_file) #(6493,9)
print(test_file.shape)


submit_file = pd.read_csv(path+'sampleSubmission.csv')
print(submit_file) #(6493,2)
print(submit_file.columns)   #['datetime', 'count']


print(type(train))     #class 'pandas.core.frame.DataFrame'
print(train.info())     #중의값과 평균값의 비교 분석 50%는 중의값 mean은 평균값   *log 변환할때 0이 나오면 안대  그래서 항상 1을 더해준다(log1p)  로그와 지수 공부  
print(train.describe())

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

# plt.plot(y)
# plt.show()


x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
x_train=x_train.reshape(x_train.shape[0], 8,1)
x_test=x_test.reshape(x_test.shape[0],8,1)


#2. 모델
model = Sequential() 
model.add(Conv1D(100, 2, input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. compile
model.compile(loss='mse', optimizer='adam')

start=time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2)
end=time.time() - start

print("걸린시간:  ", round(end,3))

#4.평가, 예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_pred= model.predict(x_test)
r2=r2_score(y_test, y_pred)
print('r2스코어:', r2)
rmse=RMSE(y_test, y_pred)
print('RMSE: ', rmse)

# loss :  23784.728515625
# r2스코어: 0.24750282070952145
# RMSE:  154.2229843904531

#Conv1D
# 걸린시간:   26.134
# 69/69 [==============================] - 0s 606us/step - loss: 24840.5176
# loss :  24840.517578125
# r2스코어: 0.21410014716790893
# RMSE:  157.60872963185452

# LSTM
# loss :  21202.609375
# r2스코어: 0.3291954230054981
# RMSE:  145.61116393571226