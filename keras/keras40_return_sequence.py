import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import time



#실습!!!
#GRU로 구현
#성능이 유사할 경우 - fit에 time 걸어서 속도 확인

#1. 데이터

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict=np.array([50,60,70])

#2. 모델구성
model=Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(3,1)))  # (N,3,1 -> (N,3,10)
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(Dense(1))

model.summary()



# print(x.shape) #(13, 3)
# print(y.shape) #(13,)


x=x.reshape(13,3,1)  #3,1  인풋 shape   (N,3,1)


# #2. 모델구성
# model = Sequential()
# model.add(GRU(32, activation='linear', input_shape=(3,1)))
# #model.add(SimpleRNN(units=10, activation='linear', input_shape=(3,1)))  
# # model.add(SimpleRNN(10, input_length=3, input_dim=1))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

# #model.summary()

# #3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

start= time.time()

model.fit(x,y, epochs=1000)

end=time.time()-start


# #4. 평가 예측
model.evaluate(x,y)
y=np.array([5,6,7]).reshape(1,3,1)
result = model.predict([[[50],[60],[70]]])
print(result)
print("걸린시간 :" , round(end, 2), '초 ')

# # LSTM
# # [[81.77936]]
# # 걸린시간 : 3.29 초

# #GRU
# # [[81.77936]]
# # 걸린시간 : 3.29 초

# # print("걸린시간 :" , round(end, 3), '초')


# [[13.823069]]
# 걸린시간 : 5.37 초
