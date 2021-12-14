import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

#1. 데이터

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict=np.array([50,60,70])

print(x.shape) #(13, 3)
print(y.shape) #(13,)


x=x.reshape(13,3,1)  #3,1  인풋 shape   (N,3,1)


#2. 모델구성
model = Sequential()
model.add(LSTM(32, activation='linear', input_shape=(3,1)))
#model.add(SimpleRNN(units=10, activation='linear', input_shape=(3,1)))  
# model.add(SimpleRNN(10, input_length=3, input_dim=1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#model.summary()

# #3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=1000)

# #4. 평가 예측
model.evaluate(x,y)
# # y=np.array([5,6,7]).reshape(1,3,1)
result = model.predict([[[50],[60],[70]]])
print(result)

# 80 넘기기

# [[80.16723]]