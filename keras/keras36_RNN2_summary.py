import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split


#1. 데이터
x=np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6]])

y=np.array([4,5,6,7])

print(x.shape, y.shape)  #(4, 3) (4,)

# input_shape= (batch_size, timesteps, feature)  2차원에서 3차원으로 바꿀때 내용물과 순서가 바뀌면 안대
# input_shape=(행, 열, 몇개씩 자르는지!!!)

x=x.reshape(4,3,1)  #3,1  인풋 shape   (N,3,1)


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(5, activation='linear', input_shape=(3,1)))  
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# #3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam')
# model.fit(x,y, epochs=100)

# #4. 평가 예측
# model.evaluate(x,y)
# # y=np.array([5,6,7]).reshape(1,3,1)
# result = model.predict([[[5],[6],[7]]])
# print(result)

