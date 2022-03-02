from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split


a = load_boston()
b = a.data
c = a.target

size = 7

def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): # 1~10 - 5 +1 //size가 나오는 가장 마지막 것을 생각해서 +1을 해주어야 함
        subset = data[i : (i+size)]      # [1 : 6] = 1,2,3,4,5 
        aaa.append(subset)                  # 1,2,3,4,5에 []를 붙인다.
    return np.array(aaa)

x = split_x(b,size)
y = split_x(c,size)
print(x.shape)   # (502, 5, 13)
print(y.shape)   # (502, 5)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42
)
print(x_train.shape) # (351, 5, 13)
print(y_test.shape) # (151, 5, 13)



#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (5,13))) # input_shape 할 때 행은 넣지 않는다. 
model.add(Dense(128, activation='relu'))       # 모든 activation은 다음으로 보내는 값을 한정 시키기 때문에 윗 줄에서도 가능하다.
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(6))

#  = split_x[:,:-1]
# y = split_x[:,-1]
# print(x.shape,y.shape)  # (506, 13) (506,)
#3. 컴파일, 훈련 
model.compile(loss = 'mae', optimizer = 'adam')   # optimizer는 loss값을 최적화 한다.
model.fit(x, y, epochs = 10)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)