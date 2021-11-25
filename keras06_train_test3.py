from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np 

#1. 데이터
x=np.array(range(100))  #0~99 (연산의 가지 파라미터)
y=np.array(range(1,101)) # 1~100    (bias= 1 why)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size=0.7, shuffle=True, random_state=66)


print(x_test) #[ 8 93  4  5 52 41  0 73 88 68]
print(y_test)

#100을 예측하시오 
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(280))
model.add(Dense(320))
model.add(Dense(280))
model.add(Dense(240))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(130))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1) #fit에는 트레인

loss = model.evaluate(x_test, y_test)  #evaluate 테스트 
print('loss:', loss)
result = model.predict([101])
print('101의 예측값: ', result)

#result
#loss: 0.00013206986477598548
#101의 예측값:  [[101.98323]]


