from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

###과제###
#train과 test비율을 8:2으로 분리하시오.

x_train = x[:8]
x_test= x[-2:]
y_train= y[:8]
y_test= y[-2:]


#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict([11])
print('11의 예측값: ', result)

#result 
# loss: 7.54681832404458e-06
#11의 예측값:  [[11.003186]]