import numpy as np
import matplotlib as plt

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,6])

#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow. keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.000001


# optimizer = Adam(learning_rate=learning_rate)
# loss :  3.3595 lr :  1e-06 결과물 :  [[10.930994]]

# optimizer = Adadelta(learning_rate=learning_rate)
# loss :  3.3465 lr :  0.001 결과물 :  [[11.092109]]

# optimizer = Adagrad(learning_rate=learning_rate)
# loss :  3.3535 lr :  2e-05 결과물 :  [[10.989416]]

# optimizer = Adamax(learning_rate=learning_rate)
# loss :  3.2637 lr :  2e-05 결과물 :  [[11.137824]]

# optimizer = RMSprop(learning_rate=learning_rate)
# loss :  3.0732 lr :  0.00015 결과물 :  [[11.260343]]

# optimizer = SGD(learning_rate=learning_rate)
# loss :  3.2566 lr :  2e-05 결과물 :  [[10.919136]]

optimizer = Nadam(learning_rate=learning_rate)
# loss :  3.3603 lr :  1e-06 결과물 :  [[10.914702]]

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer)



model.fit(x,y,epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y, batch_size=1)

y_predict = model.predict([11])

print('loss : ', round(loss,4), 'lr : ', learning_rate, '결과물 : ', y_predict)


