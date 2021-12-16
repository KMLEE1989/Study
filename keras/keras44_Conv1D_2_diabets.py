from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import time

from tensorflow.python.keras.backend import flatten

datasets = load_diabetes()
x=datasets.data
y=datasets.target

#print(x.shape) (442, 10)
#print(y.shape)  (442,)

x=x.reshape(442,10,1)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, shuffle=True, random_state=49)

model = Sequential() 
model.add(Conv1D(50, 2, input_shape=(10,1)))
model.add(Flatten())
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

start=time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.5)  #로스와 발로스 반환
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)


# LSTM
# loss : 4097.7119140625
# r2스코어: 0.26810775331507164

# Conv1D
# 걸린시간:   4.444
# 5/5 [==============================] - 0s 748us/step - loss: 3124.6042
# loss : 3124.604248046875
# r2스코어: 0.44191447641279136