from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
import time

datasets = load_boston()
x=datasets.data
y=datasets.target

#print(x.shape) (506, 13)
#print(y.shape) (506,)

x=x.reshape(506,13,1)


x_train, x_test, y_train, y_test= train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)


model = Sequential() 
model.add(Conv1D(20,2, input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(17))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

start=time.time()
model.fit(x_train, y_train, epochs=100, batch_size=13)
end=time.time()-start

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)
print("걸린시간:  ", round(end,3))

# start=time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.5)  #로스와 발로스 반환
# end=time.time() - start

# print("걸린시간:  ", round(end,3))

# Conv1D
# loss : 22.46760368347168
# r2스코어: 0.7280514167913468
# 걸린시간:   2.93

# LSTM
# loss : 17.895166397094727
# r2스코어: 0.7833963102866099