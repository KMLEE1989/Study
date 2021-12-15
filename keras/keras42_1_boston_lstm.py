from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 

datasets = load_boston()
x=datasets.data
y=datasets.target

#print(x.shape) (506, 13)
#print(y.shape) (506,)

x=x.reshape(506,13,1)


x_train, x_test, y_train, y_test= train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)


model = Sequential() 
model.add(LSTM(20, input_shape=(13,1)))
model.add(Dense(17))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=13)


loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

# 미적용
# loss : 46.80183410644531
# r2스코어: 0.4335091058466005

# LSTM
# loss : 17.895166397094727
# r2스코어: 0.7833963102866099
