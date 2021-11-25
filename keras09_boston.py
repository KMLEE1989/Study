#r2 0.8이상
#train 0.7
#80
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 

datasets = load_boston()
x=datasets.data
y=datasets.target

'''
print(x)
print(y)
print(x.shape)
print(y.shape)


print(datasets.feature_names)
print(datasets.DESCR)
'''

x_train, x_test, y_train, y_test= train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)


model = Sequential() 
model.add(Dense(5, input_dim=13))
model.add(Dense(11))
model.add(Dense(18))
model.add(Dense(32))
model.add(Dense(46))
model.add(Dense(55))
model.add(Dense(60))
model.add(Dense(52))
model.add(Dense(48))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(24))
model.add(Dense(18))
model.add(Dense(8))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=700, batch_size=13)


loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

#
#
#
