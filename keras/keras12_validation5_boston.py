from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
#import matplotlib.pyplot as plt 
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
                train_size=0.8, shuffle=True, random_state=66)


model = Sequential() 
model.add(Dense(20, input_dim=13))
model.add(Dense(17))
model.add(Dense(14))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=800, batch_size=13, validation_split=0.5)


loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

#r2 0.8이상
#train 0.7
#80

#loss : 17.10
#r2스코어: 0.79

#validation_split utilize : loss: 19.1073 val_loss: 35.6529 r2:0.7713969441408522
