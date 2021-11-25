from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data 
y = datasets.target 

'''
print(x)
print(y)

print(x.shape, y.shape) #(442,10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

model = Sequential() 
model.add(Dense(1, input_dim=10))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=10,verbose=0)

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

#result
#loss : 3003.17138671875
#r2스코어: 0.5179786787113398

#0.62  이상
#R2 
#70