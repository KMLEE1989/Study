from tensorflow.keras.models import Sequential,load_model 
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.python.keras.backend import sigmoid
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets=load_boston()
x=datasets.data
y=datasets.target

#print(np.min(x), np.max(x))  # 0.0 711.0
# x = x/711.               #부동소수점으로 나눈다 
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#scaler = MinMaxScaler()
# scaler=StandardScaler()
#scaler=RobustScaler()
scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# input1 = Input(shape=(13,))
# dense1=Dense(50)(input1)
# dense2=Dense(40)(dense1)
# dense3=Dense(30)(dense2)
# dense4=Dense(20, activation='relu')(dense3)
# dense5=Dense(10)(dense4)
# dense6=Dense(8, activation='relu')(dense5)
# dense7=Dense(5)(dense6)
# dense8=Dense(4)(dense7)
# dense9=Dense(3)(dense8)
# output1=Dense(1)(dense9)
# model=Model(inputs=input1, outputs=output1)

# model=load_model("./_save/keras23_hamsu1_boston load_save practice.h5")


#3.
# model.compile(loss='mse', optimizer='adam')
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=13, validation_split=0.5)  #로스와 발로스 반환

#model.save("./_save/keras23_hamsu1_boston load_save practice.h5")

model=load_model("./_save/keras23_hamsu1_boston load_save practice.h5")

#4.
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)




