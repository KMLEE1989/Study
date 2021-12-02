from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets=load_diabetes()
x=datasets.data
y=datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


input1 = Input(shape=(10,)) 
dense1=Dense(50)(input1)
dense2=Dense(70)(dense1)
dense3=Dense(100)(dense2)
dense4=Dense(120, activation='relu')(dense3)
dense5=Dense(180)(dense4)
dense6=Dense(200, activation='relu')(dense5)
dense7=Dense(170)(dense6)
dense8=Dense(120)(dense7)
dense9=Dense(80)(dense8)
dense10=Dense(40)(dense9)
output1=Dense(1)(dense10)
model=Model(inputs=input1, outputs=output1)


model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.5, callbacks=[es])  #로스와 발로스 반환

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)


#결과
#loss : 3345.7314453125
#r2스코어: 0.4629963705269956