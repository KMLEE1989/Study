from tensorflow.keras.models import Sequential,load_model 
from tensorflow.keras.layers import Dense
import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time

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
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

model=load_model('./_save/keras25_3_save_model.h5')
# model = Sequential() 
# model.add(Dense(40, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))
# model.summary()
# model.save("./_save/keras25_1_save_model.h5")

#model=load_model('./_save/keras25_1_save_model.h5')
#model=load_model('./_save/keras25_3_save_model.h5')

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.5)  #로스와 발로스 반환

#model=load_model('./_save/keras25_1_save_model.h5')
#model=load_model("./_save/keras25_3_save_model.h5")


#4.
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

#model=load_model('./_save/keras25_3_save_model.h5')
#model=load_model('./_save/keras25_3_save_model.h5')

#save를 핏 다음에 하면 웨이트까지 저장되고
#모델 다음에는 모델만 저장된다 