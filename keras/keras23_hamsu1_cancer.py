from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target


import numpy as np
from sklearn import datasets 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt 

#1. 데이터


#print(datasets)
#print(datasets.DESCR)
#print(datasets.feature_names)


#print(x.shape,y.shape) #(569, 30) (569,)


print(y)
#print(y[:10]) 0과 1을 본 순간 2진분류 파악 -> 시그모이드 -> 바이너리 엔트로피  

print(np.unique(y))  #[0 1]

#2. 모델구성 

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

input1=Input(shape=(30,))
dense1=Dense(100)(input1) 
dense2=Dense(80)(dense1)
dense3=Dense(70, activation='relu')(dense2)
dense4=Dense(60)(dense3)
dense5=Dense(50)(dense4)
dense6=Dense(10)(dense4, activation='relu')(dense5)
output1=Dense(1, activation='sigmoid')(dense6)
model=Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

start=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])  #얼리스타핑을 콜백으로 땡기겠다 리스트
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)
y_predict= model.predict(x_test)
print(y_predict)

#result
#loss: 0.1946 - accuracy: 0.3070