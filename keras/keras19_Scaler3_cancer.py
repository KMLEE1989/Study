###########################################################
#각각의 Scaler의 특성과 정의 정리해놓을것! 
###########################################################

#######

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

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
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)

model = Sequential() 
model.add(Dense(100, activation='linear', input_dim=30))
model.add(Dense(80))
model.add(Dense(70, activation='relu'))
model.add(Dense(60))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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

#결과 Activation Relu(미적용)
# loss: 0.1483 - accuracy: 0.9649

# MinMax
# loss: 0.3298 - accuracy: 0.8860

# Standard
# loss: 0.1166 - accuracy: 0.9649

# Robust
# loss: 0.1117 - accuracy: 0.9737

#MaxAbs
# loss: 0.1086 - accuracy: 0.9737

###############################################

#Activation Relu(적용)

#결과
# loss: 0.2733 - accuracy: 0.8860

# MinMax
# loss: 0.2015 - accuracy: 0.9649

# Standard
# loss: 0.1105 - accuracy: 0.9737

#Robust
# loss: 0.1914 - accuracy: 0.9474


#MaxAbs
# loss: 0.1723 - accuracy: 0.9561

