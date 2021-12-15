from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import numpy as np
from sklearn import datasets 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt 

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x=x.reshape(569,30,1)

#2. 모델구성 
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

# #scaler = MinMaxScaler()
# scaler=StandardScaler()
# #scaler=RobustScaler()
# #scaler=MaxAbsScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

model = Sequential() 
model.add(LSTM(64, input_shape=(30,1)))
model.add(Dense(64))
model.add(Dense(80, activation='relu'))
model.add(Dense(50))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

start=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])  
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)
y_predict= model.predict(x_test)
print(y_predict)


# LSTM
# loss : [0.16739928722381592, 0.9473684430122375]