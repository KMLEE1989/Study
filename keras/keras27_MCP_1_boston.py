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
import time

datasets=load_boston()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(13,))
dense1=Dense(50)(input1)
dense2=Dense(40)(dense1)
dense3=Dense(30)(dense2)
dense4=Dense(20, activation='relu')(dense3)
dense5=Dense(10)(dense4)
dense6=Dense(8, activation='relu')(dense5)
dense7=Dense(5)(dense6)
dense8=Dense(4)(dense7)
dense9=Dense(3)(dense8)
output1=Dense(1)(dense9)
model=Model(inputs=input1, outputs=output1)

# model=load_model("./_save/keras23_hamsu1_boston load_save practice.h5")

#3.
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  

############################################################################################################
import datetime
date=datetime.datetime.now()   
datetime = date.strftime("%m%d_%H%M")   #1206_0456
#print(datetime)

filepath='./_ModelCheckPoint/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'  # 2500-0.3724.hdf
model_path = "".join([filepath, 'k26_', datetime, '_', filename])
            #./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf
##############################################################################################################

es= EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.5, callbacks=[es,mcp])  #로스와 발로스 반환

model.save("./_save/keras27_1_save_model.h5")

#4.
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

print("============================1. 기본출력 ========================")
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

print("===============================================2. load_model 출력 ===========================")
model2=load_model('../study/_save/keras27_1_save_model.h5')
loss2=model2.evaluate(x_test, y_test) 
print('loss :', loss2)

y_predict2= model2.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict2)
print('r2스코어:', r2)

# print("==================================================3. ModelCheckPoint load 출력=======================")

# model3=load_model('./study/_ModelCheckPoint/keras26_3_MCP.hdf5')
# loss3=model3.evaluate(x_test, y_test) 
# print('loss :', loss3)

# y_predict3= model3.predict(x_test)

# from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
# r2=r2_score(y_test, y_predict3)
# print('r2스코어:', r2)




