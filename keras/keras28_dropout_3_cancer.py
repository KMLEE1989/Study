from tensorflow.keras.models import Sequential,load_model 
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dropout

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target


import numpy as np
from sklearn import datasets 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt 


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

#model=load_model("./_save/keras23_hamsu1_cancer_load_save.practice.h5")

input1=Input(shape=(30,))
dense1=Dense(100)(input1) 
dense2=Dense(80)(dense1)
drop2=Dropout(0.4)(dense2)
dense3=Dense(70, activation='relu')(drop2)
dense4=Dense(60)(dense3)
dense5=Dense(50)(dense4)
drop5=Dropout(0.2)(dense5)
dense6=Dense(10, activation='relu')(drop5)
output1=Dense(1, activation='sigmoid')(dense6)
model=Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  

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

es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)
start=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, mcp])  #얼리스타핑을 콜백으로 땡기겠다 리스트
end=time.time() - start

model.save("./_save/keras27_3_save_model.h5")

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)
y_predict= model.predict(x_test)
print(y_predict)

print("============================1. 기본출력 ========================")
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

# dropout
# loss : [0.26483431458473206, 0.9561403393745422]
# r2스코어: 0.8309625929558376

print("===============================================2. load_model 출력 ===========================")
model2=load_model('./_save/keras27_3_save_model.h5')
loss2=model2.evaluate(x_test, y_test) 
print('loss :', loss2)

y_predict2= model2.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict2)
print('r2스코어:', r2)

#미적용
# loss : [0.24079976975917816, 0.9385964870452881]
# r2스코어: 0.7847886866899058

# dropout
# loss : [0.34002283215522766, 0.9473684430122375]
# r2스코어: 0.8000613323704133

# print("==================================================3. ModelCheckPoint load 출력=======================")

# model3=load_model('./study/_ModelCheckPoint/keras26_3_MCP.hdf5')
# loss3=model3.evaluate(x_test, y_test) 
# print('loss :', loss3)

# y_predict3= model3.predict(x_test)

# from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
# r2=r2_score(y_test, y_predict3)
# print('r2스코어:', r2)
