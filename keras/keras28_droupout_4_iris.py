#relu를 사용하였을때 데이터 성능 향상이 뚜렷했으며 특히 로스 감소와 accuracy의 증폭값이 매우 컸다. Robustscaler를 사용했을때 두렷한 성능향상을 보였다. 


import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model 
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

y=to_categorical(y)
print(y)
print(y.shape)  #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

#scaler = MinMaxScaler()
#scaler=StandardScaler()
scaler=RobustScaler()
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model=load_model
input1 = Input(shape=(4,)) 
dense1=Dense(10)(input1)
dense2=Dense(10)(dense1)
drop2=Dropout(0.2)(dense2)
dense3=Dense(70,activation='relu')(drop2)
dense4=Dense(60)(dense3)
drop4=Dropout(0.3)(dense4)
dense5=Dense(50, activation='relu')(dense4)
dense6=Dense(10)(dense5)
output1=Dense(3)(dense6)
model=Model(inputs=input1, outputs=output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, mcp])  #얼리스타핑을 콜백으로 땡기겠다 리스트

model.save("./_save/keras27_4_save_model.h5")

#model=load_model("./_save/keras23_hamsu1_iris_load_save.practice.h5")

loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])

results=model.predict(x_test[:5])
print(y_test[:5])
print(results)


print("============================1. 기본출력 ========================")
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

print("===============================================2. load_model 출력 ===========================")
model2=load_model('./_save/keras27_4_save_model.h5')
loss2=model2.evaluate(x_test, y_test) 
print('loss :', loss2)

y_predict2= model2.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict2)
print('r2스코어:', r2)

#미적용
# loss : [5.90996789932251, 0.36666667461395264]
# r2스코어: -21.81256507809755

# dropout 적용
# loss : [1.5321308374404907, 0.9333333373069763]
# r2스코어: -14.164206273185144

# print("==================================================3. ModelCheckPoint load 출력=======================")

# model3=load_model('./study/_ModelCheckPoint/keras26_3_MCP.hdf5')
# loss3=model3.evaluate(x_test, y_test) 
# print('loss :', loss3)

# y_predict3= model3.predict(x_test)

# from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
# r2=r2_score(y_test, y_predict3)
# print('r2스코어:', r2)
