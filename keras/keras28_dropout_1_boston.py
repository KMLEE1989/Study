#과적합 방지에 쓸 수 있는 것 dropout

from enum import auto
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time

from tensorflow.python.keras.backend import dropout

datasets = load_boston()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

model = Sequential() 
model.add(Dense(40, input_dim=13))  #노드의 32개
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dropout(0.3))  # 노드의 21개
model.add(Dense(20))
model.add(Dropout(0.1)) #노드의 18개
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

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


start=time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp])  #로스와 발로스 반환
end=time.time() - start

print("걸린시간 :" , round(end, 3), '초')

model.save("../study/_save/keras26_4_save_model.h5")

#model=load_model('../study/_save/keras26_1_save_model.h5')
#model=load_model("./study/_ModelCheckPoint/keras26_4_MCP.hdf5")

#4. 퍙가, 예측

print("============================1. 기본출력 ========================")
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

# 미적용
# loss : 46.80183410644531
# r2스코어: 0.4335091058466005

#Dropout적용
# loss : 101.03067016601562
# r2스코어: -0.22287836596640953
print("===============================================2. load_model 출력 ===========================")
model2=load_model('../study/_save/keras26_4_save_model.h5')
loss2=model2.evaluate(x_test, y_test) 
print('loss :', loss2)

y_predict2= model2.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict2)
print('r2스코어:', r2)

# loss : 46.80183410644531
# r2스코어: 0.4335091058466005

# dropout
# loss : 80.07756042480469
# r2스코어: 0.030738712739218355

# print("==================================================3. ModelCheckPoint load 출력=======================")

# model3=load_model('./study/_ModelCheckPoint/keras26_3_MCP.hdf5')
# loss3=model3.evaluate(x_test, y_test) 
# print('loss :', loss3)

# y_predict3= model3.predict(x_test)

# from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
# r2=r2_score(y_test, y_predict3)
# print('r2스코어:', r2)
