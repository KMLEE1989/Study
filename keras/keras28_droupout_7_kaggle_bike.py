import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score,mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
 
path = "../_data/kaggle/bike/"
train = pd.read_csv(path+'train.csv')

test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+'sampleSubmission.csv')

x = train.drop(['datetime', 'casual' , 'registered', 'count'], axis=1)  #컬럼 삭제할때는 드랍에 액시스 1 준다   
test_file = test_file.drop(['datetime'], axis=1)

y=train['count']   

y=np.log1p(y) 

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_file=scaler.transform(test_file)

#2. 모델
model = Sequential() 
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
dropout=(Dropout(0.2))
model.add(Dense(110))
model.add(Dense(110, activation='relu'))
model.add(Dense(120))
dropout=(Dropout(0.4))
model.add(Dense(120))
model.add(Dense(110, activation='relu'))
model.add(Dense(110))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(1))

#3. compile
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

es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es,mcp])

model.save('./_save/keras27_7_save_model.h5')


#4.평가, 예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_pred= model.predict(x_test)

r2=r2_score(y_test, y_pred)
print('r2스코어:', r2)

rmse=RMSE(y_test, y_pred)
print('RMSE: ', rmse)


#result
print("============================1. 기본출력 ========================")
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)
rmse=RMSE(y_test, y_pred)
print('RMSE: ', rmse)





print("===============================================2. load_model 출력 ===========================")
model2=load_model('./_save/keras27_7_save_model.h5')
loss2=model2.evaluate(x_test, y_test) 
print('loss :', loss2)

y_predict2= model2.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict2)
print('r2스코어:', r2)
rmse=RMSE(y_test, y_pred)
print('RMSE: ', rmse)

#미적용
# loss : [0.9556612372398376, 0.5795981287956238]
# r2스코어: 0.14894408474201318

# dropout 적용
# loss : 1.3448148965835571
# r2스코어: 0.3137138958058263
# RMSE:  1.1596615838716267

# print("==================================================3. ModelCheckPoint load 출력=======================")

# model3=load_model('./study/_ModelCheckPoint/keras26_3_MCP.hdf5')
# loss3=model3.evaluate(x_test, y_test) 
# print('loss :', loss3)

# y_predict3= model3.predict(x_test)

# from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
# r2=r2_score(y_test, y_predict3)
# print('r2스코어:', r2)


