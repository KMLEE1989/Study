from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 

#1. 데이터
import numpy as np
x1=np.array([range(100), range(301,401)])   #삼성 저가 , 고가
x2=np.array([range(101,201), range(411, 511), range(100,200)]) #미국선물 시가, 고가, 종가


x1=np.transpose(x1)
x2=np.transpose(x2)

y1=np.array(range(1001, 1101))    #삼성전자 종가
y2=np.array(range(101,201))

print(x1.shape, x2.shape, y1.shape, y2.shape) #(100, 2) (100, 3) (100,)

from sklearn.model_selection import train_test_split

# x1=np.array(range(100))  
# x2=np.array(range(1,101))

x1_train, x1_test,x2_train,x2_test,y1_train,y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2,train_size=0.7, shuffle=True, random_state=66)
# x1_test, x1_val, y_test, y_val=train_test_split(x1_test,y_test,train_size=0.7, shuffle=True, random_state=66)
# x2_test, x2_val, y_test, y_val=train_test_split(x2_test, y_test,train_size=0.7, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)  #(70, 2) (30, 2)
print(x2_test.shape, x2_test.shape) #(30, 3) (30, 3)
print(y1_train.shape, y1_test.shape) #(70,) (30,)
print(y2_train.shape, y2_test.shape) #(#70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1=Input(shape=(2,))
dense1=Dense(5, activation='relu', name='dense1')(input1)
dense2=Dense(7, activation='relu', name='dense2')(dense1)
dense3=Dense(7, activation='relu', name='dense3')(dense2)
output1=Dense(7, activation='relu', name='dense4')(dense3)

#2-1 모델1
input2=Input(shape=(3,))
dense11=Dense(5, activation='relu', name='dense11')(input2)
dense12=Dense(10, activation='relu', name='dense12')(dense11)
dense13=Dense(7, activation='relu', name='dense13')(dense12)
dense14=Dense(7, activation='relu', name='dense14')(dense13)
output2=Dense(5, activation='relu', name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2])

#2-3 output 모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-3 output 모델1
output31 = Dense(7)(merge1)
output32 = Dense(21)(output31)
output33 = Dense(21, activation='relu')(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

# merge2 = Dense(10, activation='relu')(merge1)
# merge3 = Dense(7)(merge2)
# last_output=Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],[y1_train, y2_train], epochs=100, batch_size=8, validation_data=([x1, x2],[y1, y2])) 

# model.compile(loss='mse', optimizer='adam', matrics=['mse'])
# model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=8, verbose=1)

results=model.evaluate([x1_test, x2_test], [y1_test, y2_test])

print("===============================================")
print('비교체험:',  results)

"""
# input, output para 2
 
# 4. 평가 예측
mse = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)

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

es=EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1, restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_accuracy', mode='auto', verbose=1, 
                    save_best_only=True,filepath=model_path)


# in, out, merge 모델이 5개....라서 mse 5개?
print('loss(mse) : ', mse) # compile에서 lose = mse, loss와 mse둘다 만들 필요 없음
print('loss(mse) : ', mse[0])
print('loss(mse) : ', mse[1])
print('loss(mse) : ', mse[2])
print('loss(mse) : ', mse[3])
print('loss(mse) : ', mse[4])
 
y1_predict, y2_predict = model.predict([x1_test, x2_test])
print('PREDICT : ', y1_predict, y2_predict) #RMSE와 R2를 위해 쪼개기
 
 
''' 
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
 
RMSE1=RMSE(y1_test, y1_predict)
RMSE2=RMSE(y2_test, y2_predict)
print('RMSE(y1_test) : ', RMSE1)
print('RMSE(y2_test) : ', RMSE2)
print('AVG(RMSE) : ', (RMSE1+RMSE2)/2)
 
# R2 구하기
from sklearn.metrics import r2_score
def R2(y_test, y_predict):
    return r2_score(y_test, y_predict)
 
R2_1 = R2(y1_test, y1_predict)
R2_2 = R2(y2_test, y2_predict)
print('R2(y1_test) : ', R2_1)
print('R2(y2_test) : ', R2_2)
print('AVG(R2) : ', (R2_1+R2_2)/2)

model.summary()

'''
#mse
# loss(mse) :  [0.027071796357631683, 0.012304609641432762, 0.014767175540328026, 0.012304609641432762, 0.014767175540328026]
# loss(mse) :  0.027071796357631683
# loss(mse) :  0.012304609641432762
# loss(mse) :  0.014767175540328026
# loss(mse) :  0.012304609641432762
# loss(mse) :  0.014767175540328026
# PREDICT :  [[1008.9883 ]

#mae
# loss(mse) :  [0.0024154328275471926, 0.000258985033724457, 0.0021564478520303965, 0.015411376953125, 0.045301564037799835]
# loss(mse) :  0.0024154328275471926
# loss(mse) :  0.000258985033724457
# loss(mse) :  0.0021564478520303965
# loss(mse) :  0.015411376953125
# loss(mse) :  0.045301564037799835
# PREDICT :  [[1009.0098 ]

"""