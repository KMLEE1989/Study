from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 

#1. 데이터
import numpy as np
x1=np.array([range(100), range(301,401)])   #삼성 저가 , 고가
#x2=np.array([range(101,201), range(411, 511), range(100,200)]) #미국선물 시가, 고가, 종가


x1=np.transpose(x1)
#x2=np.transpose(x2)

y1=np.array(range(1001, 1101))    #삼성전자 종가
y2=np.array(range(101,201))
y3=np.array(range(401, 501))

print(x1.shape, y1.shape, y2.shape, y3.shape) #(100, 2) (100,) (100,) (100,)


from sklearn.model_selection import train_test_split

# x1=np.array(range(100))  
# x2=np.array(range(1,101))

x1_train, x1_test,y1_train,y1_test, y2_train, y2_test,y3_train,y3_test = train_test_split(x1,y1,y2,y3, train_size=0.7, shuffle=True, random_state=66)
# x1_test, x1_val, y_test, y_val=train_test_split(x1_test,y_test,train_size=0.7, shuffle=True, random_state=66)
# x2_test, x2_val, y_test, y_val=train_test_split(x2_test, y_test,train_size=0.7, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)  #(70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)  #(70,) (30,)
print(y2_train.shape, y2_test.shape)  #(70,) (30,)
print(y3_train.shape, y3_test.shape)  #(70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1=Input(shape=(2,))
dense1=Dense(5, activation='relu', name='dense1')(input1)
dense2=Dense(7, activation='relu', name='dense2')(dense1)
dense3=Dense(7, activation='relu', name='dense3')(dense2)
output1=Dense(7, activation='relu', name='dense4')(dense3)

'''
#2-1 모델1
input2=Input(shape=(3,))
dense11=Dense(5, activation='relu', name='dense11')(input2)
dense12=Dense(10, activation='relu', name='dense12')(dense11)
dense13=Dense(7, activation='relu', name='dense13')(dense12)
dense14=Dense(7, activation='relu', name='dense14')(dense13)
output2=Dense(5, activation='relu', name='output2')(dense14)
'''
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1])

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

output41=Dense(7)(merge1)
output42=Dense(21)(output41)
output43=Dense(21,activation='relu')(output42)
output44=Dense(11,activation='relu')(output43)
last_output3=Dense(1)(output44)

# merge2 = Dense(10, activation='relu')(merge1)
# merge3 = Dense(7)(merge2)
# last_output=Dense(1)(merge3)

model = Model(inputs=[input1], outputs=[last_output1, last_output2, last_output3])

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train],[y1_train, y2_train, y3_train], epochs=100, batch_size=8, validation_data=([x1],[y1, y2,y3]), verbose=1) 

# model.compile(loss='mse', optimizer='adam', matrics=['mse'])
# model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=8, verbose=1)

results=model.evaluate([x1_test], [y1_test, y2_test,y3_test])

print("===============================================")
print('비교체험:',  results)


loss=model.evaluate([x1_test], [y1_test, y2_test,y3_test], batch_size=1)
print('loss:',loss)
# print('loss(mse) : ', mse) 
# print('loss(mse) : ', mse[0])
# print('loss(mse) : ', mse[1])
# print('loss(mse) : ', mse[2])
# print('loss(mse) : ', mse[3])
# print('loss(mse) : ', mse[4])
#model.summary()

y_predict= model.predict([x1_test])
#print('PREDICT : ', y1_predict,y2_predict,y3_predict)

# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y1_predict,y2_predict,y3_predict):
#     return np.sqrt(mean_squared_error(y_test, y1_predict,y2_predict,y3_predict))
 
# RMSE1=RMSE(y1_test, y1_predict)
# RMSE2=RMSE(y2_test, y2_predict)
# RMSE3=RMSE(y3_test, y3_predict)
# print('RMSE(y1_test) : ', RMSE1)
# print('RMSE(y2_test) : ', RMSE1)
# print('RMSE(y3_test) : ', RMSE1)
 
# R2 구하기
from sklearn.metrics import r2_score
# def R2([y1_test, y1_pred],[y1_test,y2_pred],[y3_test,y3_pred]):
#     return r2_score(y_test, y_predict)

r2_1=r2_score(y1_test, y_predict[0])
r2_2=r2_score(y2_test, y_predict[0])
r2_3=r2_score(y3_test, y_predict[0])
print("r2스코어1:", r2_1)
print("r2스코어2:", r2_2)
print("r2스코어3:", r2_3)

# loss(mse) :  [0.030278079211711884, 0.030278079211711884]
# loss(mse) :  0.030278079211711884
# loss(mse) :  0.030278079211711884
# PREDICT :  [[1009.029  ]

# RMSE(y_test) :  0.17398233074476946
# R2(y_test) :  0.9999653814423005
# AVG(R2) :  0.49998269072115026
# Model: "model"



