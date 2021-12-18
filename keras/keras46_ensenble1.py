#1. 데이터
import numpy as np
x1=np.array([range(100), range(301,401)])   #삼성 저가 , 고가
x2=np.array([range(101,201), range(411, 511), range(100,200)]) #미국선물 시가, 고가, 종가


x1=np.transpose(x1)
x2=np.transpose(x2)

y=np.array(range(1001, 1101))    #삼성전자 종가

print(x1.shape, x2.shape, y.shape) #(100, 2) (100, 3) (100,)

from sklearn.model_selection import train_test_split

# x1=np.array(range(100))  
# x2=np.array(range(1,101))

x1_train, x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1,x2,y,train_size=0.7, shuffle=True, random_state=66)
# x1_test, x1_val, y_test, y_val=train_test_split(x1_test,y_test,train_size=0.7, shuffle=True, random_state=66)
# x2_test, x2_val, y_test, y_val=train_test_split(x2_test, y_test,train_size=0.7, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)
print(x2_test.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

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
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(7)(merge2)
last_output=Dense(1)(merge3)

model=Model(inputs=[input1, input2], outputs=last_output)

model.compile(loss='mse', optimizer='adam', matrics=['mse'])
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1)

results=model.evaluate([x1_test, x2_test], y_test)

print("===============================================")
print('비교체험:',  results)

'''
merge1=Concatenate()([output1, output2])#,zaxis=-1, **kwargs)#axis=-1,**kwargs)
merge2=Dense(10, activation='relu')(merge1)
merge3=Dense(7)(merge2)
last_output=Dense(1)(merge3)


model = Model(inputs=[input1, input2], outputs=[last_output])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y_train], epochs=100, batch_size=1, validation_data=([x1, x2],[y]))

mse=model.evaluate([x1_test, x2_test], [y_test], batch_size=1)

print('loss(mse) : ', mse) 
print('loss(mse) : ', mse[0])
print('loss(mse) : ', mse[1])
# print('loss(mse) : ', mse[2])
# print('loss(mse) : ', mse[3])
# print('loss(mse) : ', mse[4])
model.summary()

y_predict= model.predict([x1_test, x2_test])
print('PREDICT : ', y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
 
RMSE1=RMSE(y_test, y_predict)
print('RMSE(y_test) : ', RMSE1)
 
# R2 구하기
from sklearn.metrics import r2_score
def R2(y_test, y_predict):
    return r2_score(y_test, y_predict)
 
R2_1 = R2(y_test, y_predict)
print('R2(y_test) : ', R2_1)
print('AVG(R2) : ', (R2_1)/2)


# loss(mse) :  [0.030278079211711884, 0.030278079211711884]
# loss(mse) :  0.030278079211711884
# loss(mse) :  0.030278079211711884
# PREDICT :  [[1009.029  ]

# RMSE(y_test) :  0.17398233074476946
# R2(y_test) :  0.9999653814423005
# AVG(R2) :  0.49998269072115026
# Model: "model"


'''




