from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
import pandas as pd 
y = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42)

scaler =  MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(32,activation='relu',input_dim = 54)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(7, activation = 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 

start = time.time() 
model.fit(x_train, y_train, epochs=300, batch_size=50, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) # callbacks의 []는 다른게 들어갈수 있음
end = time.time() - start


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)

# learning_rate :  0.001
# loss :  0.6346
# accuracy :  0.7239
# 걸린시간 : 7614.0161

'''
Epoch 60/100
7437/7437 [==============================] - 5s 625us/step - loss: 0.6565 - accuracy: 0.7148 - val_loss: 0.6453 - val_accuracy: 0.7211
Restoring model weights from the end of the best epoch.
Epoch 00060: early stopping
3632/3632 [==============================] - 2s 427us/step - loss: 0.6448 - accuracy: 0.7214
loss :  0.6448085904121399
accuracy :  0.7213669419288635
[[0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]]
[[6.5158003e-13 6.9950908e-01 2.7308086e-01 5.5235351e-08 3.0022514e-12
  6.4606185e-04 1.4948830e-07 2.6763827e-02]
 [5.3100907e-10 7.0410617e-02 9.2861837e-01 6.0712275e-05 1.0949736e-05
  5.9415243e-04 2.5507639e-04 5.0149836e-05]
 [7.5671343e-13 7.6936799e-01 2.1677803e-01 4.0062818e-07 2.0808801e-10
  1.0680374e-03 2.3477378e-05 1.2762069e-02]
 [1.1070732e-09 6.8759859e-02 9.2618126e-01 1.0889668e-04 9.3934097e-05
  4.1897101e-03 5.1530253e-04 1.5103200e-04]
 [9.9579433e-12 4.2843568e-01 5.5917305e-01 1.8488308e-05 1.5769602e-10
  3.0524875e-03 3.5193960e-05 9.2851333e-03]
 [6.7367622e-12 2.7079758e-01 7.0719385e-01 1.5364105e-05 4.1406794e-09
  2.1669621e-02 2.4355085e-04 8.0088947e-05]
 [1.5816311e-10 1.5733100e-01 8.4114534e-01 1.4786504e-05 8.7535724e-08
  5.4727792e-04 3.8302613e-05 9.2318724e-04]]
'''

# batch_size의 디폴트 값은 32입니다.(batch_size=1로 돌렸을때의 1epoch당 돌아가야하는 값과 디폴트로 놓았을때의 값 비교 계산)