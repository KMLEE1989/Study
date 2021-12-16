from tensorflow.python.keras.backend import softmax
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Flatten, Conv1D
import time

datasets=fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

print(x.shape, y.shape)  #(581012, 54) (581012,) 
print(y)#->feature를 구해 보도록 하구요
print(np.unique(y)) #->[1 2 3 4 5 6 7] 이렇게 나옵니다.  

y=to_categorical(y)
print(y)
print(y.shape) #(581012, 8)

x=x.reshape(581012,54,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

# scaler = MinMaxScaler()
# #scaler=StandardScaler()
# #scaler=RobustScaler()
# #scaler=MaxAbsScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

#그리고 우리는 train과 test의 값중 test 값을 사용할 거랍니다. 
print(x_train.shape, y_train.shape) #->(464809, 54) (464809, 8)
print(x_test.shape, y_test.shape) #-> (116203, 54) (116203, 8)

model = Sequential() 
model.add(Conv1D(10, 2, input_shape=(54,1)))  #(116203, 54) 
model.add(Flatten())
model.add(Dense(10))                                               #54는 input의 노드가 되구요
model.add(Dense(70, activation='relu'))
model.add(Dense(60))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(8, activation='softmax'))  

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

start=time.time()
model.fit(x_train, y_train, epochs=100, batch_size=80, validation_split=0.2, callbacks=[es])
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:10])
print(y_test[:10])
print(results)

# Conv1D
# 걸린시간:   668.318
# 3632/3632 [==============================] - 2s 576us/step - loss: 0.4195 - accuracy: 0.8220
# loss : 0.41951891779899597
# accuracy: 0.8219925761222839


# LSTM
# loss : 1.080786943435669
# accuracy: 0.4941352605819702