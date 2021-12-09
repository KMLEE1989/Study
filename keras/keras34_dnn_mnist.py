from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu

#실습!!!
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= x_train.reshape(60000, 28, 28,1)
x_test=x_test.reshape(10000,28,28,1)
print(x_train.shape)
print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
                                                                             
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train=x_train.reshape(60000,-1)   #-1이라는 의미는 1열 이후의 모든 차원을 한줄로 만들었다는 이야기
x_test=x_test.reshape(10000,-1)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)



#2. 모델구성
model=Sequential()
#moel.add(Dense(64, input_shape(28*28,))) #1차원 형태로 60000을 제외한 28*28의 매트릭스가 평평하게 되어서 들어옴
model.add(Dense(64, input_shape=(784,)))
model.add(Dense(80, activation='relu'))
model.add(Dense(50))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])


loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


# loss :  0.15063810348510742
# accuracy :  0.9624000191688538