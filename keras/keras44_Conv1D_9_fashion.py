from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Flatten, Conv1D
import time 

#실습!!

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)



x_train=x_train.reshape(60000, 28, 28,1)      # (60000, 28, 28) (60000,)
x_test=x_test.reshape(10000,28,28,1)          #( 10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))   #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
 #     dtype=int64))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train=x_train.reshape(60000,784,1)   
x_test=x_test.reshape(10000,784,1)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)



model=Sequential()
model.add(Conv1D(64, 2, input_shape=(784,1)))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

start=time.time()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[es])
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#loss :  0.3624224066734314
#accuracy :  0.8729000091552734

#Conv1D
# 걸린시간:   203.372
# 313/313 [==============================] - 1s 4ms/step - loss: 0.5501 - accuracy: 0.8075
# loss :  0.5500937104225159
# accuracy :  0.8075000047683716