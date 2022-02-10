from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import time
from tensorflow.keras.layers import GlobalAvgPool2D,MaxPool2D, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3),input_shape = (28,28,1)))
model.add(Conv2D(16,(3,3), activation='relu'))
model.add(GlobalAvgPool2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
import time
from tensorflow.keras.optimizers import Adam
##############################################
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
############################################################################################
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
############################################################################################
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',
                   verbose=1, restore_best_weights=False)
ReduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, mode = 'min',
                             verbose=3, factor=0.5)

from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(log_dir='D:/Study/_save/_graph', histogram_freq=0, write_graph=True, write_images=True,
                 write_steps_per_second=False, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None)

start = time.time()
hist = model.fit(x_train, y_train, epochs=3, batch_size=32,
          validation_split=0.25, callbacks=[es, ReduceLR, tb])
end = time.time()

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict(x_test)
print('걸린시간 :', round(end - start,4))
###############################시각화####################################
import matplotlib.pyplot as plt
plt.figure(figsize = (9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss' )
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss' )
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'loss' )
plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc' )
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])


plt.show()