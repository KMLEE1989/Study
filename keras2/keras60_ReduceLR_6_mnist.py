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
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=80, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr])
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

# learning_rate :  0.001
# loss :  0.101
# accuracy :  0.9763
# 걸린시간 : 261.2944
  
'''
loss :  0.08725973963737488
accuracy :  0.9771999716758728
time :  330.9091305732727
768/768 [==============================] - 34s 45ms/step - loss: 0.0289 - accuracy: 0.9909 - val_loss: 0.1502 - val_accuracy: 0.9789
Restoring model weights from the end of the best epoch.
Epoch 00024: early stopping
375/375 [==============================] - 3s 7ms/step - loss: 0.0897 - accuracy: 0.9743
loss :  0.08968890458345413
accuracy :  0.9742500185966492
Epoch 00084: early stopping
375/375 [==============================] - 2s 4ms/step - loss: 0.1367 - accuracy: 0.9647
loss :  0.1367303729057312
accuracy :  0.9646666646003723
Restoring model weights from the end of the best epoch.
Epoch 00104: early stopping
375/375 [==============================] - 2s 4ms/step - loss: 0.1206 - accuracy: 0.9704
loss :  0.12055526673793793
accuracy :  0.9704166650772095
Restoring model weights from the end of the best epoch.
Epoch 00108: early stopping
375/375 [==============================] - 1s 3ms/step - loss: 0.1168 - accuracy: 0.9711
loss :  0.11677785217761993
accuracy :  0.9710833430290222
# 트레인 발리데이션 테스트 평가
# 평가지표 accuracy 0.98 이상
#print(x_train[0])
#print('y_train[0]번째의 값 : ', y_train[0])
#import matplotlib.pyplot as plt
#plt.imshow(x_train[0], 'gray')
#plt.show()
'''