#[실습] cifar10을 너서 완성할것!!!!

# vgg trainable : True / False
# Flatten / GAP 
# 위 4가지 결과와, 지금까지 본인이 cifar10을 돌려서 최고치가 나온가와 비교!!
#autokeras
# time 도 명시해라!!!

from sklearn.metrics import accuracy_score

#출력결과 : 
# time : ...
# loss : ...
# acc_score : ...
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, VGG19, ResNet101, InceptionV3
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)   # (50000, 10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (40000, 32, 32, 3) (40000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 10) 

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

n = x_train.shape[0]
x_train_reshape = x_train.reshape(n,-1) 
x_train_transform = scaler.fit_transform(x_train_reshape)
x_train = x_train_transform.reshape(x_train.shape) 

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

#x_test_transform = scaler.transform(x_test.reshape(m,-1))
#x_test = x_test_transform.reshape(x_test.shape)

#2. 모델구성
inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(32,32,3))

# vgg16.summary()
inceptionv3.trainable = True    # 가중치를 동결시킨다
# print(vgg16.weights)

model = Sequential()
model.add(inceptionv3)
model.add(Flatten())
#model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8)) 
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)  #-> 5번 만에 갱신이 안된다면 (factor=0.5) LR을 50%로 줄이겠다

start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))


# Traceback (most recent call last):
#   File "d:\Study\keras2\keras69_05_cifar100_InceptionV3.py", line 57, in <module>
#     inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(32,32,3))
#   File "C:\ProgramData\Anaconda3\envs\tf270gpu\lib\site-packages\keras\applications\inception_v3.py", line 127, in InceptionV3
#     input_shape = imagenet_utils.obtain_input_shape(
#   File "C:\ProgramData\Anaconda3\envs\tf270gpu\lib\site-packages\keras\applications\imagenet_utils.py", line 376, in obtain_input_shape
#     raise ValueError('Input size must be at least '
# ValueError: Input size must be at least 75x75; Received: input_shape=(32, 32, 3)