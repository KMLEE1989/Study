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
from tensorflow.keras.applications import VGG16, VGG19, ResNet101
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)   # (50000, 10)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

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
resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3))

# vgg16.summary()
resnet101.trainable = True    # 가중치를 동결시킨다
# print(vgg16.weights)

model = Sequential()
model.add(resnet101)
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


######################################## 1. ResNet101 trainable : True, Flatten ###############################################
# learning_rate :  0.0001
# loss :  3.9636
# accuracy :  0.419
# 걸린시간 : 419.4084
######################################## 2. ResNet101 : False, Flatten ###############################################
# learning_rate :  0.0001
# loss :  4.5323
# accuracy :  0.0169
# 걸린시간 : 162.5846
######################################## 3. ResNet101 : True, GAP ###############################################
# learning_rate :  0.0001
# loss :  3.6736
# accuracy :  0.1476
# 걸린시간 : 188.3779
######################################## 4. ResNet101 : False, GAP ###############################################
# learning_rate :  0.0001
# loss :  5.7067
# accuracy :  0.3988
# 걸린시간 : 424.8629
######################################## 5. preprocessing input ##########################################################
# learning_rate :  0.0001
# loss :  4.5019
# accuracy :  0.4131
# 걸린시간 : 422.3497
