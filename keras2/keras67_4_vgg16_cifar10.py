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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.layers import Dense,Flatten, Dropout, Conv2D
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)   # (50000, 10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (40000, 32, 32, 3) (40000, 10)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 10) 

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

n = x_train.shape[0]  # 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1)   #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
scaler.fit(x_train_reshape)              
x_train_transform = scaler.fit_transform(x_train_reshape)  #0~255 -> 0~1
x_train = x_train_transform.reshape(x_train.shape)    #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

#x_test_transform = scaler.transform(x_test.reshape(m,-1))
#x_test = x_test_transform.reshape(x_test.shape)


#2. 모델구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

# vgg16.summary()
vgg16.trainable = False    # 가중치를 동결시킨다
# print(vgg16.weights)

model = Sequential()
model.add(vgg16)
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 
# mcp = ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1,
#                       save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌

start = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))


######################################## 1. vgg trainable : True, Flatten ###############################################
# learning_rate :  0.0001
# loss :  0.7118
# accuracy :  0.839
# 걸린시간 : 129.6671
######################################## 2. vgg trainable : False, Flatten ###############################################
# learning_rate :  0.0001
# loss :  1.1958
# accuracy :  0.5918
# 걸린시간 : 1008.6033
######################################## 3. vgg trainable : True, GAP ###############################################
# learning_rate :  0.0001
# loss :  0.6321
# accuracy :  0.8244
# 걸린시간 : 1256.4673
######################################## 4. vgg trainable : False, GAP ###############################################
# learning_rate :  0.0001
# loss :  1.201
# accuracy :  0.5903
# 걸린시간 : 1007.4124

# 결과 비교
# vgg trainable : True / False
# Flatten / Global Average Pooling
# 위 4개 조합해서 최고결과 뽑고 이전 최고치 acc0.65와 비교



