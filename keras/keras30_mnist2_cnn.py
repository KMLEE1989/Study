##########################################################################################
#parameter가 50이 나오는 이유:  #3 x 3(필터 크기) x 3 (#입력 채널(RGB)) x 32(#출력 채널) + 32(출력 채널 bias) = 896이 됩니다. #파라미터 구하는 법
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 3, 3, 5)           205
=================================================================
Total params: 255
Trainable params: 255
Non-trainable params: 0
'''

# model.add(Conv2D(a, kernel_size = (b,c), input_shape = (q,w,e))) 
# 1) a = 출력 채널 
# 2) b,c = 필터, 커넬 사이즈와 같은 말이며, 이미지의 특징을 찾아내기 위한
# 공용 파라미터이다. 
# 3) e : 입력 채널, RGB 
# - 이미지 픽셀 하나하나는 실수이며, 컬러사진을 표현하기 위해서는 RGB 3개의
# 실수로 표현해야 한다. 

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train= x_train.reshape(60000, 28, 28,1)   #  =(60000, 28, 14, 2)           (60000, 28, 28, 1, 1) #다 곱했을때 똑같아야 한다 
x_test=x_test.reshape(10000,28,28,1)
print(x_train.shape)   #(60000, 28, 28, 1)

print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#      dtype=int64))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train, x_test_, y_train, y_test_ = train_test_split(x_train, y_train, 
         train_size = 0.8, shuffle = True, random_state = 66)

model=Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(28,28,1)))     #9,9,10
model.add(Conv2D(10,(3,3), activation='relu'))                     # 7,7,5
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))                     #6,6,7
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])


loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])




# 평가지표 acc
# 0.98 이상

#result
# loss :  0.0803435668349266
# accuracy :  0.9761000275611877





# print(x_train[0])
# print('y_train[0]번째 값 : ' ,  y_train[0])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()

