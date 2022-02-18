import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.utils import to_categorical # 1부터 시작된다. 
# one hot은 0부터
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000,28, 28, 1).astype('float32')/255.

x = tf.placeholder(tf.float32,[None, 28, 28, 1])
y = tf.placeholder(tf.float32,[None, 10])

#2. 모델 구성

w1 = tf.get_variable('w1', shape = [2,2,1,64])
                                #  [커널 사이즈 = (2,2,1), output]
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'VALID')
                                #  shape 맞춰주기 위해 허수를 채워줬다. >> 앞 뒤 두 개의 1
# model.add(Conv2d(filters = 64, kernel_size = (2,2), strides = (1,1), padding = 'valid',
#                           input_shape = (28, 28, 1)))
# 커널 사이즈는 가중치였다.

print(w1)  # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)  # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)  # 커널 사이즈 때문에 한 개가 줄었다 28>27



