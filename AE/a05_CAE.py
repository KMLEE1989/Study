# 2번 카피해서 복붙
# CNN으로 딥하게 구성

# Conv2D
# Maxpool
# Conv2D
# Maxpool
# Conv2D    -> Encoder

# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D    -> Decoder

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D

#1. 데이터

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28).astype('float') / 255
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28 * 28).astype('float') / 255
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape)

#2. 모델구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, UpSampling2D, MaxPool2D

# def autoencoder(hidden_layer_size):
#     model = Sequential([
#         Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'),
#         Dense(units=784, activation='sigmoid')
#         ])
#     return model

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, input_shape=(28, 28, 1), kernel_size=3, padding='same', strides=1, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(1, 3, padding='same'))
    
    
    return model

model = autoencoder(hidden_layer_size=32)

#3. 컴파일, 훈련

model.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=20, batch_size=128, validation_split=0.2)

#4. 평가, 예측
output = model.predict(x_test)
print(output.shape)

# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))
    
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[i], cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]], cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()