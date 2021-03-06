# 과제
# 남자 여자 데이터에 노이즈를 넣어서 
# 기미 주근깨 여드름 제거


import numpy as np, time
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 증폭하여 정의시킨다.
                                   vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   rotation_range=5, 
                                   zoom_range=1.2, 
                                   shear_range=0.7, 
                                   fill_mode='nearest',
                                   validation_split=0.3) 

train_generator = train_datagen.flow_from_directory( 
    'D:\_data\image\men_women\data', 
    target_size=(100,100), 
    batch_size=10, 
    class_mode='categorical', 
    subset='training')   
validation_generator = train_datagen.flow_from_directory( 
    'D:\_data\image\men_women\data', # same directory as training data 
    target_size=(100,100), 
    batch_size=10, 
    class_mode='categorical', 
    subset='validation')  
# Found 4634 images belonging to 3 classes.
# Found 1984 images belonging to 3 classes.

np.save('./_save_npy/keras48_4_train_x.npy', arr = train_generator[0][0])
np.save('./_save_npy/keras48_4_train_y.npy', arr = train_generator[0][1])
np.save('./_save_npy/keras48_4_test_x.npy', arr = validation_generator[0][0])
np.save('./_save_npy/keras48_4_test_y.npy', arr = validation_generator[0][1])

x_train = np.load("./_save_npy/keras48_4_train_x.npy")
x_test = np.load('./_save_npy/keras48_4_test_x.npy')
y_train = np.load('./_save_npy/keras48_4_train_y.npy')
y_test = np.load('./_save_npy/keras48_4_test_y.npy')

print(x_train.shape)  # (10, 100, 100, 3)
print(x_test.shape)   # (10, 100, 100, 3)
print(y_train.shape)  # (10,)
print(y_test.shape)   # (10,)

x_train = x_train.reshape(30,100,100).astype('float')/255
x_test = x_test.reshape(30,100,100).astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, UpSampling2D, MaxPool2D
import numpy as np
from tensorflow.keras.layers import Conv2D


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, input_shape=(100, 100, 1), kernel_size=3, padding='same', strides=1, activation='relu'))
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
    
model = autoencoder(hidden_layer_size=154)      # pca 95% -> 154

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(x_train_noised, x_train, epochs=2000)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15),
      (ax6, ax7, ax8, ax9, ax10)) = \
          plt.subplots(3, 5, figsize=(20, 7))
          
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력)이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(100,100), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(100,100), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
# 오코인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(100,100), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.tight_layout()
plt.show()
