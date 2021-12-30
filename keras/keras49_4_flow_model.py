from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5, 
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0]) #60000
print(randidx)  #[19399 10094 12869 ... 49696 12015 32824]
print(np.min(randidx), np.max(randidx))  #1 59999

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape) #(40000, 28, 28)
print(y_augmented.shape) #(40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

x_train= x_train.reshape(60000, 28, 28,1)
x_test= x_test.reshape(x_test.shape[0],28,28,1)


x_augmented = train_datagen.flow(x_augmented, y_augmented,#np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False
                                 ).next()[0]

print(x_augmented) 
print(x_augmented.shape)


x_train=np.concatenate((x_train, x_augmented)) #(40000, 28, 28)
y_train = np.concatenate((y_train, y_augmented)) #(40000,)
print(x_train)
print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

#점심특선 1. x_augmented 10개와 x_train 10개를 비교하는 이미지 출력할것
          # subplot(2,10,?) 사용

#print(x_augmented[randidx])

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_augmented[0][i], cmap='gray')
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_train[0][i], cmap='gray')
plt.show()

