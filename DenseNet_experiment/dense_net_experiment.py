import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.test.is_gpu_available()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

'''
< DenseNet Architecture>
- Dense Connectivity pattern
- Dense-121, Dense-169, Dense-201, Dense-264
- Implement Dense-121 (6, 12, 24, 16)
'''

def DenseNet(x):
    # input = 224 x 224 x 3
    k = 32  # Grow Rate
    compression = 0.5   # compression factor

    # 1. Convolution
    x = layers.Conv2D(k * 2, (7, 7), strides=2, padding='same', input_shape=(224, 224, 3))(x)    # 112x112x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 2. Pooling
    x = layers.MaxPool2D((3, 3), 2, padding='same')(x)  # 56x56x64

    # 3. Dense Block (1)
    for i in range(6) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)    # 56x56x128
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)  # 56x56x32
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])  # 96 -> 128 -> 160 -> 192 -> 224 -> 256

    # 4. Transition Layer (1)
    current_shape = int(x.shape[-1]) # 56x56x256
    x = layers.Conv2D(int(current_shape * compression), (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)   # 28x28

    # 5. Dense Block (2)
    for i in range(12) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])

    # 6. Transition Layer (2)
    current_shape = int(x.shape[-1])
    x = layers.Conv2D(int(current_shape * compression), (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)   # 14x14

    # 7. Dense Block (3)
    for i in range(24) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])

    # 8. Transition Layer (3)
    current_shape = int(x.shape[-1])
    x = layers.Conv2D(int(current_shape * compression), (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)   # 7x7

    # 9. Dense Block (4)
    for i in range(16) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])

    # 10. Classification Layer
    x = layers.GlobalAveragePooling2D()(x)
    # classes = 2 (softmax)
    x = layers.Dense(2, activation='softmax')(x)

    return x


# Parameter
batch_size = 32
epoch = 50
learning_rate = 0.001

# Dataset (Kaggle Cat and Dog Dataset)
dataset_path = os.path.join('/home/kellybjs/Cat_Dog_Dataset')
train_dataset_path = dataset_path + '/train_set'
train_data_generator = ImageDataGenerator(rescale=1. / 255)
train_dataset = train_data_generator.flow_from_directory(train_dataset_path,
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size=batch_size,
                                                         class_mode='categorical')

valid_dataset_path = dataset_path + '/validation_set'
valid_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_dataset = valid_data_generator.flow_from_directory(valid_dataset_path,
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size=batch_size,
                                                         class_mode='categorical')


# Train
input_shape = layers.Input(shape=(224, 224, 3), dtype='float32', name='input')
model = tf.keras.Model(input_shape, DenseNet(input_shape))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
model.summary()
train = model.fit_generator(train_dataset, epochs=epoch, validation_data=valid_dataset)

# Accuracy graph
plt.figure(1)
plt.plot(train.history['acc'])
plt.plot(train.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('DenseNet_Accuracy_1.png')

# Loss graph
plt.figure(2)
plt.plot(train.history['loss'])
plt.plot(train.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('DenseNet_Loss_1.png')
