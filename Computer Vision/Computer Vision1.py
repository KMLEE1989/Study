from xml.etree.ElementInclude import include
import tensorflow as tf

# K, D, N = (3, 16, 32)

# kernels_size = [K, K, D, N]
# glorot_uni_initializer = tf.initializers.GlorotUniform()

# kernels = tf.Variable(glorot_uni_initializer(kernels_size), trainable=True, name="filters")

# bias = tf.Variable(tf.zeros(shape=[N]), trainable=True, name='bias')

# @tf.function

# def conv_layer(x, kernels, bias, s):
#     z= tf.nn.conv2d(x, kernels, strides=[1, s, s, 1], padding='VALID')
#     return tf.nn.relu(z + bias)
    
    
# class SimpleCNN(tf.keras.layers.Layer):
#     def __init__(self, num_kernels=32, kernel_size=(3,3), stride=1):
#         super().__init__()
#         self.num_kernels = num_kernels
#         self.kernel_size = kernels_size
#         self.stride = stride
        
#     def build(self, input_shape):
#         input_channels = input_shape[-1]
        
#         kernels_shape = (*self.kernel_size, input_channels, self.num_kernels)
#         glorot_init = tf.initializers.GlorotUniform()
#         self.kernels = self.add_weight(
#             name='kernels' , shape=kernels_shape, initializer=glorot_init, trainable=True)
#         self.bias = self.add_weight(name='bias', shape=(self.num_kernels, ), initializer='random_normal', trainable=True)
        
#     def call(self, inputs):
#         return conv_layer(inputs, self.kernels, self.bias, self.stride)  
    


# from tensorflow.keras.layers import Conv2D

# s = 1

# conv = Conv2D(filters=N, kernel_size=(K,K), strides=s, padding='valid', activation='relu')

# from tensorflow.keras.layers import AvgPool2D, MaxPool2D

# K, S=(3,1)

# avg_pool = AvgPool2D(pool_size=K, strides=[S, S], padding='valid')
# max_pool = MaxPool2D(pool_size=K, strides=[S, S], padding='valid')


# from tensorflow.keras.layers import Dense

# output_size = 64

# fc = Dense(units=output_size, activation='relu')

# from tensorflow.keras import Model
# from tensorflow.keras.layers import Conv2D,Flatten, Dense, MaxPooling2D
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.layers import Dense,Flatten
# from tensorflow.keras.callbacks import EarlyStopping

# import numpy as np
# from torch import conv2d

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train, x_test = x_train[..., np.newaxis], x_test[...,np.newaxis]

# # print(x_train.shape)
# # print(y_train.shape)
# # print(x_test.shape)
# # print(y_test.shape)

# # (60000, 28, 28, 1)
# # (60000,)
# # (10000, 28, 28, 1)
# # (10000,)

# # print(x_train[0, :, :, 0])

# x_train, x_test = x_train/255.0, x_test/255.0
# # print(x_train[0,:,:,0])

# num_classes = 10
# epochs = 100
# batch_size = 32

# class le_net5(Model):
#     def __init__(self, num_classes):
#         super(le_net5, self).__init__()
#         self.conv1 = Conv2D(6, kernel_size=(5,5), padding='same', activation='relu')
#         self.conv2 = Conv2D(16, kernel_size=(5,5), activation='relu')
#         self.max_pool = MaxPooling2D(pool_size=(2,2))
#         self.flatten = Flatten()
#         self.dense1 = Dense(120, activation='relu')
#         self.dense2 = Dense(84, activation='relu')
#         self.dense3 = Dense(num_classes, activation='softmax')
        
#     def call(self, input_data):
#         x = self.max_pool(self.conv1(input_data))
#         x = self.max_pool(self.conv2(x))
#         x = self.flatten(x)
#         x = self.dense3(self.dense2(self.dense1(x)))
        
#         return x


# model = le_net5(num_classes)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=10, batch_size=8, validation_split=0.5, callbacks=[es])

import tensorflow as tf
# vgg_net = tf.keras.applications.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# vgg_net.summary()

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input

input_shape = (28,28,3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 24, 24, 32)        2432

#  max_pooling2d (MaxPooling2D  (None, 12, 12, 32)       0
#  )

#  flatten (Flatten)           (None, 4608)              0

#  dense (Dense)               (None, 10)                46090

# =================================================================
# Total params: 48,522
# Trainable params: 48,522
# Non-trainable params: 0
# _________________________________________________________________

def build_model():
    inputs = input(shape=input_shape)
    conv1 = Conv2D(32, kernel_size=(5,5))(inputs)
    maxpool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    predictions = Dense(10, activation = 'softmax')(Flatten()(maxpool1))
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate

def naive_inception_block(prev_layer, filters=[64, 128, 32]):
    conv1X1 = Conv2D(filters[0], kernel_size=(1,1), padding='same', activation='relu')(prev_layer)
    conv3X3 = Conv2D(filters[1], kernel_size=(3,3), padding='same', activation='relu')(prev_layer)
    conv5X5 = Conv2D(filters[2], kernel_size=(5,5), padding='same', activation='relu')(prev_layer)
    max_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    
    return concatenate([conv1X1, conv3X3, conv5X5, max_pool], axis=-1)

inceptionV3_net =tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

inceptionV3_net.summary()

from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, add

def residual_block_basic(x, filters, kernel_size=3, strides=1):
    conv_1 = Conv2D(filters=filters, kernel_size=kernel_size, paddint='same', strides=strides)(x)
    bn_1 = BatchNormalization(axis=-1)(conv_1)
    act_1 = Activation('relu')(bn_1)
    conv_2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)(act_1)
    residual = BatchNormalization(axis=-1)(conv_2)
    
    shortcut = x if strides == 1 else Conv2D(filters, kernel_size=1, padding='valid', strides=strides)(x)
    
    return Activation('relu')(add([shortcut, residual]))


resnet50 = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

resnet50.summary()

# Model: "resnet50"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_2 (InputLayer)           [(None, 224, 224, 3  0           []
#                                 )]

#  conv1_pad (ZeroPadding2D)      (None, 230, 230, 3)  0           ['input_2[0][0]']

#  conv1_conv (Conv2D)            (None, 112, 112, 64  9472        ['conv1_pad[0][0]']
#                                 )

#  conv1_bn (BatchNormalization)  (None, 112, 112, 64  256         ['conv1_conv[0][0]']
#                                 )

#  conv1_relu (Activation)        (None, 112, 112, 64  0           ['conv1_bn[0][0]']
#                                 )

#  pool1_pad (ZeroPadding2D)      (None, 114, 114, 64  0           ['conv1_relu[0][0]']
#                                 )

#  pool1_pool (MaxPooling2D)      (None, 56, 56, 64)   0           ['pool1_pad[0][0]']

#  conv2_block1_1_conv (Conv2D)   (None, 56, 56, 64)   4160        ['pool1_pool[0][0]']

#  conv2_block1_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_1_conv[0][0]']
#  ization)

#  conv2_block1_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_1_bn[0][0]']
#  n)

#  conv2_block1_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block1_1_relu[0][0]']

#  conv2_block1_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_2_conv[0][0]']
#  ization)

#  conv2_block1_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_2_bn[0][0]']
#  n)

#  conv2_block1_0_conv (Conv2D)   (None, 56, 56, 256)  16640       ['pool1_pool[0][0]']

#  conv2_block1_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block1_2_relu[0][0]']

#  conv2_block1_0_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block1_0_conv[0][0]']
#  ization)

#  conv2_block1_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block1_3_conv[0][0]']
#  ization)

#  conv2_block1_add (Add)         (None, 56, 56, 256)  0           ['conv2_block1_0_bn[0][0]',
#                                                                   'conv2_block1_3_bn[0][0]']

#  conv2_block1_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block1_add[0][0]']

#  conv2_block2_1_conv (Conv2D)   (None, 56, 56, 64)   16448       ['conv2_block1_out[0][0]']

#  conv2_block2_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_1_conv[0][0]']
#  ization)

#  conv2_block2_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_1_bn[0][0]']
#  n)

#  conv2_block2_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block2_1_relu[0][0]']

#  conv2_block2_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_2_conv[0][0]']
#  ization)

#  conv2_block2_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_2_bn[0][0]']
#  n)

#  conv2_block2_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block2_2_relu[0][0]']

#  conv2_block2_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block2_3_conv[0][0]']
#  ization)

#  conv2_block2_add (Add)         (None, 56, 56, 256)  0           ['conv2_block1_out[0][0]',
#                                                                   'conv2_block2_3_bn[0][0]']

#  conv2_block2_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block2_add[0][0]']

#  conv2_block3_1_conv (Conv2D)   (None, 56, 56, 64)   16448       ['conv2_block2_out[0][0]']

#  conv2_block3_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block3_1_conv[0][0]']
#  ization)

#  conv2_block3_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block3_1_bn[0][0]']
#  n)

#  conv2_block3_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block3_1_relu[0][0]']

#  conv2_block3_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block3_2_conv[0][0]']
#  ization)

#  conv2_block3_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block3_2_bn[0][0]']
#  n)

#  conv2_block3_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block3_2_relu[0][0]']

#  conv2_block3_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block3_3_conv[0][0]']
#  ization)

#  conv2_block3_add (Add)         (None, 56, 56, 256)  0           ['conv2_block2_out[0][0]',
#                                                                   'conv2_block3_3_bn[0][0]']

#  conv2_block3_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block3_add[0][0]']

#  conv3_block1_1_conv (Conv2D)   (None, 28, 28, 128)  32896       ['conv2_block3_out[0][0]']

#  conv3_block1_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_1_conv[0][0]']
#  ization)

#  conv3_block1_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_1_bn[0][0]']
#  n)

#  conv3_block1_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block1_1_relu[0][0]']

#  conv3_block1_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_2_conv[0][0]']
#  ization)

#  conv3_block1_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_2_bn[0][0]']
#  n)

#  conv3_block1_0_conv (Conv2D)   (None, 28, 28, 512)  131584      ['conv2_block3_out[0][0]']

#  conv3_block1_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block1_2_relu[0][0]']

#  conv3_block1_0_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block1_0_conv[0][0]']
#  ization)

#  conv3_block1_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block1_3_conv[0][0]']
#  ization)

#  conv3_block1_add (Add)         (None, 28, 28, 512)  0           ['conv3_block1_0_bn[0][0]',
#                                                                   'conv3_block1_3_bn[0][0]']

#  conv3_block1_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block1_add[0][0]']

#  conv3_block2_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block1_out[0][0]']

#  conv3_block2_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_1_conv[0][0]']
#  ization)

#  conv3_block2_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_1_bn[0][0]']
#  n)

#  conv3_block2_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block2_1_relu[0][0]']

#  conv3_block2_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_2_conv[0][0]']
#  ization)

#  conv3_block2_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_2_bn[0][0]']
#  n)

#  conv3_block2_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block2_2_relu[0][0]']

#  conv3_block2_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block2_3_conv[0][0]']
#  ization)

#  conv3_block2_add (Add)         (None, 28, 28, 512)  0           ['conv3_block1_out[0][0]',
#                                                                   'conv3_block2_3_bn[0][0]']

#  conv3_block2_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block2_add[0][0]']

#  conv3_block3_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block2_out[0][0]']

#  conv3_block3_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_1_conv[0][0]']
#  ization)

#  conv3_block3_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_1_bn[0][0]']
#  n)

#  conv3_block3_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block3_1_relu[0][0]']

#  conv3_block3_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_2_conv[0][0]']
#  ization)

#  conv3_block3_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_2_bn[0][0]']
#  n)

#  conv3_block3_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block3_2_relu[0][0]']

#  conv3_block3_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block3_3_conv[0][0]']
#  ization)

#  conv3_block3_add (Add)         (None, 28, 28, 512)  0           ['conv3_block2_out[0][0]',
#                                                                   'conv3_block3_3_bn[0][0]']

#  conv3_block3_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block3_add[0][0]']

#  conv3_block4_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block3_out[0][0]']

#  conv3_block4_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_1_conv[0][0]']
#  ization)

#  conv3_block4_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_1_bn[0][0]']
#  n)

#  conv3_block4_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block4_1_relu[0][0]']

#  conv3_block4_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_2_conv[0][0]']
#  ization)

#  conv3_block4_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_2_bn[0][0]']
#  n)

#  conv3_block4_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block4_2_relu[0][0]']

#  conv3_block4_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block4_3_conv[0][0]']
#  ization)

#  conv3_block4_add (Add)         (None, 28, 28, 512)  0           ['conv3_block3_out[0][0]',
#                                                                   'conv3_block4_3_bn[0][0]']

#  conv3_block4_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block4_add[0][0]']

#  conv4_block1_1_conv (Conv2D)   (None, 14, 14, 256)  131328      ['conv3_block4_out[0][0]']

#  conv4_block1_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_1_conv[0][0]']    
#  ization)

#  conv4_block1_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_1_bn[0][0]']
#  n)

#  conv4_block1_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block1_1_relu[0][0]']

#  conv4_block1_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_2_conv[0][0]']
#  ization)

#  conv4_block1_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_2_bn[0][0]']
#  n)

#  conv4_block1_0_conv (Conv2D)   (None, 14, 14, 1024  525312      ['conv3_block4_out[0][0]']
#                                 )

#  conv4_block1_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block1_2_relu[0][0]']
#                                 )

#  conv4_block1_0_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block1_0_conv[0][0]']
#  ization)                       )

#  conv4_block1_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block1_3_conv[0][0]']
#  ization)                       )

#  conv4_block1_add (Add)         (None, 14, 14, 1024  0           ['conv4_block1_0_bn[0][0]',
#                                 )                                 'conv4_block1_3_bn[0][0]']

#  conv4_block1_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block1_add[0][0]']
#                                 )

#  conv4_block2_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block1_out[0][0]']

#  conv4_block2_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_1_conv[0][0]']
#  ization)

#  conv4_block2_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_1_bn[0][0]']
#  n)

#  conv4_block2_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block2_1_relu[0][0]']

#  conv4_block2_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_2_conv[0][0]']
#  ization)

#  conv4_block2_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_2_bn[0][0]']
#  n)

#  conv4_block2_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block2_2_relu[0][0]']
#                                 )

#  conv4_block2_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block2_3_conv[0][0]']
#  ization)                       )

#  conv4_block2_add (Add)         (None, 14, 14, 1024  0           ['conv4_block1_out[0][0]',
#                                 )                                 'conv4_block2_3_bn[0][0]']

#  conv4_block2_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block2_add[0][0]']
#                                 )

#  conv4_block3_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block2_out[0][0]']

#  conv4_block3_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_1_conv[0][0]']
#  ization)

#  conv4_block3_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_1_bn[0][0]']
#  n)

#  conv4_block3_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block3_1_relu[0][0]']

#  conv4_block3_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_2_conv[0][0]']
#  ization)

#  conv4_block3_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_2_bn[0][0]']
#  n)

#  conv4_block3_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block3_2_relu[0][0]']
#                                 )

#  conv4_block3_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block3_3_conv[0][0]']
#  ization)                       )

#  conv4_block3_add (Add)         (None, 14, 14, 1024  0           ['conv4_block2_out[0][0]',
#                                 )                                 'conv4_block3_3_bn[0][0]']

#  conv4_block3_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block3_add[0][0]']
#                                 )

#  conv4_block4_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block3_out[0][0]']

#  conv4_block4_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_1_conv[0][0]']
#  ization)

#  conv4_block4_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_1_bn[0][0]']
#  n)

#  conv4_block4_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block4_1_relu[0][0]']

#  conv4_block4_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_2_conv[0][0]']
#  ization)

#  conv4_block4_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_2_bn[0][0]']
#  n)

#  conv4_block4_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block4_2_relu[0][0]']
#                                 )

#  conv4_block4_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block4_3_conv[0][0]']
#  ization)                       )

#  conv4_block4_add (Add)         (None, 14, 14, 1024  0           ['conv4_block3_out[0][0]',
#                                 )                                 'conv4_block4_3_bn[0][0]']

#  conv4_block4_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block4_add[0][0]']
#                                 )

#  conv4_block5_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block4_out[0][0]']

#  conv4_block5_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_1_conv[0][0]']
#  ization)

#  conv4_block5_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_1_bn[0][0]']      
#  n)

#  conv4_block5_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block5_1_relu[0][0]']

#  conv4_block5_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_2_conv[0][0]']
#  ization)

#  conv4_block5_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_2_bn[0][0]']
#  n)

#  conv4_block5_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block5_2_relu[0][0]']
#                                 )

#  conv4_block5_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block5_3_conv[0][0]']
#  ization)                       )

#  conv4_block5_add (Add)         (None, 14, 14, 1024  0           ['conv4_block4_out[0][0]',
#                                 )                                 'conv4_block5_3_bn[0][0]']

#  conv4_block5_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block5_add[0][0]']
#                                 )

#  conv4_block6_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block5_out[0][0]']

#  conv4_block6_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block6_1_conv[0][0]']
#  ization)

#  conv4_block6_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block6_1_bn[0][0]']
#  n)

#  conv4_block6_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block6_1_relu[0][0]']

#  conv4_block6_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block6_2_conv[0][0]']
#  ization)

#  conv4_block6_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block6_2_bn[0][0]']
#  n)

#  conv4_block6_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block6_2_relu[0][0]']
#                                 )

#  conv4_block6_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block6_3_conv[0][0]']
#  ization)                       )

#  conv4_block6_add (Add)         (None, 14, 14, 1024  0           ['conv4_block5_out[0][0]',
#                                 )                                 'conv4_block6_3_bn[0][0]']

#  conv4_block6_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block6_add[0][0]']
#                                 )

#  conv5_block1_1_conv (Conv2D)   (None, 7, 7, 512)    524800      ['conv4_block6_out[0][0]']

#  conv5_block1_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_1_conv[0][0]']
#  ization)

#  conv5_block1_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_1_bn[0][0]']
#  n)

#  conv5_block1_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block1_1_relu[0][0]']

#  conv5_block1_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_2_conv[0][0]']
#  ization)

#  conv5_block1_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_2_bn[0][0]']
#  n)

#  conv5_block1_0_conv (Conv2D)   (None, 7, 7, 2048)   2099200     ['conv4_block6_out[0][0]']

#  conv5_block1_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block1_2_relu[0][0]']

#  conv5_block1_0_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block1_0_conv[0][0]']
#  ization)

#  conv5_block1_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block1_3_conv[0][0]']
#  ization)

#  conv5_block1_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_0_bn[0][0]',
#                                                                   'conv5_block1_3_bn[0][0]']

#  conv5_block1_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block1_add[0][0]']

#  conv5_block2_1_conv (Conv2D)   (None, 7, 7, 512)    1049088     ['conv5_block1_out[0][0]']

#  conv5_block2_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_1_conv[0][0]']
#  ization)

#  conv5_block2_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_1_bn[0][0]']
#  n)

#  conv5_block2_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block2_1_relu[0][0]']

#  conv5_block2_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_2_conv[0][0]']
#  ization)

#  conv5_block2_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_2_bn[0][0]']
#  n)

#  conv5_block2_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block2_2_relu[0][0]']

#  conv5_block2_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block2_3_conv[0][0]']
#  ization)

#  conv5_block2_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_out[0][0]',
#                                                                   'conv5_block2_3_bn[0][0]']

#  conv5_block2_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block2_add[0][0]']

#  conv5_block3_1_conv (Conv2D)   (None, 7, 7, 512)    1049088     ['conv5_block2_out[0][0]']

#  conv5_block3_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_1_conv[0][0]']
#  ization)

#  conv5_block3_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_1_bn[0][0]']
#  n)

#  conv5_block3_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block3_1_relu[0][0]']

#  conv5_block3_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_2_conv[0][0]']
#  ization)

#  conv5_block3_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_2_bn[0][0]']      
#  n)

#  conv5_block3_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block3_2_relu[0][0]']

#  conv5_block3_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block3_3_conv[0][0]']
#  ization)

#  conv5_block3_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block2_out[0][0]',
#                                                                   'conv5_block3_3_bn[0][0]']

#  conv5_block3_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block3_add[0][0]']

#  avg_pool (GlobalAveragePooling  (None, 2048)        0           ['conv5_block3_out[0][0]']
#  2D)

#  predictions (Dense)            (None, 1000)         2049000     ['avg_pool[0][0]']

# ==================================================================================================
# Total params: 25,636,712
# Trainable params: 25,583,592
# Non-trainable params: 53,120
# __________________________________________________________________________________________________

    