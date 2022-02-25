import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

n = 10
plt.figure(figsize=(20,2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.title('Original + Noise')
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
plt.show()

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        
        self.encoder = Sequential([Input(shape=(28, 28, 1)), Conv2D(16,(3,3), activation='relu', padding='same', strides=2), Conv2D(8,(3,3), activation='relu', padding='same', strides=2)])
        self.decoder = Sequential([Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu',padding='same'),
                                   Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

autoencoder.fit(x_train_noisy, x_train, epochs=10, shuffle=True, validation_data=(x_test_noisy, x_test))

autoencoder.encoder.summary()

autoencoder.decoder.summary()

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n=10
plt.figure(figsize=(20,4))

for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.title('Original + Noise')
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    bx = plt.subplot(2, n, i+1+n)
    plt.title('Reconstructed (Denoise)')
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 14, 14, 16)        160

#  conv2d_1 (Conv2D)           (None, 7, 7, 8)           1160

# =================================================================
# Total params: 1,320
# Trainable params: 1,320
# Non-trainable params: 0
# _________________________________________________________________
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d_transpose (Conv2DTra  (None, 14, 14, 8)        584
#  nspose)

#  conv2d_transpose_1 (Conv2DT  (None, 28, 28, 16)       1168
#  ranspose)

#  conv2d_2 (Conv2D)           (None, 28, 28, 1)         145

# =================================================================
# Total params: 1,897
# Trainable params: 1,897
# Non-trainable params: 0
# _________________________________________________________________