from email import generator
from re import X
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = Input(shape=(latent_dim,))

x = Dense(128 * 16 * 16)(generator_input)
x = LeakyReLU()(x)
x = Reshape((16,16,128))(x)

x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(256,4, strides=2, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(channels, 7, activation='tanh', padding='same')(x)

generator = Model(generator_input, x)
generator.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 32)]              0
# _________________________________________________________________
# dense (Dense)                (None, 32768)             1081344   
# _________________________________________________________________
# leaky_re_lu (LeakyReLU)      (None, 32768)             0
# _________________________________________________________________
# reshape (Reshape)            (None, 16, 16, 128)       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 16, 16, 256)       819456
# _________________________________________________________________
# leaky_re_lu_1 (LeakyReLU)    (None, 16, 16, 256)       0
# _________________________________________________________________
# conv2d_transpose (Conv2DTran (None, 32, 32, 256)       1048832
# _________________________________________________________________
# leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 256)       0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 32, 32, 256)       1638656
# _________________________________________________________________
# leaky_re_lu_3 (LeakyReLU)    (None, 32, 32, 256)       0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 32, 32, 256)       1638656
# _________________________________________________________________
# leaky_re_lu_4 (LeakyReLU)    (None, 32, 32, 256)       0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 32, 32, 3)         37635
# =================================================================
# Total params: 6,264,579
# Trainable params: 6,264,579
# Non-trainable params: 0
# _________________________________________________________________

discriminator_input = Input(shape=(height, width, channels))

x = Conv2D(128,3)(discriminator_input)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(1, activation='sigmoid')(x)

discriminator = Model(discriminator_input, x)
discriminator.summary()

# _________________________________________________________________
# Model: "model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_2 (InputLayer)         [(None, 32, 32, 3)]       0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 30, 30, 128)       3584
# _________________________________________________________________
# leaky_re_lu_5 (LeakyReLU)    (None, 30, 30, 128)       0
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 14, 14, 128)       262272
# _________________________________________________________________
# leaky_re_lu_6 (LeakyReLU)    (None, 14, 14, 128)       0
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 6, 6, 128)         262272
# _________________________________________________________________
# leaky_re_lu_7 (LeakyReLU)    (None, 6, 6, 128)         0
# _________________________________________________________________
# conv2d_7 (Conv2D)            (None, 2, 2, 128)         262272
# _________________________________________________________________
# leaky_re_lu_8 (LeakyReLU)    (None, 2, 2, 128)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 512)               0
# _________________________________________________________________
# dropout (Dropout)            (None, 512)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 513
# =================================================================
# Total params: 790,913
# Trainable params: 790,913
# Non-trainable params: 0
# _________________________________________________________________


from tensorflow.keras.optimizers import RMSprop
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

discriminator_optimizer = RMSprop(learning_rate = 0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

gan_optimizer = RMSprop(learning_rate=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image

(x_train, y_train), (_, _) = cifar10.load_data()

x_train = x_train[y_train.flatten()==6]

x_train = x_train.reshape((x_train.shape[0], ) + (height, width, channels)).astype('float32')/255.

iterations = 10000
batch_size = 20

save_dir = 'D:\Study'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    
    stop = start + batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])
    
    labels = np.concatenate([np.ones((batch_size,1)),
                             np.zeros((batch_size,1))])
    labels += 0.05 * np.random.random(labels.shape)
    
    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1)) 
    
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets) 
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
        
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        
        print('\nstep: {}'.format(step))
        print(' discriminator loss: {}'.format(d_loss))
        print(' adversarial loss: {}'.format(a_loss))
        
        img = image.array_to_img(generated_images[0] * 255.,scale=False)
        img.save(os.path.join(save_dir, 'generated_frog'+str(step)+'.png'))
        
        img = image.array_to_img(real_images[0] * 255.,scale=False)
        img.save(os.path.join(save_dir, 'real_frog'+str(step)+'.png'))
        

import matplotlib.pyplot as plt

random_latent_vectors = np.random.normal(size=(10, latent_dim))

generated_images = generator.predict(random_latent_vectors)

print(generated_images.shape)

plt.figure(figsize=(20,8))
for i in range(generated_images.shape[0]):
    plt.subplot(2,5, i+1)
    img = image.array_to_img(generated_images[i]*255.,scale=False)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    
plt.show()
