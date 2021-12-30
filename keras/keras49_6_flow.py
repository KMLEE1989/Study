from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

test_datagen = ImageDataGenerator(
    rescale=1./255,)

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


xy_train = train_datagen.flow(x_augmented, y_augmented,#np.zeros(augment_size),
                                 batch_size=32, #augment_size, 
                                 shuffle=False
                                 )# .next()

xy_test = test_datagen.flow(x_test, y_test,#np.zeros(augment_size),
                                 batch_size=32, #augment_size, 
                                 shuffle=False
                                 )# .next()

# print(xy_train)
# print(xy_train[0].shape, xy_train[1].shape) #(40000, 28, 28, 1) (40000,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(len(xy_train))

model.fit_generator(xy_train, epochs=10, steps_per_epoch=len(xy_train))

#4. 평가

loss, acc = model.evaluate_generator(xy_test)
print("Accuracy : ", str(np.round(acc ,2)*100)+ "%")
#TypeError: 'float' object is not subscriptable

test_loss, test_acc = model.evaluate(xy_test)
print('loss :', test_loss)
print('acc :', test_acc)

