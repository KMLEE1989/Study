#모델링 구성!!!
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
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

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
#       dtype=int64))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#모델링 구성!! 시작!!!!
x_train, x_test_, y_train, y_test_ = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

model=Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(28,28,1)))     
model.add(Conv2D(10,(3,3), activation='relu'))                     
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))                     
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=1, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[es])


loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict=model.predict(x_test)
# loss :  0.41873008012771606
# accuracy :  0.8590999841690063
print(y_test)
print(y_test.shape, y_predict.shape)

y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import r2_score, accuracy_score

a=accuracy_score(y_test, y_predict)
print('accuracy score:' , a)
# accuracy score: 0.8299





