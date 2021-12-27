import numpy as np
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  


# np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
x_test=np.load('./_save_npy/keras47_5_test_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
y_test=np.load('./_save_npy/keras47_5_test_y.npy')

print(x_train)
print(x_train.shape) #(160, 150, 150, 3)
print(y_train.shape) #(160,)

#모델 구성하시오!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (2,2),strides=2, padding='same', input_shape=(150,150,3)))
model.add(Conv2D(32,(1,1), activation='relu'))
model.add(MaxPooling2D())   
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

############################################################################################################
import datetime
date=datetime.datetime.now()   
datetime = date.strftime("%m%d_%H%M")   #1206_0456
#print(datetime)

filepath='./_ModelCheckPoint/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'  # 2500-0.3724.hdf
model_path = "".join([filepath, 'k26_', datetime, '_', filename])
            #./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf
##############################################################################################################

es = EarlyStopping(monitor='val_loss', patience=100, mode='max', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_accuracy', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)


#model.fit(xy_train[0][0], xy_train[0][1])
hist=model.fit(x_train, y_train, epochs=1000, #steps_per_epoch=32, #전체데이터/batch=160/5=32)
                    validation_split=0.3,
                    # validation_steps=4,
                    callbacks=[es, mcp]
                    )

accuracy=hist.history['accuracy']
val_accuracy=hist.history['val_accuracy']
loss=hist.history['loss']
val_loss=hist.history['val_loss']

#점심때 그래프 그려보아요!

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('accuracy: ', accuracy[-1])
print('val_accuracy: ', val_accuracy[-1])


import matplotlib.pyplot as plt

epochs=range(1, len(loss)+1)

plt.plot(epochs, loss, 'r--', label="loss")
plt.plot(epochs, val_loss, 'r:', label="val_loss")
plt.plot(epochs, accuracy, 'b--', label="accuracy")
plt.plot(epochs, val_accuracy, 'b:', label="val_accuracy")

plt.grid()
plt.legend()
plt.show()
