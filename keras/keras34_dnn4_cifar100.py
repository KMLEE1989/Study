from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler 

(x_train, y_train), (x_test, y_test)=cifar100.load_data()

print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

x_train=x_train.reshape(50000, 32, 32, 3)
x_test=x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train, return_counts=True))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train=x_train.reshape(50000,-1)   
x_test=x_test.reshape(10000,-1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

model=Sequential()
model.add(Dense(64, input_shape=(3072,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

es=EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1, 
                    save_best_only=True,filepath=model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
model.save('./_save/keras33_save_model.h5')

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :  4.6053466796875
# accuracy :  0.009999999776482582