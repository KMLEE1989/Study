import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  

#1.데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5, 
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen=ImageDataGenerator(
    rescale=1./255
)
    
#D:\_data\image\brain


xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/', 
    target_size=(150,150),
    batch_size=5, 
    class_mode='binary',
    shuffle=True,
)     #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary', 
    shuffle=True,
)    #Found 120 images belonging to 2 classes.

'''
print(xy_train)
#<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022087FB6F40>

#from sklearn.datasets import load_boston
#datasets = load_boston()
#print(datasets)

#print(xy_train[31])  #마지막 배치
# print(xy_train[0][0])
# print(xy_train[0][1])
# print(xy_train[0][2]) #error
print(xy_train[0][0].shape, xy_train[0][1].shape) #(5, 150, 150, 3) #(5,)


print(type(xy_train)) #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

#2. 모델
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
hist=model.fit_generator(xy_train, epochs=1000, steps_per_epoch=32, #전체데이터/batch=160/5=32)
                    validation_data=xy_test,
                    validation_steps=4,
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
'''

# summarize history for accuracy
# plt.plot(accuracy)
# plt.plot(val_accuracy)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()









