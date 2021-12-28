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
    '../_data/image/cat_dog/training_set/training_set/', 
    target_size=(150,150),
    batch_size=10, 
    class_mode='categorical',
    shuffle=True,
)     

# print(xy_train)

xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set/test_set/',
    target_size=(150,150),
    batch_size=10,
    class_mode='categorical', 
    shuffle=True,
)    #Found 120 images belonging to 2 classes.


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

np.save('./_save_npy/keras48_1_train_x.npy', arr=xy_train[0][0])
np.save('./_save_npy/keras48_1_train_y.npy', arr=xy_train[0][1])
np.save('./_save_npy/keras48_1_test_x.npy', arr=xy_test[0][0])
np.save('./_save_npy/keras48_1_test_y.npy', arr=xy_test[0][1])

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
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

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

es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_accuracy', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

#model.fit(xy_train[0][0], xy_train[0][1])
hist=model.fit_generator(xy_train, epochs=10, steps_per_epoch=800, #전체데이터/batch=160/5=32)
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

# loss:  0.6880254149436951
# val_loss:  0.6556395292282104
# accuracy:  0.5452157855033875
# val_accuracy:  0.6499999761581421

#model.evaluate_generator

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = 'D:\\_data\\image\\NA\\JAE HO LEE.jpg'
model_path = 'D:\\Study\\_ModelCheckPoint\\k26_1228_1902_0004-0.6925.hdf5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(150,150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        plt.append('off')
        plt.show()
    
    return img_tensor

if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    dog = pred[0][0]*100
    cat = pred[0][1]*100
    if cat > dog:
        print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
    else:
        print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
        
        