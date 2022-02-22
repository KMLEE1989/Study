from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, ResNet101, DenseNet121, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip = True,  
    vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode= 'nearest')   
 
test_datagen = ImageDataGenerator(rescale=1./255) 

xy_train = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/training_set/training_set/', 
    target_size = (150,150), 
    batch_size=8200,
    class_mode = 'binary',
    shuffle= True) 

#print(xy_train)
# Found 8005 images belonging to 2 classes.
#<keras.preprocessing.image.DirectoryIterator object at 0x000002AA555DD0A0>

xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set/test_set/',
    target_size = (150,150),
    batch_size = 11000,
    class_mode = 'binary') 

#print(xy_test)
# Found  10028 images belonging to 2 classes. (8005+2023)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0] 
y_test = xy_test[0][1]

#print(x_train.shape, y_train.shape) # (8005, 150, 150, 3) (8005,)
#print(x_test.shape, y_test.shape) # (2023, 150, 150, 3) (2023,)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

augment_size = 3000
randidx = np.random.randint(x_train.shape[0], size = augment_size)


# print(x_train.shape[0])  # 8005
# print(randidx) # [4718 1326 5431 ... 2839 2663 3981]
# print(np.min(randidx), np.max(randidx)) # 0 8000


x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
#print(x_augmented.shape) # (3000, 150, 150, 3)
print(y_augmented.shape) # (3000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False).next()[0] #save_to_dir='../_temp/').next()[0]  # 증폭

# print(x_augmented)
# print(x_augmented.shape)  # (3000, 150, 150, 3)

# concatenate, merge 합치기 위한 것!

x_train = np.concatenate((x_train, x_augmented))  
y_train = np.concatenate((y_train, y_augmented))  
#print(x_train.shape, y_train.shape)  # (11005, 150, 150, 3) (11005,)
#print(np.unique(y_train)) # [0. 1.]

eb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(150,150,3))

eb0.trainable = True

#2. 모델
model = Sequential()
model.add(eb0)
model.add(Conv2D(128, (2,2), input_shape= (150,150,3), activation= 'relu'))
model.add(Conv2D(64, (2,2), padding='same'))  
model.add(Conv2D(16, (2,2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=20, validation_split=0.3, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss[0])    
print('acc: ', loss[1]) 

y_predict=model.predict(x_test)

# print(y_test)
# print(y_test.shape, y_predict.shape)

y_predict=np.argmax(y_predict,axis=1)
#y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import r2_score, accuracy_score

a=accuracy_score(y_test, y_predict)
print('acc score:' , a)   


# loss:  0.6939378976821899
# acc:  0.5007414817810059
# acc score: 0.49975284231339595


# loss:  1.2145861387252808
# acc:  0.5007414817810059
# acc score: 0.49975284231339595
