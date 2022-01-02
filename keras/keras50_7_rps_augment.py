from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest')     

test_datagen = ImageDataGenerator(rescale = 1./255)             

xy_train = train_datagen.flow_from_directory(
    '../_data/image/rps/rps/', 
    target_size = (100,100), 
    batch_size=2600,
    class_mode = 'categorical',
    shuffle= True) 

#Found 2520 images belonging to 3 classes.


xy_test = test_datagen.flow_from_directory(
    '../_data/image/rps/rps/',
    target_size = (100,100),
    batch_size = 2600,
    class_mode = 'categorical') 

#Found 2520 images belonging to 3 classes.


x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0] 
y_test = xy_test[0][1]
#print(x_train.shape, y_train.shape) # (2520, 100, 100, 3) (2520, 3)
#print(x_test.shape, y_test.shape) # (2520, 100, 100, 3) (2520, 3)


augment_size = 3000
randidx = np.random.randint(x_train.shape[0], size = augment_size)


#print(x_train.shape[0])  # 2520
#print(randidx) # [2078 1535  194 ... 1311  241  661]
#print(np.min(randidx), np.max(randidx)) # 0 2519


x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
# print(x_augmented.shape) # (3000, 100, 100, 3)
#print(y_augmented.shape) # (3000, 3)


x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False).next()[0] #save_to_dir='../_temp/').next()[0]  # 증폭

# print(x_augmented) <keras.preprocessing.image.NumpyArrayIterator object at 0x0000026E9CD6FD00>
#print(x_augmented.shape)  # (3000, 100, 100, 3)


# concatenate, merge 합치기 위한 것!

x_train = np.concatenate((x_train, x_augmented))  
y_train = np.concatenate((y_train, y_augmented))  
#print(x_train.shape, y_train.shape)  # (5520, 100, 100, 3) (5520, 3)
#print(np.unique(y_train)) # [0. 1.]

#2. 모델
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape= (100,100,3), activation= 'relu'))
model.add(Conv2D(64, (2,2), padding='same'))  
model.add(Conv2D(32, (2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=20, validation_split=0.2, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss[0])    
print('acc: ', loss[1]) 

y_predict=model.predict(x_test)

# print(y_test)
# print(y_test.shape, y_predict.shape)

y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import r2_score, accuracy_score

accuracy=accuracy_score(y_test, y_predict)
print('accuracy score:' , accuracy) 


# loss:  2.239055633544922
# acc:  0.5142857432365417
# accuracy score: 0.5142857142857142  