import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  

#1.데이터

x_train = np.load('./_save_npy/keras48_1_train_x.npy')
x_test=np.load('./_save_npy/keras48_1_test_x.npy')
y_train = np.load('./_save_npy/keras48_1_train_y.npy')
y_test=np.load('./_save_npy/keras48_1_test_y.npy')

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
model.add(Dense(2, activation='sigmoid'))

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

es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_accuracy', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)


#model.fit(xy_train[0][0], xy_train[0][1])
hist=model.fit(x_train, y_train,epochs=100, #steps_per_epoch=800, #전체데이터/batch=160/5=32)
               validation_split=0.3,
               #validation_steps=4,
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

# loss:  0.40780216455459595
# val_loss:  2.6422226428985596
# accuracy:  0.7142857313156128
# val_accuracy:  0.3333333432674408


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
        
# loss:  0.23094747960567474
# val_loss:  0.054401058703660965
# accuracy:  0.8571428656578064
# val_accuracy:  1.0
# 당신은 50.17 % 확률로 고양이 입니다
