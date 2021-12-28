import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.backend import binary_crossentropy 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  
 
# 1. 데이터 
 
#이미지 폴더 정의 # D:\_data\image\horse-or-human 
train_datagen = ImageDataGenerator( 
    rescale = 1./255,               
    horizontal_flip = True,         
    vertical_flip= True,            
    width_shift_range = 0.1,        
    height_shift_range= 0.1,        
    rotation_range= 5, 
    zoom_range = 1.2,               
    shear_range=0.7, 
    fill_mode = 'nearest', 
    validation_split=0.3           
    )                   # set validation split  
 
train_generator = train_datagen.flow_from_directory( 
    'D:\_data\image\horse-or-human\horse-or-human', 
    target_size=(150,150), 
    batch_size=10, 
    class_mode='categorical', 
    subset='training') # set as training data 
#Found 719 images belonging to 2 classes. 

#print(train_generator)
 
validation_generator = train_datagen.flow_from_directory( 
    'D:\_data\image\horse-or-human\horse-or-human', # same directory as training data 
    target_size=(150,150), 
    batch_size=10, 
    class_mode='categorical', 
    subset='validation') # set as validation data 
#print(validation_generator)
#Found 308 images belonging to 2 classes.


#print(train_generator[0][0].shape)  # (10, 150, 150, 3)
print(validation_generator[0][0].shape) # (10, 150, 150, 3)

test_datagen = ImageDataGenerator(rescale = 1./255) 

np.save('./_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0]) 
np.save('./_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1]) 
np.save('./_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0]) 
np.save('./_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1]) 

 
# 2. 모델 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout 
 
 
model = Sequential() 
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu')) 
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(Flatten()) 
model.add(Dense(16, activation='relu')) 
model.add(Dense(2, activation='softmax')) 
 
 
# 3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy']) 
 
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
                    save_best_only=False,
                    filepath=model_path)
 
 
hist = model.fit_generator(train_generator, epochs = 10, steps_per_epoch = 72,  
                    validation_data = validation_generator, 
                    validation_steps = 4,
                    callbacks=[es,mcp]) 
 

accuracy = hist.history['accuracy'] 
val_acc = hist.history['val_accuracy'] 
loss = hist.history['loss'] 
val_loss = hist.history['val_loss'] 
 
# 그래프 그려 보기 
print('loss:', loss[-1]) 
print('val_loss:', val_loss[-1]) 
print('accuracy:', accuracy[-1]) 
print('val_accuracy:',val_acc [-1]) 
 
epochs = range(1, len(loss)+1) 
import matplotlib.pyplot as plt 

plt.plot(epochs, loss, 'r--', label = 'loss') 
plt.plot(epochs, val_loss, 'r:', label = 'val_loss') 
plt.plot(epochs, accuracy, 'b--', label = 'accuracy') 
plt.plot(epochs, val_acc, 'b:', label = 'val_accuracy') 
 
plt.grid() 
plt.legend() 
plt.show 
 
# summarize history for accuracy 
plt.plot(accuracy) 
plt.plot(val_acc) 
plt.title('model accuracy') 
plt.ylabel('accuracy') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='upper left') 
plt.show() 
# summarize history for loss 
plt.plot(loss) 
plt.plot(val_loss) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='upper left') 
plt.show()

# loss: 0.25273552536964417
# val_loss: 2.1172337532043457
# acc: 0.8998609185218811
# val_acc: 0.6000000238418579

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = 'D:\\_data\\image\\NA\\JAE HO LEE.jpg'
model_path = 'D:\\Study\\_ModelCheckPoint\\k26_1228_1751_0010-1.5944.hdf5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(150,150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        # plt.append('off')
        plt.show()
    
    return img_tensor

if __name__ == '__main__':
    # model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    horse = pred[0][0]*100
    human = pred[0][1]*100
    if human > horse:
        print(f"당신은 {round(human,2)} % 확률로 말 입니다")
    else:
        print(f"당신은 {round(horse,2)} % 확률로 인간 입니다")
        
# loss: 0.6929413676261902
# val_loss: 0.6964551210403442
# accuracy: 0.5132128000259399
# val_accuracy: 0.42500001192092896
# 당신은 51.03 % 확률로 말 입니다