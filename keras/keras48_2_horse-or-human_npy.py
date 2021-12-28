import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.backend import binary_crossentropy 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint 

x_train=np.load('./_save_npy/keras48_2_train_x.npy')
x_test=np.load('./_save_npy/keras48_2_train_x.npy')
y_train=np.load('./_save_npy/keras48_2_train_y.npy')
y_test=np.load('./_save_npy/keras48_2_train_y.npy')

# 2. 모델 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout 
 
 
model = Sequential() 
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu')) 
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(Flatten()) 
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 


 
# 3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy']) 

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

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_accuracy', mode='auto', verbose=1, save_best_only=True, filepath=model_path)

hist = model.fit(x_train, y_train, epochs = 20, validation_split=0.3, callbacks=[es, mcp]) 

accuracy = hist.history['accuracy'] 
val_acc = hist.history['val_accuracy'] 
loss = hist.history['loss'] 
val_loss = hist.history['val_loss'] 
 
# 그래프 그려 보기 
print('loss:', loss[-1]) 
print('val_loss:', val_loss[-1]) 
print('acc:', accuracy[-1]) 
print('val_acc:',val_acc [-1]) 
 
epochs = range(1, len(loss)+1) 
import matplotlib.pyplot as plt 
plt.plot(epochs, loss, 'r--', label = 'loss') 
plt.plot(epochs, val_loss, 'r:', label = 'val_loss') 
plt.plot(epochs, accuracy, 'b--', label = 'acc') 
plt.plot(epochs, val_acc, 'b:', label = 'val_acc') 
 
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

# loss: 0.6927468180656433
# val_loss: 0.6928723454475403
# acc: 0.7142857313156128
# val_acc: 0.6666666865348816