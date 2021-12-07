from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model=Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(10,10,1)))     #9,9,10
model.add(Conv2D(5,(3,3), activation='relu'))                     # 7,7,5
model.add(Dropout(0.2))
model.add(Conv2D(7,(2,2), activation='relu'))                     #6,6,7
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))
model.summary()

#반장, 이한, 예람, 명재, 모나리자 -> 
#LabelEncoder
#0,1,2,3,4   -> (5,)  -> (5,1)
            # [0,1,2,3,4]  [0],[1],[2],[3],[4]


# model=Sequential()
# model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10,10,1)))     # 9,9  ,10   명칭들  trainable params 652가 왜 나오는지     keras.io   전부 명칭들
# model.add(Conv2D(5,(3,3), activation='relu'))                       #7,7,5
# model.add(Conv2D(7,(2,2), activation='relu'))                       #6,6,7


# model.summary()