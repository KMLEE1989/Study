from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape,Conv1D,LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import softmax

model=Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides=1, padding='same', input_shape=(28,28,1)))     #padding= Valid  #padding=same 차이
model.add(MaxPooling2D())
model.add(Conv2D(5,(2,2), activation='relu'))         #13,13,5            
model.add(Conv2D(7,(2,2), activation='relu'))   #12,12,7
model.add(Conv2D(7,(2,2), activation='relu'))   #11,11,7
model.add(Conv2D(10,(2,2), activation='relu'))   #10,10,10                     
model.add(Flatten())                            #(N, 1000)
model.add(Reshape(target_shape=(100,10)))   #(N,100, 10)
model.add(Conv1D(5 ,2))    #(N,99,5)
model.add(LSTM(15))
model.add(Dense(10, activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))
model.summary()

#model.summary()

#반장, 이한, 예람, 명재, 모나리자 -> 
#LabelEncoder
#0,1,2,3,4   -> (5,)  -> (5,1)
            # [0,1,2,3,4]  [0],[1],[2],[3],[4]


# model=Sequential()
# model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10,10,1)))     # 9,9  ,10   명칭들  trainable params 652가 왜 나오는지     keras.io   전부 명칭들
# model.add(Conv2D(5,(3,3), activation='relu'))                       #7,7,5
# model.add(Conv2D(7,(2,2), activation='relu'))                       #6,6,7


# model.summary()
