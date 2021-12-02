import numpy as np

#1. 데이터
x=np.array([range(100), range(301,401), range(1,101)])
y=np.array([range(701,801)])
print(x.shape, y.shape)  #(3, 100) (2, 100)
x = np.transpose(x)
y= np.transpose(y)
print(x.shape, y.shape) #(100, 3) (100, 1)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

model = Sequential()
#model.add(Dense(10, input_dim=3))  #(100, 3) -> (N, 3)
model.add(Dense(10, input_shape=(3,)))  #4차원 가장 앞에 있는 걸 빼주면 안대 (1,10,10,3)-> (10,10,3)
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(1))
model.summary()

#model:"Sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 9)                 99
# _________________________________________________________________
# dense_2 (Dense)              (None, 8)                 80
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 9
# =================================================================
# Total params: 228
# Trainable params: 228
# Non-trainable params: 0

