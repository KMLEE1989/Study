from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test)= reuters.load_data(
    num_words=10000, test_split=0.2    
)

print(x_train, len(x_train), len(x_test)) #8982, 2246
print(y_train[0]) #3
print(np.unique(y_train)) #46개의 뉴스 카테고리
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train), type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape) #(8982,) (8982,)

print(len(x_train[0]), len(x_train[1])) #87, 56
print(type(x_train[0]), type(x_train[1])) #<class 'list'> <class 'list'>

#print("뉴스기사의 최대길이 : ", max(len(x_train)))  #error
print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))  #2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/len(x_train)) #145.53

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train=pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
#print(x_train.shape) #(8982, 2376) -> (8982, 100)

x_test=pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) #(8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)  #(8982, 100) (2246, 46)

####################################################질문금지, 예람센세 구차나 #####################################################################

word_to_index = reuters.get_word_index()
#print(word_to_index)
#print(sorted(word_to_index.items()))
import operator
print(sorted(word_to_index.items(),key=operator.itemgetter(1)))

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key
    
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
        index_to_word[index]=token
        
print(' '.join([index_to_word[index] for index in x_train[0]]))



'''

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=100))  #텐서플로우의 inputdim 여기서는 아웃풋이 아니다 유일하게  13,5,28->28*10
#model.add(Embedding(28,10, input_length=5))
#model.add(Embedding(28,10))   #(N,N,10)
#model.add(LSTM(32, activation='relu', input_shape=(100,1)))  #<- 통상 아웃풋 근데 임베딩에서는 달라
model.add(LSTM(32))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

#완성 실습

'''
