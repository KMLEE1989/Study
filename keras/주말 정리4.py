# from tensorflow.keras.models import Sequential          
# from tensorflow.keras.layers import Dense
# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.backend import binary_crossentropy, sigmoid 
 
# # 1. 데이터 정제
# # 싸이킷런은 데이터를 x,y 붙여서 제공한다?
# datasets = load_breast_cancer()
# #print(datasets)

# x = datasets.data
# y = datasets.target
# #y = to_categorical(y)
# #print(datasets.DESCR)   
# #print(datasets.feature_names)   # 이름 확인
# #print(x.shape, y.shape)         # 모양분석 (569, 30) (569, )
# # print(x)
# # print(y)
# #****print(y)    # *****************결과값이 0,1 밖에 없다. ******************2진분류 + loss 값 
# # binary cross Entropy랑 sigmoid() 함수인거 까지 자동으로 생각********************
# print(np.unique(y))     # [0,1] 중복값 빼고 결과값이 뭐인지 보여주는거. 분류값에서 unique한 것이 몇개있는지 뭐 있는지 보여줌.

# '''
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
#  1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
#  1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
#  1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
#  1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
#  1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
#  1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
#  1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
#  0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
#  1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
#  0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
#  1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
#  1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 0 0 0 0 0 0 1]
# '''

# x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66) 

# #2. 모델링 
# model = Sequential()
# model.add(Dense(30, activation='linear', input_dim=30))    #activation='linear' -> 원래값을 그대로 넘겨준다. 생략해도 되는데 그냥 공부하라고 표기함.
# model.add(Dense(25, activation='linear'))   #값이 너무 큰거같으면 중간에 sigmoid한번써서 줄여줄 수도 있다.
# model.add(Dense(20, activation='linear'))
# model.add(Dense(15, activation='linear'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(5, activation='linear'))
# model.add(Dense(1, activation='sigmoid'))   # sigmoid함수는 레이어를 거치며 커져버린 y=wx+b의 값들을 0.0~1.0사이로 잡아주는 역할을 한다. 

# #이진분류는 다중분류에 속한다. softmax를 사용해도 모방 


# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    # 다중분류 softmax할거면 loss도 categorical_crossentropy로 바꿔줘야한다.
# #                      [0,1] 이니 binary_crossentropy  이진분류 sigmoid 

# #회귀모델: mse loss가 필요하다. 
# #이진분류-> binary_crossentropy  다중분류-> categorical_crossentropy 

# # 회귀모델을 하기위해서 mse loss가 필요하고 이진분류하기위해서 binary_crossentropy가 필요하고 다중분류를 하기위해서 categorical_crossentropy가 필요하다 그런개념.
# # 각모델에 맞는 mse값들이 있다.   
# # matrics=['accuracy']는 그냥 r2 스코어처럼 지표를 보여주는거지 fit에 영향을 끼치지않는다. 다른것도넣을수 있다 matrics에.
# # loss가 제일 중요하다  accuracy는 그냥평가지표이다. 몇개중에 몇개맞췄는지 보여주는 지표. 설령 잘 맞췄다 해도 loss값이 크면 운으로 때려맞춘거지. 좋다고 보장할순없다.
# # loss와 val_loss를 따지면 val_loss가 더 믿을만하다.

# es = EarlyStopping
# es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)

# hist = model.fit(x_train,y_train,epochs=10000, batch_size=1, verbose=1,validation_split=0.2, callbacks=[es])

# #4. 평가, 예측

# loss = model.evaluate(x_test,y_test)    # evaluate의 평가값은 1개로 귀결된다. loss를 출력해보면 값이 1개만 나온다.

# # 다 쳐봐서 데이터를 확인

'''
from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical   # 원핫 인코딩 도와주는 함수 기능 
from sklearn.datasets import load_iris  # 데이터셋이 아주 편하게 되어있다. 꽃잎의 모양과 줄기 넓이 등등 해서 어떤꽃인지 판별

# 나중엔 컬럼 분석도 해야함
# 1. 데이터 정제 <-- 여기서 one hot encoding까지 해줘야함.
datasets = load_iris()
#print(datasets.DESCR)   # Instances 행이 150개이다 Attributes 속성, 컬럼이 4개이다. class는 3개 꽃의 종류.
print(datasets.feature_names)   # 컬럼 이름 확인



x = datasets.data   
y = datasets.target

# print(x.shape, y.shape)     # (150,4) (150,)
# print(y)
# print(np.unique(y))         # [0 1 2] 3개인거확인

# one hot encoding 3가지 방법이 있다.
y = to_categorical(y)
print(y.shape)  # 150에 3으로 바뀌어져 있다.
print(y)        # [1,0,0],[0,1,0],[0,0,1]로 바뀌어져있다.


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) # (120, 4) (120, 3)
print(x_test.shape, y_test.shape)   # (30, 4) (30, 3)


#2. 모델링 모델구성
model = Sequential()
model.add(Dense(120, activation='linear', input_dim=4))    
model.add(Dense(100, activation='linear'))   
model.add(Dense(80, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))   

#회귀모델 activation = linear (default값) 이진분류 sigmoid 다중분류 softmax 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)

hist = model.fit(x_train,y_train,epochs=10000, batch_size=1, verbose=1,validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)   
print('loss : ' , loss[0])
print('accuracy : ', loss[1])

results = model.predict(x_test[:7])
print(x_test[:7])
print(y_test[:7])
print(results)

'''