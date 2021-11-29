from tensorflow.python.keras.backend import softmax

#softmax의 fetch covtype 케이스 과제를 하며 순서를 학습하고 구동원리를 이해하여 봅시다!
#먼저 사이킥 런과 텐서플로를 소환합니다.
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#데이터셋을 불러오고 그다음 데이터의 분석을 위해 데이터를 출력해봅니다.
datasets=fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

# **Data Set Characteristics:** 대략적으로 이런 데이타 테이블의 특성이 나옵니다. 
#     =================   ============
#     Classes                        7
#     Samples total             581012
#     Dimensionality                54
#     Features                     int
#     =================   ============
# 일단 이해 하고 다음은 변수 설정을 해보도록 합니다. 

x=datasets.data
y=datasets.target

#변수설정이 되었으니 우리는 x와 y의 shape를 구해보도록 합ㅅ다. 
print(x.shape, y.shape) #->구해 보았더니 (581012, 54) (581012,) 이렇게 나오네요
print(y)#->feature를 구해 보도록 하구요
print(np.unique(y)) #->[1 2 3 4 5 6 7] 이렇게 나옵니다.  
#print(np.unique) #-> NumPy 배열에서 모든 고유 값을 검색하고 이러한 고유 값을 정렬합니다.

#중간정리 (x,y)= (581012, 54) (581012,) 
#np.unique는 [1 2 3 4 5 6 7] 이렇게 나온답니다. 

#그다음
# from tensorflow.keras.utils import to_categorical 여기서 콜백한 to_categorical을 사용하여
#  y=[1 2 3 4 5 6 7]를 재정렬 해보도록 합니다. 
y=to_categorical(y)
print(y)
print(y.shape) #(581012, 8)

#다음은 train test split 과정으로 넘어갑니다. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

#그리고 우리는 train과 test의 값중 test 값을 사용할 거랍니다. 
print(x_train.shape, y_train.shape) #->(464809, 54) (464809, 8)
print(x_test.shape, y_test.shape) #-> (116203, 54) (116203, 8)

#사용할 값! (116203, 54) (116203, 8)

model = Sequential() 
model.add(Dense(10, activation='linear', input_dim=54))  #(116203, 54) 
model.add(Dense(10))                                               #54는 input의 노드가 되구요
model.add(Dense(70, activation='linear'))
model.add(Dense(60))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(8, activation='softmax'))  #(116203, 8) 8은 아웃풋의 노드가 된답니다. 꼭 잊지 맙시다. activation은 softmax입니다. 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#컴파일 훈련 부분에서 loss는 categorical_crossentropy를 사용하고 메트릭스에는 어큐리씨가 추가 됩니다.

#다음은 얼리 스타핑을 사용합니다. 
from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:10])
print(y_test[:10])
print(results)

#두번째 부분 과제 batchsize의 디폴트 값을 알아보고 뱃치사이즈를 지운후 코딩을 구동하여 봅시다.
#구하는 방법은 디폴트때 epoch가 돌아가는 속도를 구하고 뱃치사이즈가 1일때의 속도를 점착한후 나누면
# 디폴트 값이 나옵니다. 아래에 구하였으니 참조하여 봅시다. 

# #result
# loss : 0.644293487071991
# accuracy: 0.7220553755760193
# batch_size 의 디폴트 값은 371847/ 11621 = 31.99==32 