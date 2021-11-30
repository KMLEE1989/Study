#Softmax(실습)-> Onehot encoder 
#One Hot encoding 사용 이유 ----> categorical data를 다룰때 사용 한다. 쉽게 이해한다면 범주형 텍스트 데이터를 숫자 데이터로 
#컴퓨터가 이해할 수 있는 데이터로 변경하는 과정을 말한다. 

#인덱스       값
#데이터1      컴퓨터
#데이터2      마우스  
#데이터3      키보드
#데이터4      모니터
#데이터5      책상
#데이터6      컴퓨터
#데이터7      책상

#이렇게 보면 우리는 다섯 개의 범주를 가지는 범주형 데이터라고 볼 수 있다. 다시 반복해서 보면 텍스트 형태의
#범주형 데이터를 머신러닝에 이용 하기 위해 숫자형 데이터로 바꿔 주어야 하는데 변형 시킨다면 이렇게 볼 수 있다.

# 원래값  대응값
# 컴퓨터   1
# 마우스   2
# 키보드   3
# 모니터   4
# 책상     5

#위 표를 이용하여 텍스트로 된 범주형 데이터를 숫자에 대응시키는 방식으로 수치형 데이터로 변경한 결과는 이와 같다.
#인덱스(원래값)   변경 후
#데이터1(컴퓨터)   1
#데이터2(마우스)   2
#데이터3(키보드)   3
#데이터4(모니터)   4
#데이터5(책상)     5
#이제 숫자로 된 데이터를 얻었으니 위 데이터를 이용하여 프로젝트를 진행할 수 있다. 그러나 위의 방식으로 프로젝트를 진행하면
#한 가지 중요한 오류를 범하게 되는데, 바로 데이터의 연속성에 대한 문제이다. ex)가령 컴퓨터1과 5의 중간이 키보드 3라고 할 수
#있을까? 아니면 컴퓨터1과 키보드 3을 더하면 모니터4가 되는결과가 우리가 원하는 결과일까?
#즉 데이터의 가중치는 동등한데..컴퓨터는 연속성을 이해 할 수 없을지도 모른다. 이걸 연속성이 없다는 것을
#확실히 하기 위해서 원 핫 인코딩을 한다. 

#인덱스(원래값)  컴퓨터  마우스  키보드  모니터  책상
#데이터1(컴퓨터)   1      0       0      0      0
#데이터2(마우스)   0      1       0      0      0 
#데이터3(키보드)   0      0       1     0       0
#데이터4(모니터)   0      0      0       1      0
#데이터5(책상)    0       0       0     0       1
#데이터6(컴퓨터)   1      0      0      0       0
#데이터7(책상)    0      0      0      0        1

from tensorflow.python.keras.backend import softmax
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder

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

print(x.shape, y.shape) #->구해 보았더니 (581012, 54) (581012,) 이렇게 나오네요
print(y)#->feature를 구해 보도록 하구요
print(np.unique(y)) #->[1 2 3 4 5 6 7] 이렇게 나옵니다.  
#print(np.unique) #-> NumPy 배열에서 모든 고유 값을 검색하고 이러한 고유 값을 정렬합니다.

#중간정리 (x,y)= (581012, 54) (581012,) 
#np.unique는 [1 2 3 4 5 6 7] 이렇게 나온답니다. 

#그다음
# from tensorflow.keras.utils import to_categorical 여기서 콜백한 to_categorical을 사용하여
#  y=[1 2 3 4 5 6 7]를 재정렬 해보도록 합니다. 
#y=to_categorical(y)
#print(y)
#print(y.shape) #(581012, 8)

ohe = OneHotEncoder(sparse=False)
y=ohe.fit_transform(y.reshape(-1,1))
print(x.shape, y.shape)  #-> (581012, 54) (581012, 7)


#다음은 train test split 과정으로 넘어갑니다. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

#그리고 우리는 train과 test의 값중 test 값을 사용할 거랍니다. 
print(x_train.shape, y_train.shape) #->(464809, 54) (464809, 7)
print(x_test.shape, y_test.shape) #-> (116203, 54) (116203, 7)

#사용할 값! (116203, 54) (116203, 8)

model = Sequential() 
model.add(Dense(10, activation='linear', input_dim=54))  #(116203, 54) 
model.add(Dense(10))                                               #54는 input의 노드가 되구요
model.add(Dense(70, activation='linear'))
model.add(Dense(60))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='softmax'))  #(116203, 7) 7은 아웃풋의 노드가 된답니다. 꼭 잊지 맙시다. activation은 softmax입니다. 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#컴파일 훈련 부분에서 loss는 categorical_crossentropy를 사용하고 메트릭스에는 어큐리씨가 추가 됩니다.

#다음은 얼리 스타핑을 사용합니다. 
from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])

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



#*이진분류는 다중분류 (원핫 인코딩) 이다.