###

# 딥러닝=머신러닝의 한 분야--> 연속된 층(Layer) 에서 점진적 학습에 강점. 기계학습의 새로운 방식
#                                 층의 숫자는 모델의 깊이를 의미 

#  *AI>머신러닝>딥러닝 

#신경망 (Neural Network): 딥러닝은 기본 층들을 쌓아서 구성한 신경망
#BNN-> Biological Neural Network -> SOMA, Dendrities, Synapse, Axon -> Synapse, Dendrities, Soma, Synapse, Axon 

#Perceptron:  레거시한 초기 형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘.
#ANN (Aritificial Neural Network (ANN)
#뉴런을 본따서 만드는  ** 두개의 층(Layers) 이상으로 구성되어 있다는 것이 특징
# 
# 뉴런(생물학적인 뉴런의 개념에 기초한 수학적인 함수)  *뉴런이 활성 중인지에 따라 활성 함수가 결정* *IN AI*   해당 뉴런의 결과가 0이라면 비활성화된 뉴런이라고 특정 지을 수 있다.


# input                node          output
#   X   --------------> f -----------> y
#          weight      bias
#         f: activation function

# input Layer
# 입력 뉴런들로 구성된 층
# output Layer
# 결과물을 생산하는 출력 뉴런
# Hidden Layer
# 입력층과 출력층 사이에 있는 Layer

# Deep Neural Network (DNN)
# *적어도 3개의 층이상으로 이루어진 신경망을 의미합니다. (2개이상의 Hidden Layers)

# 신경망 학습의 목표는 가중치의 정확한 값을 찾는 것 --> 어떤 층이 있을 떄, 해당 층을 데이터가 거치면서 일어나는 변환은 해당 층의 가중치를 매개변수로 가지는 함수를 통해서 일어납니다. 

# 최적의 Weight 최소의 loss

# 데이터를 입력받습니다.
# 층에 도달할 때마다 해당 층의 가중치를 사용하여 출력 값을 계산합니다.
# 모든 층에서 반복합니다. 마지막으로는 예측 값이 나옵니다.
# 예측 값은 실제 값과 비교하는 손실 함수를 통해서 점수를 계산합니다.
# 해당 점수를 피드백으로 삼아서, 손실 점수가 감소되는 방향으로 가중치를 수정합니다. (Optimizer) *최소의 로스*
# 위의 과정을 반복(Loop)합니다.
# 손실 함수를 통해 계산되는 손실 점수를 최소화시키는 가중치를 찾아냅니다.

#################weight#################
# 뉴런사이의 연결 강도를 의미합니다.
# 신경망이 훈련을 하는동안, 업데이트되어 weight가 변경됩니다.
# Learnable Parameters 라고도 부릅니다. 

#################BIAS#####################

#BIAS뉴런의 가중치를 의미
#해당 값 또한, 훈련 시 업데이트 됩니다.

##############Activation Function ############
# sigmoid function
# 초창기에 많이 사용된 활성함수로, S자형 곡선을 가지는 함수입니다.

# AND, OR, XOR 연산을 다루는데 적합한 함수입니다.

# 또한 Binary Classification에 적합한 함수입니다.

# 출력 값의 범위: [0, 1]

#############################ReLU###############################
# Sigmoid의 대안으로 나온 활성함수로, 0이하의 수는 0을, 양수인 경우에는 양수를 그대로 반환하여 값의 왜곡이 적어집니다.
# 값의 범위: [0, inf]

'''
import numpy as np
from tensorflow.keras import models 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 멀티 레이어 퍼섹트론?
#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])   # 10행 2열의 데이터를 넘겨줘야함.
y = np.array([11,12,13,14,15,16,17,18,19,20])

# print(x.shape) # (2,10)
# print(y.shape)  #(10,)    y맞추기 항상 명심. 
x = x.T

print(x.shape)
print(y.shape)
'''

#x.T 이거 하나만으로 배열의 행과 열을 바꿔줌.
# x = np.transpose(x)   이거도 똑같은 기능이다 
# x = reshape(10,2)     변환은 되는데 print해서 확인해보면 짝이 다 깨져있다. 구조가 엉망이 되어있다.
# transpose 와 reshape의 차이점 transpose는 변환의 개념이고 , reshape는 데이터를 늘였다 줄였다하면서 형태를 바꿔주는 개념

# 데이터를 train과 test로 나눠주는 이유
# fit에서 모델학습을 시킬때 모든 데이터를 다 학습시켜버리면 x = [1~10] y = [1~10]
# 실제로 원하는 미래의 데이터를 넣어봤을때 크게 오류가 날 수 있다 model.predict[11]
# 왜냐하면 컴퓨터는 모든 주어진 값으로만 훈련을 하고 실전을 해본적이 없기때문이다
# 이때문에 train과 test로 나누어 x_train [1~7] x_test[8~10]
# train으로 학습을 시키고 test로 실전같은 모의고사를 한번 미리해보면
# fit단계에서의 loss값과 evaluate의 loss값의 차이가 큰 걸 확인할 수 있다.
# 확인까지만 가능하고 그 이상은 뭐 할 수 없다? evaluate은 평가만 가능한거지 
# 여기서 나온 loss값과 fit 단게의 loss값들의 차이가 크다 하더라도 그 차이가 fit단계에 
# 적용되지는 않는다.
'''
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])             
y = np.array([1,2,3,4,5,6,7,8,9,10])
#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])      # 훈련
x_test = np.array([8,9,10])             # 평가
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss) # 결과값에서 나온 loss?
result = model.predict([11])
print('[11]의 예측값 : ', result)
'''
'''
x_train = random.sample(list(x), 30) x데이터 값들중 30개를 중복없이 뽑는다. sample기능
x_test = [x for x in x if x not in x_train] 구글링해온 명령어 x에서 x_train인것들을 뺀다.
y_train = list(x_train+int(1))  x_train의 모든 값들에 1씩 더해주려던 시도.
#y_test = 내가 작업하던곳... 근데 난 랜덤난수가지 생각못해서 할때마다 값이 바뀌었을거 같다.
랜덤난수 --> 하나의 Train-test set에서 여러번 훈련 돌려가면서 weight측정할때 오차 없게하기 위해 
랜덤난수 없이 반복훈련하면 다른 Train-test set 작업하는거랑 다를게없다 쉽게 말해서
x_train = [1 3 5 7 9] x_train = [2 3 4  5 6 ] compile할때마다 train값이 바뀌어서 그전의 측정값들과
아무 연관이 없어서 실험하는 의미가 없다.
x,y를 train과 test로 원하는 비율로 나누고 값들을 랜덤하게 뽑아주는 작업까지 모두 한번에
from sklearn.model_selection import train_test_split 이 기능을 가져와서 쓸수있다.
'''

"""
from numpy.core.fromnumeric import shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
x = np.array(range(100))            
y = np.array(range(1,101))          

print(x.shape) #(100,)
print(y.shape) #(100,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, shuffle=True,
    random_state=66)

#print(x_test)   #[ 8 93  4  5 52 41  0 73 88 68]
#rint(y_test)
#랜덤 난수 넣어준다 -> 훈련을 반복해도 동일한 값이 나와야 제대로 된 훈련이 가능하기때문. 
#이게 없으면 한번 다시돌릴때마다 x_train~~y_test 값이 계속 바뀐다. 

print(x_test,shape)
print(y_test.shape)
"""
'''
#2. 모델링
model =  Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) #evaluate에서 나온값들을 loss에 담는다. loss가 저거라는뜻이 아니다.
print('loss: ', loss) # 결과값에서 나온 loss?
result = model.predict([150])
print('[100]의 예측값 : ', result)
'''
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt     # 그래프나 그림그릴때 많이 씀
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,17, 8,14,21, 9, 6,19,23,21])

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

print(x_train.shape)
print(y_train.shape)
'''
'''
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train,y_train,epochs=100, batch_size=1)

#4. 평가, 예측      이건 진짜 말그대로 그냥 해보는것. 여기서 나온 loss값과 훈련에서 나온 loss값의 차이가 적을수록 정확한것.
# 근데 생각해보니까 기본적으로 데이터를 정제 잘하고 뺄건 빼서 넘겨주는게 제일 중요할것 같다.
loss = model.evaluate(x_test,y_test) # 평가해보는 단계. 이미 다 나와있는  w,b에 test데이터를 넣어보고 평가해본다.
print('loss : ', loss)

y_predict = model.predict(x)

plt.scatter(x, y) 
plt.plot(x, y_predict, color='red') # scatter 점찍다 plot 선을 보여준다 
plt.show() 
'''
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
# import matplotlib.pyplot as plt     # 그래프나 그림그릴때 많이 씀
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score  

# x=np.array(range(100))
# y=np.array(range(1,101))

# x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.7, shuffle=True, random_state=66)

# model=Sequential()
# model.add(Dense(100, input_dim=1))

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
# from sklearn.metrics import r2_score    # r2_score 구하는 공식? 및 작업을 다 해놓은걸  import해서 가져다쓴다 

#1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,4,3,5])  

# print(x.shape)
# print(y.shape)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(5))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')  # mean squared error , 평균제곱오차    
# model.fit(x,y,epochs=1000, batch_size=1)

# #4. 평가, 예측 
# #loss = model.evaluate(x,y) # 평가해보는 단계. 이미 다 나와있는  w,b에 test데이터를 넣어보고 평가해본다.
# #print('loss : ', loss)

# y_predict = model.predict(x) #y의 예측값은 x의 테스트값에 wx + b 

# r2 = r2_score(y,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
# print('r2스코어 : ', r2)

