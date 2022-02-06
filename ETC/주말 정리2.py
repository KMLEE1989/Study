# MATRIX 카운트 연습


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score


# datasets = load_boston()
# x = datasets.data
# y = datasets.target

# print(x)
# print(y)
# print(x.shape) #(506, 13)
# print(y.shape)  #(506,)
# print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(datasets.DESCR)

"""
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
"""

#x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


'''
#2. 모델링 
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train,y_train,epochs=200, batch_size=1)

#4. 평가 , 예측  평가는 말그대로 평가만 해보는것
# fit에서 구해진 y = wx + b에 x_test와 y_test를 넣어보고 그 차이가 loss로 나온다?
#loss = model.evaluate(x_test,y_test)
#print('loss : ', loss)

y_predict = model.predict(x_test) #y의 예측값은 x의 테스트값에 wx + b 

r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)
'''

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
# from sklearn.metrics import r2_score    
# import time

# #1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,4,3,5])  

# #2. 모델구성
# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(5))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')      

# start = time.time()
# model.fit(x,y,epochs=1000, batch_size=1, verbose=2)
# end = time.time() - start
# print("걸린시간 : ", end) 
# # verbose = 훈련 보여줄지 안보여 줄지 체크하는 역할

# '''
# 0 결과값만  2.3366405963897705
# 1 다보여줌  3.520465850830078
# 2 loss까지  2.8723676204681396
# 3~ epoch만 보임 훈련횟수 ㅇㅇ
# 사람이 진행정도를 확인할 수 있게 정보표기를 컨트롤 할수 있게해준다. 편의 기능 ㅇㅇ 
# 또한 불필요한 정보를 안보이게 해줌으로써 시간 단축의 역할도 한다.
# '''

# start=time.time()         # fit을 사이에 놓고 입력! 
# model.fit
# end=time.time() -start

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score


# datasets = load_diabetes()
# x = datasets.data
# y = datasets.target

# # print(x)
# # print(y)

# print(x.shape) #(442, 10)
# print(y.shape)  #(442,)

# x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

# 행렬 카운트 연습

