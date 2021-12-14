from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets, metrics
import numpy as np
from tensorflow.python.keras.metrics import accuracy

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

path = "../_data/dacon/heart_disease/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

x=train.drop(['id', 'target','thal'], axis=1)
test_file=test_file.drop(['id','thal'], axis=1)
y=train['target']

# print(train.shape) #(151, 15)
#print(train.columns)
# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
#       dtype='object')

# print(test_file.shape) #(152, 14)
# print(test_file.columns)
# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
#       dtype='object')
# print(submit_file.shape) #(152, 2)
# print(submit_file.columns)
# Index(['id', 'target'], dtype='object')

# y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size =0.8, shuffle=True, random_state = 66)

# print(x_train.shape)  #(120, 13)
# print(x_test.shape)   #(31, 13)

scaler = MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)


x_train = x_train.reshape(120,3,4,1) 
x_test = x_test.reshape(31,3,4,1)
test_file=test_file.reshape(152,3,4,1)
# print(y.shape)   (151,1)

model = Sequential() 
model.add(Conv2D(32, kernel_size = (2,2),input_shape = (3,4,1)))                      
model.add(Dropout(0.2))       
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',
                   verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_7_MCP.hdf5')
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.3, callbacks=[es,mcp])

# model.save('./_save/keras27_7_save_model.h5')

# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)

loss = model.evaluate(x_test, y_test)
y_predict=model.predict(x_test)
y_predict=y_predict.round(0).astype(int)
f1=f1_score(y_test, y_predict)
print('loss : ', loss[0])
print('accuracy :  ', loss[1])
# print('f1_score :  ', f1)

results=model.predict(test_file)
results=results.round(0).astype(int)

print('F1_Score :', f1_score(y,results[1:]))

submit_file['target']=results
submit_file.to_csv(path + "Neptune.csv", index=False)  




# # Import library
# import pandas as pd
# from tqdm import tqdm
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# import csv

# # Model list
# def models(model):
#     if model == 'knn':
#         mod = KNeighborsClassifier(2)
#     elif model == 'svm':
#         mod = SVC(kernel="linear", C=0.025)
#     elif model == 'svm2':
#         mod = SVC(gamma=2, C=1)
#     elif model == 'gaussian':
#         mod = GaussianProcessClassifier(1.0 * RBF(1.0))
#     elif model == 'tree':
#         mod =  DecisionTreeClassifier(max_depth=5)
#     elif model == 'forest':
#         mod =  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#     elif model == 'mlp':
#         mod = MLPClassifier(alpha=1, max_iter=1000)
#     elif model == 'adaboost':
#         mod = AdaBoostClassifier()
#     elif model == 'gaussianNB':
#         mod = GaussianNB()
#     elif model == 'qda':
#         mod = QuadraticDiscriminantAnalysis()
#     return mod

# ## Data load
# datapath = 'C:/Users/ImedisynRnD2/Desktop/KTH/기타/DaconHRV/dataset/'
# train_data = pd.read_csv(datapath + 'train.csv').to_numpy()
# test_data = pd.read_csv(datapath + 'test.csv').to_numpy()

# #make model list in models function
# model_list = ['knn', 'svm', 'svm2', 'gaussian', 'tree', 'forest', 'mlp', 'adaboost', 'gaussianNB', 'qda']

# cnt = 0
# empty_list = [] #empty list for progress bar in tqdm library
# for model in tqdm(model_list, desc = 'Models are training and predicting ... '):
#     empty_list.append(model) # fill empty_list to fill progress bar
#     #classifier
#     clf = models(model)

#     #Training
#     clf.fit(train_data[:,1:-1], train_data[:,-1:].T[0]) #학습할때는 id와 target을 제외하고 학습! 마지막 column이 라벨이므로 라벨로 설정!

#     #Predict
#     pred = clf.predict(test_data[:,1:]) #마찬가지로 예측을 할 때에도 id를 제외하고 나머지 feature들로 예측

#     #Make answer sheet
#     savepath = datapath + 'answers/' #정답지 저장 경로
#     with open(savepath + '%s_answer.csv' % model_list[cnt], 'w', newline='') as f:
#         sheet = csv.writer(f)
#         sheet.writerow(['id', 'target'])
#         for idx, p in enumerate(pred):
#             sheet.writerow([idx+1, p])

#     cnt += 1



# loss = model.evaluate(x_test, y_test)
# y_predict=model.predict(x_test)
# y_predict=y_predict.round(0).astype(int)
# f1=f1_score(y_test, y_predict)
# print('loss : ',loss[0])
# print('accuracy :  ', loss[1])
# print('f1_score :  ', f1)

# results=model.predict(test_file)
# results=results.round(0).astype(int)

# submit_file['target']=results
# submit_file.to_csv(path + "Venus.csv", index=False)  


#print('F1_Score :', f1_score(y,results[1:]))


'''
loss = model.evaluate(x_test, y_test)
y_predict=model.predict(x_test)
y_predict=y_predict.round(0).astype(int)
f1=f1_score(y_test, y_predict)
print('loss : ',loss[0])
print('accuracy :  ', loss[1])
print('f1_score :  ', f1)

results=model.predict(test_file)
results=results.round(0).astype(int)

submit_file['target']=results
submit_file.to_csv(path + "MARS.csv", index=False)  
'''



"""
#y=to_categorical(y)
# print(y)
# print(train.shape) 
# print(y.shape) 
 
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=49)

print(x_train.shape,y_train.shape)      #(120, 4) (120,)
print(x_test.shape,y_test.shape)        #(30, 4) (30,)


scaler =MinMaxScaler() 
x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,2,1)
x_test = scaler.transform(x_test).reshape(len(x_test),2,2,1)

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(2,2,1), activation='relu'))    # 2,2,10                                                                           # 1,1,10
model.add(Conv2D(10,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       # 2,2,10 
model.add(MaxPooling2D(2,2))                                                                                # 1,1,10     
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_1_boston{krtime}.hdf5')
model.fit(x_train,y_train,epochs=1000, batch_size=10,validation_split=0.2, callbacks=[es])#,mcp


#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)
"""

'''
import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten ,MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
#1. 데이터 
path = '../_data/kaggle/bike/'   
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
# print(submit_file.columns)
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']
# 로그변환
y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.7, shuffle=True, random_state = 42)

print(x_train.shape)  # (7620, 8)
print(x_test.shape)  # (3266, 8)
scaler = MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

x_train = x_train.reshape(7620,2,2,2) 
x_test = x_test.reshape(3266, 2,2,2)
# print(y.shape)   # (10886,)


#2. 모델구성
model = Sequential() 
model.add(Conv2D(7, kernel_size = (2,2),input_shape = (2,2,2)))                      
model.add(Dropout(0.2))       
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',
                   verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_7_MCP.hdf5')
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras27_7_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

'''