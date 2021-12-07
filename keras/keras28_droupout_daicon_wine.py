import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

path = "../_data/dacon/wine/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

train = pd.read_csv(path+'train.csv')
print(train)  #[3231 rows x 14 columns]
print(train.shape) #(3231, 14)

test_file = pd.read_csv(path+'test.csv')
print(test_file) #[3231 rows x 13 columns]
print(test_file.shape) #(3231, 13)

submit_file = pd.read_csv(path+'sample_submission.csv')
print(submit_file) # [3231 rows x 2 columns]
print(submit_file.shape) #(3231, 2)
print(submit_file.columns) #Index(['id', 'quality'], dtype='object')

print(type(train))     
print(train.info())      
print(train.describe())

print(train.columns)
# Index(['id', 'fixed acidity', 'volatile acidity', 'citric acid',
#        'residual sugar', 'chlorides', 'free sulfur dioxide',
#        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type',
#        'quality'],
#       dtype='object')
print(train.head(3))
print(train.tail())

x = train.drop(['id','quality'], axis=1)  #컬럼 삭제할때는 드랍에 액시스 1 준다   
test_file=test_file.drop(['id'], axis=1)
y = train['quality']

le=LabelEncoder()
le.fit(x['type'])
x['type']=le.transform(x['type'])

print("Label mapping:")
for i, item in enumerate(le.classes_):
    print(item, i)
    
le.fit(test_file['type'])
test_file['type']=le.transform(test_file['type'])

print(test_file['type'])
print(np.unique(y))

y=pd.get_dummies(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#scaler = MinMaxScaler()
scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_file=scaler.transform(test_file)

input1=Input(shape=(12,))
dense1=Dense(100, activation='relu')(input1)
dense2=Dense(100, activation='relu')(dense1)
drop2=(Dropout(0.2))(dense2)
dense3=Dense(110, activation='relu')(drop2)
dense4=Dense(110, activation='relu')(dense3)
dense5=Dense(100, activation='relu')(dense4)
drop5=(Dropout(0.5))(dense5)
dense6=Dense(100, activation='relu')(drop5)
dense7=Dense(100, activation='relu')(dense6)
output1=Dense(5, activation='softmax')(dense7)
model=Model(inputs=input1, outputs=output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  

############################################################################################################
import datetime
date=datetime.datetime.now()   
datetime = date.strftime("%m%d_%H%M")   #1206_0456
#print(datetime)

filepath='./_ModelCheckPoint/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'  # 2500-0.3724.hdf
model_path = "".join([filepath, 'k26_', datetime, '_', filename])
            #./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf
##############################################################################################################

es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.5, callbacks=[es, mcp])

model.fit(x_train, y_train, epochs=500, batch_size=16, validation_split=0.2, callbacks=[es, mcp])

loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy :  ', loss[1])

resulte = model.predict(x_test[:5])
print(y_test[:5])
print(resulte)

results = model.predict(test_file)
results_int = np.argmax(results, axis=1).reshape(-1,1) + 4

submit_file['quality'] = results_int

print(submit_file[:10])


submit_file.to_csv(path + "ten.csv", index=False)
model.save('./_save/keras27_8_save_model.h5')
           
#result
print("============================1. 기본출력 ========================")
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

print("===============================================2. load_model 출력 ===========================")
model2=load_model('./_save/keras27_8_save_model.h5')
loss2=model2.evaluate(x_test, y_test) 
print('loss :', loss2)

y_predict2= model2.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict2)
print('r2스코어:', r2)

#미적용
# loss : [0.9556612372398376, 0.5795981287956238]
# r2스코어: 0.14894408474201318

# dropout적용
# loss : [0.9668137431144714, 0.5765069723129272]
# r2스코어: 0.13358163646298923

# print("==================================================3. ModelCheckPoint load 출력=======================")

# model3=load_model('./study/_ModelCheckPoint/keras26_3_MCP.hdf5')
# loss3=model3.evaluate(x_test, y_test) 
# print('loss :', loss3)

# y_predict3= model3.predict(x_test)

# from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
# r2=r2_score(y_test, y_predict3)
# print('r2스코어:', r2)