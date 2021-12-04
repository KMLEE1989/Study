import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
#scaler=StandardScaler()
#scaler=RobustScaler()
scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_file=scaler.transform(test_file)

input1=Input(shape=(12,))
dense1=Dense(100)(input1)
dense2=Dense(100)(dense1)
dense3=Dense(110)(dense2)
dense4=Dense(110, activation='relu')(dense3)
dense5=Dense(120)(dense4)
dense6=Dense(120)(dense5)
dense7=Dense(100,activation='relu')(dense6)
dense8=Dense(100)(dense7)
dense9=Dense(100)(dense8)
output1=Dense(5, activation='softmax')(dense9)
model=Model(inputs=input1, outputs=output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:5])
print(y_test[:5])
print(resulte)

results = model.predict(test_file)
results_int = np.argmax(results, axis=1).reshape(-1,1) + 4

submit_file['quality'] = results_int

print(submit_file[:10])


submit_file.to_csv(path + "final1.csv", index=False)

# print(x.columns)
# print(x.shape)    
# print(y)
# print(y.shape)


'''


le = LabelEncoder()
le.fit(x[train.type])
x_type = le.transform(train['type'])
#x=x.drop(['type'], axis=1)
#x=pd.concat([x,x_type])
x['type'] = x_type
print(x.type.value_counts())
y=to_categorical(y)

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#scaler = MinMaxScaler()
scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_file=scaler.transform(test_file)

print(np.unique(y)) #[4 5 6 7 8]
y=to_categorical(y)
print(y)
print(y.shape)

ohe = OneHotEncoder(sparse=False)
y=ohe.fit_transform(y.reshape(-1,1))
print(x.shape, y.shape) #(3231, 13) (3231,2)

#print(x.shape, y.shape) #(3231, 13) (3231,)

input1=Input(shape=(13,))
dense1=Dense(100)(input1)
dense2=Dense(100)(dense1)
dense3=Dense(110)(dense2)
dense4=Dense(110, activation='relu')(dense3)
dense5=Dense(120)(dense4)
dense6=Dense(120)(dense5)
dense7=Dense(100,activation='relu')(dense6)
dense8=Dense(100)(dense7)
dense9=Dense(100)(dense8)
output1=Dense(5, activation='softmax')(dense9)
model=Model(inputs=input1, outputs=output1)


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, batch_size=32, callbacks=[es])

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_pred = model.predict(x_test)

###################################################################

results = model.predict(test_file)
submit_file['quality'] = results


submit_file.to_csv(path + "wine1.csv", index=False) 
'''
