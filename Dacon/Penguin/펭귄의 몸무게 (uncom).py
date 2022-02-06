import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.layers import Dense, Input, Model

path = "../_data/dacon/peg/data/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")


train = pd.read_csv(path+'train.csv')
# print(train)  #[114 rows x 11 columns]
# print(train.shape) #(114, 11)

test_file = pd.read_csv(path+'test.csv')
# print(test_file) #[228 rows x 10 columns]
# print(test_file.shape) #(228, 10)


submit_file = pd.read_csv(path+'sample_submission.csv')
# print(submit_file) #[228 rows x 2 columns]
# print(submit_file.shape) #(228, 2)
# print(submit_file.columns) #Index(['id', 'Body Mass (g)'], dtype='object')


# print(type(train))     
# print(train.info())      
# print(train.describe())

# print(train.columns)
# Index(['id', 'Species', 'Island', 'Clutch Completion', 'Culmen Length (mm)',
#        'Culmen Depth (mm)', 'Flipper Length (mm)', 'Sex', 'Delta 15 N (o/oo)',
#        'Delta 13 C (o/oo)', 'Body Mass (g)'],
#       dtype='object')
# print(train.head(3))
# print(train.tail())
#print(test_file)

x = train.drop(['id','Species', 'Island', 'Clutch Completion'], axis=1)     
test_file=test_file.drop(['id', 'Species','Island','Clutch Completion'], axis=1)
y = train['Body Mass (g)']

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
dense3=Dense(110, activation='relu')(dense2)
dense4=Dense(110, activation='relu')(dense3)
dense5=Dense(100, activation='relu')(dense4)
dense6=Dense(100, activation='relu')(dense5)
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

# print("===============================================2. load_model 출력 ===========================")
# model2=load_model('./_save/keras27_8_save_model.h5')
# loss2=model2.evaluate(x_test, y_test) 
# print('loss :', loss2)

# y_predict2= model2.predict(x_test)

# from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
# r2=r2_score(y_test, y_predict2)
# print('r2스코어:', r2)
