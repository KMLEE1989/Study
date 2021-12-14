import pandas as pd
import numpy as np
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
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
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  
from tensorflow.keras. optimizers import SGD
from keras import optimizers
from sklearn import metrics
from tensorflow.python.keras.layers.core import Activation


path = "../_data/dacon/heart_disease/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

# train = pd.read_csv(path+'train.csv')
# print(train)  #[151 rows x 15 columns]
# print(train.shape)  #(151, 15)

# test_file = pd.read_csv(path+'test.csv')
# print(test_file) #[152 rows x 14 columns]
# print(test_file.shape) #(152, 14)

 
# submit_file = pd.read_csv(path+'sample_submission.csv')
# print(submit_file)  #[152 rows x 2 columns]
# print(submit_file.shape) #(152, 2)
# print(submit_file.columns)  #Index(['id', 'target'], dtype='object')


print(type(train))     
print(train.info())      
print(train.describe())

print(train.columns)

# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
#       dtype='object')


x = train.drop(['id','target'], axis=1)  #컬럼 삭제할때는 드랍에 액시스 1 준다   
test_file=test_file.drop(['id'], axis=1)
y = train['target']

print(train.describe)

# le=LabelEncoder()
# le.fit(x['sex'])
# x['sex']=le.transform(x['sex'])

# print("Label mapping:")
# for i, item in enumerate(le.classes_):
#     print(item, i)
    
# le.fit(test_file['sex'])
# test_file['sex']=le.transform(test_file['sex'])

# print(test_file['sex'])
#print(np.unique(y))

# y=pd.get_dummies(y)
#print("y.shape")
#print(y.shape)

x = x.to_numpy()
y = y.to_numpy()
test_file = test_file.to_numpy()
print(x.shape, test_file.shape)  


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_file=scaler.transform(test_file)


# input1=Input(shape=(13,))
# dense1=Dense(100, activation='relu')(input1)
# drop1=Dropout(0.5)(dense1)
# dense2=Dense(50, activation='relu')(drop1)
# drop2=Dropout(0.5)(dense2)
# dense3=Dense(100, activation='relu')(drop2)
# dens4=Dense(100)(dense3)
# output1=Dense(1, activation='sigmoid')(dens4)
# model=Model(inputs=input1, outputs=output1)
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

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

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.3, callbacks=[es, mcp])

model.save('../Study/_save/Dacon_heart_disease.h5')

loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy :  ', loss[1])

# print('score :  '.score())

# results = model.predict(x_test[:5])
# print(y_test[:5])
# print(results)

#results_test = model.predict(x_test)
#print(results_test)
results = model.predict(test_file)
results = results.round(0).astype(int)

print(results)
# results_int = np.argmax(results, axis=1).reshape(-1,1) + 4

# pred=model.predict(test_file)
#num2 = f1_score(y_test, pred)


def f1_score(answer, submit_file):
    true = answer
    pred = submit_file
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

print('F1_Score :', f1_score(y,results[1:]))

# submit_file['target'] = results

#print(submit_file[:10])

# submit_file.to_csv(path + "dacon heart.csv", index=False)  


#accuracy :   0.9032257795333862
