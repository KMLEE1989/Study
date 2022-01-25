import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default

datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

# scaler = MinMaxScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

#model= SVC()
model=make_pipeline(MinMaxScaler(), RandomForestClassifier())

model.fit(x_train, y_train)

result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score, r2_score
y_predict=model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2 : ", result)

# r2 :  0.9559391754085522