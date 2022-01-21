import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

# scaler = MinMaxScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)


from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

#model= SVC()
#model=make_pipeline(MinMaxScaler(), StandardScaler(), SVC())
#model=make_pipeline(MinMaxScaler(), PCA(), SVC())
model=Pipeline([('mm', MinMaxScaler()), ("svc",SVC())])

model.fit(x_train, y_train)

result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict=model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("model.score : ", result)

# model.score :  1.0


