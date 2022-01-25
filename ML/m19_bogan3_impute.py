import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2,4,np.nan,8,np.nan],
                     [np.nan,4,np.nan,8,10],
                     [np.nan,4,np.nan,8,np.nan]])

print(data.shape) #(4, 5)

data=data.transpose()
data.columns=['a','b','c','d']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

#imputer = SimpleImputer(strategy='mean')
imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent')
# imputer = SimpleImputer(strategy='constant')
# imputer=SimpleImputer(strategy='constant', fill_value=777)

imputer.fit(data[['a']])

data2=data.copy()
#data2 = imputer.transform(data[['a']])
#print(data2)

#fit에는 dataframe이 들어가는데, 우리는 컬럼만 바꾸고 싶다.
#시리즈를 넣으면 에러가 난다
#처리해 보아라!!!

# 1개만 할 경우
#data2[['a']] = imputer.fit_transform(data[['a']])

# 2개이상 할 경우
#data2[['a','c']] = imputer.fit_transform(data[['a','c']])
data2[['b','d']] = imputer.fit_transform(data[['b','d']])
print(data2)

