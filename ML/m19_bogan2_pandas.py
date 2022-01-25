import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],[2,4,np.nan,8,np.nan],[np.nan,4,np.nan,8,10],[np.nan,4,np.nan,8,np.nan]])

print(data.shape) #(4, 5)

data=data.transpose()
data.columns=['a','b','c','d']
print(data)

#결측치 확인
# print(data.isnull()) 
# print(data.isnull().sum())
# print(data.info())

#1. 결측치 삭제
# print(data.dropna())
#print(data.dropna(axis=0))
#print(data.dropna(axis=1))

#2-1. 특정값 -평균
# means = data.mean()
# print(means)
# data = data.fillna(means)
# print(data)


#2-2. 특정값 - 중위값(중간값)
# meds=data.median()
# print(meds)
# data2 = data.fillna(meds)
# print(data2)

#2-3. 특정값 - ffill
data2=data.fillna(method='ffill')
print(data2)

data2=data.fillna(method='ffill', limit=1)
print(data2)

data2=data.fillna(method='bfill')
print(data2)

data2=data.fillna(method='bfill',limit=1)
print(data2)

#2-3.특정값 - 채우기
data2=data.fillna(747474)
print(data2)

###########################################특정컬럼만!!##################################################

means = data['a'].mean()
print(means)
data['a'] = data['a'].fillna(means)
print(data)

meds = data['b'].median()
print(meds)
data['a'] = data['a'].fillna(meds)
print(data)





