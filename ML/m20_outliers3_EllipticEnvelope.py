#실습 다차원의 아웃라이어가 출력 되도록
from matplotlib.pyplot import boxplot
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np

aaa = np.array([[1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600,7, 800, 900, 190, 1001, 1002, 99]])
aaa = np.transpose(aaa)

# df = pd.DataFrame(aaa, columns=['x','y'])

# data1 = df[['x']]
# data2 = df[['y']]

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.15)
pred = outliers.fit_predict(aaa)
print(pred.shape) # (13,)

b = list(pred)
print(b.count(-1))
index_for_outlier = np.where(pred == -1)
print('outier indexex are', index_for_outlier)
outlier_value = aaa[index_for_outlier]
print('outlier_value :', outlier_value)

#aaa= np.array([[1,2,-20,4,5,6,7,8,30,100,500,12,13],
#              [100,200,3,400,500,600,7,800,900,190,1001,1002,99]])

#(2,13) -> (13,2)
"""
columns=['x','y']
aaa = np.array([[1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600,7, 800, 900, 190, 1001, 1002, 99]])
aaa = np.transpose(aaa)
df = pd.DataFrame(aaa, columns=['x','y'])
data1 = df[['x']]
data2 = df[['y']]


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)

outliers.fit(data1)
outliers.fit(data2)

result1 = outliers.predict(data1)
result2 = outliers.predict(data2)

print(result1)
print(result2)


plt.boxplot(aaa)
plt.show()
"""
'''
aaa= np.array([[1,2,-20,4,5,6,7,8,30,100,500,12,13],
              [100,200,3,400,500,600,7,800,900,190,1001,1002,99]])
aaa=np.transpose(aaa) #(13,2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)

outliers.fit(aaa)

results=outliers.predict(aaa)
print(results)

plt.boxplot(aaa)
plt.show()

'''