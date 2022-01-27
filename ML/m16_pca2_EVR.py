import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action="ignore")
import sklearn as sk

#1. 데이터
#datasets = fetch_california_housing()
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

# print(x.shape) #(569, 30)

pca=PCA(n_components=14)
x = pca.fit_transform(x)
# print(x)
# print(x.shape)  #(569, 14)

pca_EVR= pca.explained_variance_ratio_
# print(pca_EVR)

# [9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
#  8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
#  3.44135279e-07 1.86018721e-07 6.99473205e-08 1.65908880e-08
#  6.99641650e-09 4.78318306e-09]=0.982, 0.016,0.0015

# print(sum(pca_EVR)) 0.9999999930016484

cumsum=np.cumsum(pca_EVR)
print(cumsum)
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999]

import matplotlib.pyplot as plt
plt.plot(cumsum)
#plt.plot(pca_EVR)
plt.grid()
plt.show()








