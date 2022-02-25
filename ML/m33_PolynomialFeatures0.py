import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)

print(x)
print(x.shape)  #(5,2)

# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# (4, 2)

pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)
print(xp)
print(xp.shape) 

############################################################################################################################

x = np.arange(12).reshape(4,3)

print(x)
print(x.shape)  #(4,3)

# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# (4, 2)

pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)
print(xp)
print(xp.shape)  #(4, 10)

############################################################################################################################

x = np.arange(8).reshape(4,2)

print(x)
print(x.shape)  #(4,3)

# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# (4, 2)

pf = PolynomialFeatures(degree=3)

xp = pf.fit_transform(x)
print(xp)
print(xp.shape)  #(4, 10)