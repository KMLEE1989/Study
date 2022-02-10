# LeakyReLU 함수

import numpy as np
import matplotlib.pyplot as plt

def lekrelu(x) :
    return np.maximum(0.01*x, x) 

x = np.arange(-5, 5, 0.1)
y = lekrelu(x)

print(x)
print(y)

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()

