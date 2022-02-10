import numpy as np
import matplotlib.pyplot as plt

alpa = 0.5
def elu(x) :
    
    return (x>=0)*x + (x<0)*alpa*(np.exp(x)-1)  

x = np.arange(-5, 5, 0.1)
y = elu(x)

print(x)
print(y)

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()