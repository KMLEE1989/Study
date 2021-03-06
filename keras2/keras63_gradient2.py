import numpy as np


f = lambda x: x**2 - 4*x + 6 

x=2
print(f(x))

gradient = lambda x : 2*x - 4

x = 8.0             #초기값
epochs=20
learning_rate = 0.15

print("step\t   x\t f(x)")
print("{:02d}\t {:6.5f}\t   {:6.5f}\t".format(0, x, f(x)))

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    
    print("{:02d}\t {:6.5f}\t   {:6.5f}\t".format(0+1, x, f(x)))
    
    
# step     x       f(x)
# 00       0.00000         6.00000
# 01       1.00000         3.00000
# 02       1.50000         2.25000
# 03       1.75000         2.06250
# 04       1.87500         2.01562
# 05       1.93750         2.00391
# 06       1.96875         2.00098
# 07       1.98438         2.00024
# 08       1.99219         2.00006
# 09       1.99609         2.00002
# 10       1.99805         2.00000
# 11       1.99902         2.00000
# 12       1.99951         2.00000
# 13       1.99976         2.00000
# 14       1.99988         2.00000
# 15       1.99994         2.00000
# 16       1.99997         2.00000
# 17       1.99998         2.00000
# 18       1.99999         2.00000
# 19       2.00000         2.00000
# 20       2.00000         2.00000