#Original
import numpy as np

"""
a=np.array(range(1,11))
size= 5


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-size) :
        aaa=[]
        for i in range(len(dataset)-size+1):
            subset = dataset[i: (i+size)]
            aaa.append(subset)
        return np.array(aaa)
dataset=split_x(a,size)

print(dataset)
"""


a=np.array(range(1,11))
size= 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-size) :
        aaa=[]
        for i in range(len(dataset)-size+1):
            subset = dataset[i : (i+size)]
            aaa.append(subset)
        return np.array(aaa)
bbb=split_x(a,size)

print(bbb)


print(bbb)

print(bbb.shape) #(6,5)

x=bbb[:, :4]
y=bbb[:, 4]
print(x,y)


print(x.shape, y.shape)  #(6,4) (6,)

"""
a=np.array(range(1,11))  # a=np.array
b=5                        #b=int

def split_x(a,b) :
    for i in range(len(a-b+1)):
        c=a[i:1+b]  #c=반복해서 적용할 리스트
"""