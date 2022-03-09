import numpy as np
import tensorflow as tf

print(tf.__version__)

a = tf.constant(2)
# print(tf.rank(a))
# print(a)

# tf.Tensor(0, shape=(), dtype=int32)
# tf.Tensor(2, shape=(), dtype=int32)

b = tf.constant([2, 3])
# print(tf.rank(b))
# print(b)
# tf.Tensor(1, shape=(), dtype=int32)
# tf.Tensor([2 3], shape=(2,), dtype=int32)

c = tf.constant([[2,3], [6,7]])
# print(tf.rank(c))
# print(c)


# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor(
# [[2 3]
#  [6 7]], shape=(2, 2), dtype=int32)

d = tf.constant(['Hello'])
# print(tf.rank(d))
# print(d)
# tf.Tensor(1, shape=(), dtype=int32)
# tf.Tensor([b'Hello'], shape=(1,), dtype=string)

rand = tf.random.uniform([1], 0, 1)
# print(rand.shape)
# print(rand)

# (1,)
# tf.Tensor([0.4524871], shape=(1,), dtype=float32)

rand2 = tf.random.normal([1,2], 0, 1)
# print(rand2.shape)
# print(rand2)

rand3 = tf.random.normal(shape=(3,2), mean=0, stddev=1)
# print(rand3.shape)
# print(rand3)

a = tf.constant(3)
b = tf.constant(2)

# print(tf.add(a,b))
# print(a + b)

# print(tf.subtract(a,b))
# print(a-b)

# print(tf.multiply(a,b))
# print(a*b)

c = tf.add(a,b).numpy()
# print(type(c))

c_square = np.square(c, dtype=np.float32)
c_tensor = tf.convert_to_tensor(c_square)

# print(c_tensor)
# print(type(c_tensor))

t = tf.constant([[1.,2.,3.],[4.,5.,6.]])

# print(t.shape)
# print(t.dtype)

# print(t[:, 1:])

# print(t[...,1,tf.newaxis])

# print(t + 10)

# print(tf.square(t))

t @ tf.transpose(t)
# print(t @ tf.transpose(t))


a= tf.constant(2)
# print(a)

b= tf.constant(2.)
# print(b)

#tf.constant(2.) + tf.constant(40)

# tf.constant(2.) + tf.constant(30., dtype=tf.float64)

t = tf.constant(30., dtype=tf.float64)
t2 = tf.constant(4. )

print(t2 + tf.cast(t, tf.float32))

import timeit

@tf.function 

def my_function(x):
    return x**2 -10*x + 3

# print(my_function(2))
# print(my_function(tf.constant(2)))

def my_function_(x):
    return x**2 -10*x +3

# print(my_function_(2))
# print(my_function_(tf.constant(2)))

tf_my_func = tf.function(my_function_)
# print(tf_my_func)
# print(tf_my_func(2))

tf_my_func.python_function(2)

def function_to_get_faster(x,y,b):
    x = tf.matmul(x,y)
    x= x+b
    return x

a_function_that_uses_a_graph = tf.function(function_to_get_faster)

x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

# print(a_function_that_uses_a_graph(x1, y1, b1).numpy())

def inner_function(x,y,b):
    x = tf.matmul(x,y)
    x=x+b
    return x

@tf.function
def outer_function(x):
    y=tf.constant([[2.0],[3.0]])
    b=tf.constant(4.0)
    return inner_function(x,y,b)

outer_function(tf.constant([[1.0,2.0]])).numpy()

# print(tf.autograph.to_code(my_function.python_function))
# print(tf.autograph.to_code(tf_my_func.python_function))
# print(tf.autograph.to_code(outer_function.python_function))

class SequentialModel(tf.keras.Model):
    def __init__(self, **kwargs): 
        super(SequentialModel, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28)) 
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(10)
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x
    
input_data = tf.random.uniform([60, 28, 28])
eager_model = SequentialModel()
graph_model = tf.function(eager_model)

# print('Eager time:', timeit.timeit(lambda:eager_model(input_data), number=10000))
# print('Graph time:', timeit.timeit(lambda:graph_model(input_data), number=10000))

X = tf.Variable(20.0)

# print(X)

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2
    
dy_dx = tape.gradient(y, x)
# print(dy_dx.numpy())

# x2 = tf.Variable(4)
# dy_dx = tape.gradient(y, x2)
# dy_dx.numpy()

x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y_sq = y**2
    z = x**2 + tf.stop_gradient(y_sq)
    
grad = tape.gradient(z, {'x':x , 'y':y})

# print('dz/dx:', grad['x'])

# print('dz/dy:', grad['y'])

weight = tf.Variable(tf.random.normal((3,2)), name='weight')
biases = tf.Variable(tf.zeros(2, dtype=tf.float32), name='biases')
x = [[1.,2.,3.]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ weight +biases
    loss = tf.reduce_mean(y**2)
    
[dl_dw, dl_db] = tape.gradient(loss, [weight, biases])

# print(weight.shape)
# print(dl_dw.shape)

weight2 = tf.Variable(tf.random.normal((3,2)), name='weight')
biases2 = tf. Variable(tf.zeros(2, dtype=tf.float32), name='biases')

x = [[4.,5.,6.]]

[dl_dw2, dl_db2] = tape.gradient(loss, [weight2, biases2])
  
# print(weight2.shape) 
# print(dl_dw.shape)

del tape

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def Neuron(x, W, bias=0):
    z = x * W + bias
    return sigmoid(z)

x = tf.random.normal((1,2), 0, 1)
W = tf.random.normal((1,2), 0 ,1)

# print('x.shape:', x.shape)
# print('W.shape:', W.shape)

# print(x)
# print(W)

# print(Neuron(x, W))


x = tf.random.normal((2,1), 0, 1)
W = tf.random.normal((2,1), 0 ,1)

print('x.shape:', x.shape)
print('W.shape:', W.shape)

# print(x)
# print(W)

# print(Neuron(x, W))


x = tf.random.normal((1,), 0, 1)
W = tf.random.normal((2,1), 0 ,1)

print('x.shape:', x.shape)
print('W.shape:', W.shape)

print(x)
print(W)

# print(Neuron(x, W))

x = 1
y = 0
W = tf.random.normal([1], 0, 1)
print(Neuron(x, W))
print('y:', y)

for i in range(1000):
    output = Neuron(x, W)
    error = y - output
    W = W + x *0.1*error

    if i % 100 == 99:
        print("{}\t{}\t{}". format(i+1, error, output))
        
        
def Neuron2(x, W, bias=0):
    z = tf.matmul(x, W, transpose_b=True) + bias
    return sigmoid(z)

x = tf.random.normal((1,3),0,1)
y = tf.ones(1)
W = tf.random.normal((1,3),0,1)

print(Neuron2(x, W))
print('y:', y)

for i in range(1000):
    output = Neuron2(x,W)
    error = y - output
    W = W + x * 0.1 * error
    
    if i % 100 == 99:
        print("{}\t{}\t{}".format(i+1, error, output))
        
for i in range(1000):
    output = Neuron2(x,W)
    error = y - output
    W = W + x*0.1*error
    
    if i % 100 == 99:
        print("{}\t{}\t{}".format(i+1, error, output))
        
        
x = tf.random.normal((1,3),0,1)
weights = tf.random.normal((1,3),0,1)
bias = tf.zeros((1,1))

y = tf.ones((1,))

print("x\t: {}\nweights\t: {}\nbias\t: {}".format(x,weights, bias))

for i in range(1000):
    output = Neuron2(x, weights, bias=bias)
    error = y - output
    weights = weights + x * 0.1 * error
    bias = bias + 1 * 0.1 * error
    
    if i % 100 == 99:
        print("{}\t{}\t{}".format(i+1, error, output))
        
print("x\t: {}\nweights\t: {}\nbias\t: {}".format(x,weights, bias))

X = np.array([[1,1],[1,0],[0,1],[0,0]])
Y = np.array([[1],[0],[0],[0]])

W = tf.random.normal([2], 0, 1)
b = tf.random.normal([1], 0, 1)
b_x = 1

for i in range(2000):
    error_sum = 0
    
    for j in range(4):
        output = sigmoid(np.sum(X[j] * W) + b_x + b)
        error = Y[j][0] - output
        W = W + X[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error
        
    if i % 200 == 0:
        print("Epoch {:4d}\tError Sum {}".format(i, error_sum))
        
print("\n가중치\t: {}".format(W))
print('편향\t:{}'.format(b))

for i in range(4):
    print("X: {} Y:{} Output : {}".format(X[i], Y[i], sigmoid(np.sum(X[i] * W)+ b)))
    
X2 = np.array([[1,1],[1,0],[0,1],[0,0]])
Y2 = np.array([[1],[0],[0],[0]])

W2= tf.random.normal([2], 0, 1)
b2 = tf.random.normal([1], 0, 1)
b_x = 1

for i in range(2000):
    error_sum = 0
    
    for j in range(4):
        output = sigmoid(np.sum(X2[j] * W2) + b_x + b2)
        error = Y2[j][0] - output
        W2 = W2 + X2[j] * 0.1 * error
        b2 = b2 + b_x * 0.1 * error
        error_sum += error
        
    if i % 200 == 0:
        print("Epoch {:4d}\tError Sum {}".format(i, error_sum))
        
print("\n가중치\t: {}".format(W2))
print('편향\t:{}'.format(b2))

for i in range(4):
    print("X: {} Y:{} Output : {}".format(X2[i], Y2[i], sigmoid(np.sum(X2[i] * W2)+ b2)))
    
X3 = np.array([[1,1],[1,0],[0,1],[0,0]])
Y3 = np.array([[0],[1],[1],[0]])

W3= tf.random.normal([2], 0, 1)
b3 = tf.random.normal([1], 0, 1)
b_x = 1

for i in range(2000):
    error_sum = 0
    
    for j in range(4):
        output = sigmoid(np.sum(X3[j] * W3) + b_x + b3)
        error = Y2[j][0] - output
        W3 = W3 + X3[j] * 0.1 * error
        b3 = b3 + b_x * 0.1 * error
        error_sum += error
        
    if i % 200 == 0:
        print("Epoch {:4d}\tError Sum {}".format(i, error_sum))
        
print("\n가중치\t: {}".format(W3))
print('편향\t:{}'.format(b3))

for i in range(4):
    print("X: {} Y:{} Output : {}".format(X3[i], Y3[i], sigmoid(np.sum(X3[i] * W3)+ b3)))
    
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

np.random.seed(111)

X4 = np.array([[1,1], [1,0],[0,1],[0,0]])
Y4 = np.array([[0],[1],[1],[0]])

model = Sequential([Dense(units=2, activation = 'sigmoid', input_shape=(2,)),
                    Dense(units=1, activation='sigmoid')])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),loss='mse')

model.summary()

history = model.fit(X4, Y4, epochs=2000, batch_size=1, verbose=0)

model.predict(X4)

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

x = range(50)
y = tf.random.normal([50], 0 ,1)
plt.plot(x,y)
plt.show()

x = range(50)
y = tf.random.normal([50], 0 ,1)
plt.plot(x,y, 'ro')
plt.show()

x = range(30)
y = tf.random.normal([30], 0 ,1)
plt.plot(x,y, 'g-')
plt.show()

x = range(30)
y = tf.random.normal([30], 0 ,1)
plt.plot(x,y, 'b--')
plt.show()

random_normal = tf.random.normal([10000], 0 ,1)
plt.hist(random_normal, bins=100)
plt.show()

plt.plot(history.history['loss']);







