from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성

# Layer1
w1 = tf.get_variable('w1', shape=[2, 2, 1, 128])   # shape=[2, 2, 1] -> kernel_size // ,64 -> output(filter)
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')  # strides=[1, "1, 1", 1] -> 가운데 두개만 stride임. 양 사이드 두개는 shape맞추기위한 허수임(1로채우면됨)
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='same', input_shape=(28,28,1)))

print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 128) dtype=float32_ref>
print(L1)   # Tensor("Relu:0", shape=(?, 28, 28, 128), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool:0", shape=(?, 14, 14, 128), dtype=float32)

# Layer2
w2 = tf.get_variable('w2', shape=[3, 3, 128, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME')  
# print(L2)   # Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2_maxpool)   # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# Layer3
w3 = tf.get_variable('w3', shape=[3, 3, 64, 32])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')  
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L3_maxpool)   # Tensor("MaxPool_2:0", shape=(?, 4, 4, 32), dtype=float32)

# Layer4
w4 = tf.get_variable('w4', shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='SAME')  
L4 = tf.nn.elu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L4_maxpool)   # Tensor("MaxPool_3:0", shape=(?, 2, 2, 32), dtype=float32)

# Flatten 
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*32])
print("플래튼 : ", L_flat)    # 플래튼 : Tensor("Reshape:0", shape=(?, 128), dtype=float32)

# Layer5 DNN
w5 = tf.get_variable('w5', shape=[2*2*32, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([64]), name='b5')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.7)
print(L5)

# Layer6 DNN
w6 = tf.get_variable('w6', shape=[64, 32],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]), name='b6')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
L6 = tf.nn.dropout(L6, keep_prob=0.7)
print(L6)

# Layer7 softmax
w7 = tf.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random_normal([10]), name='b7')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis)

#3-1. 컴파일 훈련
# categorica_crossentropy

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

training_eopchs = 10
batch_size = 100
total_batch = int(len(x_train)/batch_size)
print(total_batch)

#3-2. 훈련
for epochs in range(training_eopchs):
    avg_loss = 0
    
    for i in range(total_batch):  #몊번? 600 번!!    
        start = i*batch_size                   #0
        end = start + batch_size     #100
        batch_x, batch_y = x_train[start:end], y_train[start:end]   #0~100
        
        feed_dict = {x:batch_x, y:batch_y}  #단순변수
        
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        
        avg_loss += batch_loss / total_batch
        
    print('Epoch: ', '%04d' %(epochs + 1), 'loss :  {:.9f}'.format(avg_loss))
    # print(f'Epoch : {epochs + 1:%04d}, loss : {avg_loss:.9f}')
    
print("훈련 끝!!!!")

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
    
    
# WARNING:tensorflow:From d:\Study\tf114\tf20_cnn3_mnist.py:122: arg_max (from tensorflow.python.ops.gen_math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use `tf.math.argmax` instead
# 2022-02-14 15:00:19.874178: W tensorflow/core/framework/allocator.cc:107] Allocation of 4014080000 exceeds 10% of system memory.
# ACC :  0.9863  ->training_eopchs = 10




'''
w5 = tf.compat.v1.Variable(tf.random.normal([128,64]), name='weight5')
L5 = tf.nn.relu(tf.matmul(L_flat, w5))
layers = tf.nn.dropout(L5, keep_prob=0.8)
print(layers)   # Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

w6 = tf.compat.v1.Variable(tf.random.normal([64,10]), name='weight6')
L6 = tf.nn.relu(tf.matmul(layers, w6))
print(L6)   # Tensor("Relu_2:0", shape=(?, 10), dtype=float32)
# print(w6)   # <tf.Variable 'weight6:0' shape=(64, 10) dtype=float32_ref>

w7 = tf.compat.v1.Variable(tf.random.normal([10,10]), name='weight7')

hypothesis = tf.nn.softmax(tf.matmul(L6, w7))

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000001).minimize(loss)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer,loss], feed_dict={x:x_train, y:y_train})
        if step % 200 ==0:
            print(step, loss_val)
    
    results = sess.run(hypothesis, feed_dict={x:x_test})
    print(results, sess.run(tf.math.argmax(results, 1)))     #[[9.3190324e-01 6.8059169e-02 3.7637248e-05]] [0]
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, results), dtype=tf.float32))
    pred, acc = sess.run([tf.math.argmax(results, 1), accuracy], feed_dict={x:x_test, y:y_test})

    print("예측결과 : ", pred)
    print("accuracy : ", acc)

    sess.close()
'''