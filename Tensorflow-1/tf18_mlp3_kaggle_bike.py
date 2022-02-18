from sklearn.datasets import load_boston, load_diabetes
import tensorflow as tf
import pandas as pd
import numpy as np

tf.set_random_seed(66)

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file.shape)  # (6493, 2)

x_data = train.drop(['datetime', 'casual','registered','count'], axis=1)  
#print(x.shape)  # (10886, 8)

y_data = train['count']
# print(y_data.shape)  # (10886,)

y_data = np.array(y_data)
y_data = y_data.reshape(10886,1)
print(y_data.shape)   # (10886,1)

test_file = test_file.drop(['datetime'], axis=1)  

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.7, shuffle=True, random_state=66)

x = tf.placeholder(tf.float32, shape = [None,8])
y = tf.placeholder(tf.float32, shape = [None,1])
# w = tf.compat.v1.Variable(tf.random.normal([13,1], name='weight'))
# b = tf.compat.v1.Variable(tf.random.normal([1], name='bias'))

w1 = tf.compat.v1.Variable(tf.random.normal([8,80]), name = 'weight1')     
b1 = tf.compat.v1.Variable(tf.random.normal([80]), name = 'bias1')

Hidden_layer1=tf.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.random.normal([80,100]), name = 'weight2')     
b2 = tf.compat.v1.Variable(tf.random.normal([100]), name = 'bias2')

Hidden_layer2=tf.matmul(Hidden_layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([100,100]), name = 'weight3')     
b3 = tf.compat.v1.Variable(tf.random.normal([100]), name = 'bias3')

Hidden_layer3=tf.matmul(Hidden_layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([100,80]), name = 'weight4')     
b4 = tf.compat.v1.Variable(tf.random.normal([80]), name = 'bias4')

Hidden_layer4=tf.matmul(Hidden_layer3, w4) + b4

w5 = tf.compat.v1.Variable(tf.random.normal([80,1]), name = 'weight5')     
b5 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias5')

# Hidden_layer5=tf.matmul(Hidden_layer4, w5) + b5

# w6 = tf.compat.v1.Variable(tf.random.normal([100,100]), name = 'weight6')     
# b6 = tf.compat.v1.Variable(tf.random.normal([100]), name = 'bias6')

# Hidden_layer6=tf.matmul(Hidden_layer5, w6) + b6

# w7 = tf.compat.v1.Variable(tf.random.normal([100,100]), name = 'weight7')     
# b7 = tf.compat.v1.Variable(tf.random.normal([100]), name = 'bias7')

# Hidden_layer7=tf.matmul(Hidden_layer6, w7) + b7

# w8 = tf.compat.v1.Variable(tf.random.normal([100,100]), name = 'weight8')     
# b8 = tf.compat.v1.Variable(tf.random.normal([100]), name = 'bias8')

# Hidden_layer8=tf.matmul(Hidden_layer7, w8) + b8

# w9 = tf.compat.v1.Variable(tf.random.normal([100,100]), name = 'weight9')     
# b9 = tf.compat.v1.Variable(tf.random.normal([100]), name = 'bias9')

# Hidden_layer9=tf.matmul(Hidden_layer8, w9) + b9

# w10 = tf.compat.v1.Variable(tf.random.normal([100,1]), name = 'weight10')     
# b10 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias10')

hypothesis=tf.matmul(Hidden_layer4, w5) + b5
#2. 모델구성
# hypothesis =  tf.matmul(x,w) + b

# w1 = tf.compat.v1.Variable(tf.random.normal([2,30], name='weight1'))
# b1 = tf.compat.v1.Variable(tf.random.normal([30]), name='bias1')


# Hidden_layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)


#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)  
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val = sess.run([train, loss, w5], feed_dict={x:x_train, y:y_train})
    if epochs % 200 ==0:
        print(epochs, '\t', loss_val, '\t', w_val)
    
#4. 예측
predict = tf.matmul(Hidden_layer4, w_val) + b5  
y_predict = sess.run(predict, feed_dict={x:x_test, y:y_test})
# print("예측 : " , y_predict)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)

# r2스코어 :  0.25942368641835334
# mae :  116.7271304882821

sess.close()
