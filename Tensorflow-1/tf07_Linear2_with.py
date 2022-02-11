# y = wx + b
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

w = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

#2. 모델구성
hypothesis = x_train * w + b      # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
with tf.compat.v1.Session() as sess:
    # sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))
        
sess.close()