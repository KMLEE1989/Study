import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0], [1],[1],[0]]


#input Layer
x= tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y= tf.compat.v1.placeholder(tf.float32, shape=[None, 1])


#2. 모델구성
#input Layer
w1 = tf.compat.v1.Variable(tf.random.normal([2,30], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random.normal([30]), name='bias1')


Hidden_layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)
# Hidden_layer1 = tf.matmul(x,w1)+b1
# Hidden_layer1 = tf.nn.relu(tf.matmul(x,w1)+b1)
#hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

w2 = tf.compat.v1.Variable(tf.random.normal([30,1], name='weight2'))
b2 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias2')

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer1 , w2)+b2)
# #시작

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(10001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, 'loss:', loss_val, '\n', hy_val)
    
#4. 예측
y_predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)       # tf.cast = 자료형을 바꿔주는것 (FFFTTT를 float형인 0.0.0.1.1.1.로 바꿔주겠다!) // False면 0, True면 1      
# print(y_predict) # Tensor("Cast:0", shape=(?, 1), dtype=float32)
# print(sess.run(hypothesis > 0.5, feed_dict ={x:x_data, y:y_data}))  # [False] , [True]
# print(sess.run(y_predict, feed_dict ={x:x_data, y:y_data}))  # [0.] , [1.]
# print(sess.run(tf.equal(y, y_predict), feed_dict ={x:x_data, y:y_data}))

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype = tf.float32))     # tf.equal: 비교역할 / tf.cast: 동일하면 1 동일하지 않으면 0 반환

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})    # 사람이 해석할 수 있게 변환됨
print("=========================")
print("예측값: \n", hy_val)
print("예측결과: \n", pred)
print("Accuracy: ", acc)

# sess.close() 

# 예측값:
#  [[0.50303024]
#  [0.5006782 ]
#  [0.5002729 ]
#  [0.49792078]]
# 예측결과:
#  [[1.]
#  [1.]
#  [1.]
#  [0.]]
# Accuracy:  0.75

# 예측값:
#  [[0.00651015]
#  [0.9693821 ]
#  [0.96618986]
#  [0.04648123]]
# 예측결과:
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# Accuracy:  1.0