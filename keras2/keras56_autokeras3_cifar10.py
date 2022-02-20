import autokeras as ak
import tensorflow as tf
cifar10 = tf.keras.datasets.cifar10
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#2. 모델
model = ak.ImageClassifier(
    overwrite = True,
    max_trials = 5,     # model을 두 번 돌리겠다. 
)

#3. 컴파일, 훈련
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측
y_pred = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)
# model.summary()

# [0.7934976816177368, 0.7299000024795532]