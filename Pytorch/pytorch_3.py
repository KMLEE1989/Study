from lightgbm import early_stopping
import numpy as np
import tensorboard
import tensorflow as tf

print(tf.__version__)

from tensorflow import keras
print(keras.__version__)

keras.layers.Dense(10, activation='sigmoid')


keras.Model()

keras.models.Sequential()

from tensorflow.keras.layers import Dense, Input, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

Dense(10, activation='relu')

Flatten(input_shape=[28, 28])

X_train = np.random.randn(5500, 2)

Input(shape=X_train.shape[1:])

dense = Dense(10, activation='relu', name='Dense Layer')

dense

dense2 = Dense(15, activation='softmax')

dense

dense = Dense(10, kernel_initializer='he_normal', name = 'Dense Layer')
dense = Activation(dense)
dense

Flatten(input_shape=(28,28))

input_1 = Input(shape=(28,28), dtype=tf.float32)
input_2 = Input(shape=(8,), dtype=tf.int32)

input_1

input_2

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model

model = Sequential()
model.add(Input(shape=(28,28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

plot_model(model)

plot_model(model, to_file='model1.png')

model = Sequential([Input(shape=(28,28), name="Input"),
                    Dense(300, activation='relu', name='Dense1'),
                    Dense(100, activation='relu', name='Dense2'),
                    Dense(10, activation='softmax', name='Output')])

model.summary()

plot_model(model)

plot_model(model, to_file='model2.png')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.utils import plot_model

inputs = Input(shape=(28, 28, 1))
x = Flatten(input_shape=(28,28,1))(inputs)
x = Dense(300, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

model.summary()

plot_model(model, to_file='model3.png')

from tensorflow.keras.layers import Concatenate

input_layer = Input(shape=(28,28))
hidden1 = Dense(100, activation='relu')(input_layer)
hidden2 = Dense(30, activation='relu')(hidden1)
concat = Concatenate()([input_layer, hidden2])
output = Dense(1)(concat)

model = Model(inputs=[input_layer], outputs=[output])

model.summary()

plot_model(model)

plot_model(model, to_file='model4.png')

input_1 = Input(shape=(10,10), name='input_1')
input_2 = Input(shape=(10,28), name='input_2')

hidden1 = Dense(100, activation='relu')(input_2)
hidden2 = Dense(10, activation='relu')(hidden1)
concat = Concatenate()([input_1, hidden2])
output = Dense(1, activation='sigmoid', name='output')(concat)

model = Model(inputs=[input_1, input_2], outputs=[output])

model.summary()

plot_model(model)

plot_model(model, to_file='model5.png')

input_ = Input(shape=(10,10), name='input_')

hidden1 = Dense(100, activation='relu')(input_)
hidden2 = Dense(10, activation='relu')(hidden1)

output = Dense(1, activation='sigmoid', name='main_output')(hidden2)
sub_out = Dense(1, name='sum_output')(hidden2)

model = Model(inputs=[input_], outputs=[output, sub_out])

model.summary()

plot_model(model)

plot_model(model, to_file='model6.png')

input_1 = Input(shape=(10,10), name='input_1')
input_2 = Input(shape=(10, 28), name='input_2')

hidden1 = Dense(100, activation='relu')(input_2)
hidden2 = Dense(10, activation='relu')(hidden1)
concat = Concatenate()([input_1, hidden2])
output = Dense(1, activation='sigmoid', name = 'main_output')(concat)
sub_out = Dense(1, name='sum_output')(hidden2)

model = Model(inputs=[input_1, input_2], outputs=[output, sub_out])

model.summary()

plot_model(model)

plot_model(model, to_file='model7.png')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.utils import plot_model

class MyModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super(MyModel, self).__init__(**kwargs)
        
        self.dense_layer1 = Dense(300, activation=activation)
        self.dense_layer2 = Dense(100, activaiton=activation)
        self.dense_layer3 = Dense(units, activation=activation)
        
        self.output_layer = Dense(10, activation='softmax')
        
    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        x = self.dense_layer3(x) 
        x = self.output_layer(x) 
        return x
    
from tensorflow.keras.models import Model                   
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.utils import plot_model

inputs = Input(shape=(28,28,1))

x = Flatten(input_shape=(28,28,1))(inputs)
x = Dense(300, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

model.summary()

model.layers

hidden_2 = model.layers[2]
hidden_2.name

model.get_layer('dense_20') is hidden_2

weights, biases = hidden_2.get_weights()

weights

biases

print(weights.shape)
print(biases.shape)

model.compile(loss='sparse_categorica_crossentropy', optimizer='sgd', metrics=['accuracy'])

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

tf.random.set_seed(111)
(x_train_full, y_train_full), (x_test, y_test) = load_data(path='mnist.npz')

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3, random_state=111)

num_x_train = (x_train.shape[0])
num_x_val = (x_val.shape[0])
num_x_test = (x_test.shape[0])

print('학습 데이터: {}\t레이블:{}'.format(x_train_full.shape, y_train_full.shape))
print('학습 데이터: {}\t레이블:{}'.format(x_train.shape, y_train.shape))
print('학습 데이터: {}\t레이블:{}'.format(x_val.shape, y_val.shape))
print('test 데이터: {}\t레이블:{}'.format(x_test.shape, y_test.shape))

num_sample = 5

random_idxs = np.random.randint(60000, size=num_sample)

plt.figure(figsize=(14,8))
for i, idx in enumerate(random_idxs):
    img = x_train_full[idx, :]
    label = y_train_full[idx]
    
    plt.subplot(1, len(random_idxs), i+1)
    plt.imshow(img)
    plt.title('Index:{}, Label: {}'.format(idx,label))
    
y_train[0]

x_train = x_train / 255.
x_val = x_val /255.
x_test = x_test / 255.

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

model = Sequential([Input(shape=(28,28), name='input'),
                  Flatten(input_shape=[28,28], name='flatten'),
                  Dense(100, activation='relu', name='dense1'),
                  Dense(64, activation='relu', name='dense2'),
                  Dense(32, activation='relu', name='dense3'),
                  Dense(10, activation='softmax', name='output')])

model.summary()

plot_model(model)

plot_model(model, show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val))

history.history.keys()

history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss)+ 1)
fig= plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(1,2,1)
ax1.plot(epochs,loss,color='blue', label='train_loss')
ax1.plot(epochs, val_loss, color='red', label='val_loss')
ax1.set_title('Train and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid()
ax1.legend()

accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']


ax2 = fig.add_subplot(1,2,1)
ax2.plot(epochs,loss,color='blue', label='train_accuracy')
ax2.plot(epochs, val_loss, color='red', label='val_accuracy')
ax2.set_title('Train and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid()
ax2.legend()

plt.show()

model.evaluate(x_test, y_test)

pred_ys = model.predict(x_test)
print(pred_ys.shape)

np.set_printoptions(precision=7)
print(pred_ys[0])

arg_pred_y = np.argmax(pred_ys, axis=1)

plt.imshow(x_test[0])
plt.title('predicted label: {}'.format(arg_pred_y[0]))
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.set(style='white')

plt.figure(figsize=(10,10))
cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(pred_ys, axis=-1))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(classification_report(np.argmax(y_test, axis=-1), np.argmax(pred_ys, axis=-1)))

model.save('mnist_model.h5')

loaded_model = models.load_model('mnist_model.h5')

pred_ys2 = loaded_model.predict(x_test)
print(pred_ys2.shape)

np.set_printoptions(precision=7)
print(pred_ys2[0])

arg_pred_y2 = np.argmax(pred_ys2, axis=1)

plt.imshow(x_test[0])
plt.title('predicted label: {}'.format(arg_pred_y2[0]))
plt.show()

(x_train_full, y_train_full), (x_test, y_test) = load_data(path='mnist.npz')

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3, random_state=111)

print('학습 데이터: {}\t레이블:{}'.format(x_train_full.shape, y_train_full.shape))
print('학습 데이터: {}\t레이블:{}'.format(x_train.shape, y_train.shape))
print('학습 데이터: {}\t레이블:{}'.format(x_val.shape, y_val.shape))
print('test 데이터: {}\t레이블:{}'.format(x_test.shape, y_test.shape))


x_train = x_train / 255.
x_val = x_val /255.
x_test = x_test / 255.

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

def build_model():
    model = Sequential([Input(shape=(28,28), name='input'),
                  Flatten(input_shape=[28,28], name='flatten'),
                  Dense(100, activation='relu', name='dense1'),
                  Dense(64, activation='relu', name='dense2'),
                  Dense(32, activation='relu', name='dense3'),
                  Dense(10, activation='softmax', name='output')])
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return model

model = build_model()

model.summary()

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard

check_point_cb = ModelCheckpoint('keras_mnist_model.h5')
history = model.fit(x_train, y_train, epochs=10, callbacks=[check_point_cb])

history.history.keys()

loaded_model = load_model('keras_mnist_model.h5')
loaded_model.summary()

model = build_model()

cp = ModelCheckpoint('keras_best_model.h5', save_best_only=True)

history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_val, y_val), callbacks=[cp])

history.history.keys()

loaded_model2 = load_model('keras_best_model.h5')
loaded_model2.summary()

model = build_model()
cp = ModelCheckpoint('keras_best_model2.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), callbacks=[cp, early_stopping_cb])

def scheduler(epoch, learning_rate):
    if epoch < 10:
        return learning_rate
    else: 
        return learning_rate * tf.math.exp(-0.1)
    
model = build_model()

round(model.optimizer.lr.numpy(), 5)

lr_scheduler_cb = LearningRateScheduler(scheduler)

history = model.fit(x_train, y_train, epochs=15, callbacks=[lr_scheduler_cb], verbose=0)

round(model.optimizer.lr.numpy(), 5)

TensorBoard(log_dir='.logs', histogram_freq=0, write_graph=True, write_images=True)

log_dir='./logs'

tensor_board_cb= [TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)]

model = build_model()

model.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=30, callbacks=tensor_board_cb)




