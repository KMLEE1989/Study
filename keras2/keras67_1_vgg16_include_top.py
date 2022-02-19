from pyexpat import model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

model = VGG16()
model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

model.summary()

print(len(model.weights))  #32
print(len(model.trainable_weights)) #32

############################################################ include_top=True#######################################
# 1. FC layer 원래꺼 그대로 쓴다.
# 2. input_shape = (224,224,3)  고정 - 바꿀 수 없다.
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928

#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856

#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584

#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168

#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160

#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

####################################################################################################################


################################################### include_top = False ##########################################
# 1. FC layer 원래꺼 삭제!!! -> 앞으로 난 커스터마이징을 할거야!!!
# 2. input_shape = (32,32,3) 바꿀 수 있다.  - 커스터마이징을 할거야!!!!!
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_2 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928

#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856

#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584

#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168

#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160

#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# 26
# 26
#####################################################################################

# FC(Fully connected layer)를 정의하자면,

# 완전히 연결 되었다라는 뜻으로,

# 한층의 모든 뉴런이 다음층이 모든 뉴런과 연결된 상태로

# 2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층입니다.


# 1. 2차원 배열 형태의 이미지를 1차원 배열로 평탄화

# 2. 활성화 함수(Relu, Leaky Relu, Tanh,등)뉴런을 활성화

# 3. 분류기(Softmax) 함수로 분류




