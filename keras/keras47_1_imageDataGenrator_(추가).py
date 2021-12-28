import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5, 
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen=ImageDataGenerator(
    rescale=1./255
)
    
#D:\_data\image\brain


xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/', 
    target_size=(150,150),
    batch_size=5, 
    class_mode='binary',
    shuffle=True, seed=66,
    color_mode='grayscale',
    save_to_dir='../_temp/'
)     #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary', 
    shuffle=True,
)    #Found 120 images belonging to 2 classes.

print(xy_train)
# #<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022087FB6F40>

# #from sklearn.datasets import load_boston
# #datasets = load_boston()
# #print(datasets)

print(xy_train[31])  #마지막 배치
print(xy_train[0][0])
print(xy_train[0][1])
# # print(xy_train[0][2]) #error
print(xy_train[0][0].shape, xy_train[0][1].shape) #(5, 150, 150, 3) #(5,)


# print(type(xy_train)) #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>










