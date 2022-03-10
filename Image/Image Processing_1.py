import cv2
from cv2 import imshow
import matplotlib
from matplotlib import image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline

from PIL import Image
import requests
from io import BytesIO

from torch import imag

url = 'https://cdn.pixabay.com/photo/2018/10/01/09/21/pets-3715733_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))

# pic.show()

# print(type(pic))
# <class 'PIL.JpegImagePlugin.JpegImageFile'>

pic_arr = np.asarray(pic)


# print(type(pic_arr))

# print(pic_arr.shape)
# # (639, 960, 3)

# print(pic_arr)
# [[[215 124 141]
#   [215 124 141]
#   [215 124 141]
#   ...
#   [233 145 169]
#   [233 145 169]
#   [233 145 169]]

#  [[215 124 141]
#   [215 124 141]
#   [215 124 141]
#   ...
#   [233 145 169]
#   [233 145 169]
#   [233 145 169]]

#  [[215 124 141]
#   [215 124 141]
#   [215 124 141]
#   ...
#   [233 145 169]
#   [233 145 169]
#   [233 145 169]]

#  ...

#  [[210 132 145]
#   [211 133 146]
#   [211 133 146]
#   ...
#   [231 153 179]
#   [230 152 178]
#   [229 151 177]]

#  [[210 132 145]
#   [211 133 146]
#   [211 133 146]
#   ...
#   [229 151 177]
#   [229 151 177]
#   [229 151 177]]

#  [[210 132 145]
#   [211 133 146]
#   [211 133 146]
#   ...
#   [228 150 176]
#   [230 152 178]
#   [231 153 179]]]

# plt.imshow(pic_arr)
# plt.show()

# pic_copy = pic_arr.copy()

# plt.imshow(pic_copy)
# plt.show()

# print(pic_copy.shape)
# # (639, 960, 3)

# print(pic_copy[:,:,0])
# print(pic_copy[:,:,0].shape)

# plt.imshow(pic_copy[:,:,0])
# plt.show()

# plt.imshow(pic_copy[:,:,0], cmap='gray')
# plt.show()

# print(pic_copy[:,:,1])
# print(pic_copy[:,:,1].shape)
# (639, 960)

# plt.imshow(pic_copy[:,:,1], cmap='gray')
# plt.show()

# print(pic_copy[:,:,2])
# print(pic_copy[:,:,2].shape)

# plt.imshow(pic_copy[:,:,2], cmap='gray')
# plt.show()

# pic_red = pic_arr.copy()
# pic_red[:,:,1] = 0
# pic_red[:,:,2] = 0

# print(pic_red)

# plt.imshow(pic_red)
# plt.show()

# pic_green = pic_arr.copy()
# pic_green[:,:,0] = 0
# pic_green[:,:,2] = 0

# print(pic_green)
# plt.imshow(pic_green)
# plt.show()

# pic_blue = pic_arr.copy()
# pic_blue[:,:,0] = 0
# pic_blue[:,:,1] = 0

# print(pic_blue)

# plt.imshow(pic_blue)
# # plt.show()

# print(pic_arr)


# cv2.imshow('IMG', pic_arr)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()


# image = cv2.cvtColor(pic_arr, cv2.COLOR_RGB2BGR)
# cv2.imshow('IMG', image)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()

# print(image[0][0])
# # [141 124 215]

# print(pic_arr[0][0])
# [215 124 141]

# temp_arr = pic_arr[:,:,::-1]

# print(pic_arr[0][0])
# # [215 124 141]
# print(temp_arr[0][0])
# # [141 124 215]

# print(temp_arr)

# print(pic_arr)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

path = '../open/'

image = cv2.imread(path + 'Lenna.png', cv2.IMREAD_UNCHANGED)

# print(type(image))

# print(image)

# cv2.imshow('IMG', image)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()

# plt.imshow(image)
# plt.show()

image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.imshow(image_temp)
# plt.show()

image_gray = cv2.imread(path + 'Lenna.png', cv2.IMREAD_GRAYSCALE)

# print(image_gray.shape)
# (440, 440)

# cv2.imshow('IMG',image_gray)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()

# plt.imshow(image_gray)
# plt.show()

# plt.imshow(image_gray, cmap='gray')
# plt.show()

# plt.imshow(image_gray, cmap='magma')
# plt.show()

random_image = np.random.randint(0, 256, size=(200,200,3))
# print(random_image.shape)

# (200, 200, 3)

# cv2.imwrite('./random_image.png', random_image)
# no_img = cv2.imread('./no_image.png')

# print(no_img)
# print(type(no_img))

my_img = cv2.imread('random_image.png')

# print(type(my_img))
# print(my_img.shape)

# cv2.imshow('iMG',my_img)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()


path = '../open/'

# image = cv2.imread(path + 'dog.jpg', cv2.IMREAD_UNCHANGED)

origin_img = cv2.imread(path + 'dog.jpg')
# print(origin_img.shape)

plt.imshow(origin_img)
plt.show()

img_rgb = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.show()

img_hsv = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)

# plt.imshow(img_hsv)
# plt.show()

# print(img_hsv)

# print(np.max(img_hsv),np.min(img_hsv))

# print(np.max(img_hsv[:,0,0]))
# print(np.min(img_hsv[:,0,0]))

img_hsl = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HLS)
plt.show()

# print(img_hsl)

# print(np.max(img_hsl), np.min(img_hsl))

# print(np.max(img_hsl[:.0,0]))
# print(np.min(img_hsl[:,0,0]))

img_ycrb=cv2.cvtColor(origin_img, cv2.COLOR_BGR2YCrCb)
# plt.imshow(img_ycrb)
# plt.show()

# print(img_ycrb)

# print(np.max(img_ycrb), np.min(img_ycrb))

# img_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
# print(img_gray.shape)

# plt.imshow(img_gray, cmap='gray')
# plt.show()

# print(img_gray)

# print(np.max(img_gray), np.min(img_gray))

img = np.zeros((512,512, 3), np.uint8)

# plt.imshow(img)
# plt.show()

# img = cv2.line(img, (0,0), (511,511), (255,0,0), 5)
# plt.imshow(img)
# plt.show()

# img = cv2.rectangle(img, (383, 0), (510,128), (0,255,0), 3)

# plt.imshow(img)
# plt.show()

# img=cv2.circle(img, (447, 63), 63,(0,0,255), -1)

# plt.imshow(img)
# plt.show()

# img = cv2.circle(img, (63,447), 63, (0,255,255),2)
# plt.imshow(img)
# plt.show()

img = cv2.ellipse(img, (255,255), (150,39), 0, 0, 180,(0,255,0), -1)

plt.imshow(img)
plt.show()

img = cv2.ellipse(img, (255,255), (150,50), 45,0,360, (255,255,255),2)

plt.imshow(img)
plt.show()

img = cv2.ellipse(img, (255,255), (150,10), 135,0,270, (0,0,255), 2)
plt.imshow(img)
plt.show()

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
print(pts.shape)

pts = pts.reshape((-1,2,1))
print(pts.shape)
img = cv2.polylines(img, [pts], True, (0,155,255), 5)

plt.imshow(img)
plt.show()

pts2 = np.array([[150,5],[200,30],[100,70],[50,20]],np.int32)
print(pts2.shape)

pts2 = pts2.reshape((-1,1,2))
print(pts2.shape)
img = cv2.polylines(img, [pts2], True, (172,200,255), 4)

plt.imshow(img)
plt.show()

img = cv2.putText(img,'OpenCV', (10,500), cv2.FONT_HERSHEY_COMPLEX,4,(255,255,255), 3)
plt.imshow(img)
plt.show()






