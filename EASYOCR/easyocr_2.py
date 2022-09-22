import easyocr
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

reader = easyocr.Reader(['ko', 'en'], gpu = True)
result = reader.readtext('C:/Users/User/Desktop/dawn/name_card.jpeg')
img    = cv2.imread('C:/Users/User/Desktop/dawn/name_card.jpeg')
img = Image.fromarray(img)
font = ImageFont.truetype('fonts/gulim.ttc', 50)
draw = ImageDraw.Draw(img)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(255, 3),dtype="uint8")
for i in result :
    x = i[0][0][0] 
    y = i[0][0][1] 
    w = i[0][1][0] - i[0][0][0] 
    h = i[0][2][1] - i[0][1][1]

    color_idx = random.randint(0,255) 
    color = [int(c) for c in COLORS[color_idx]]

    draw.rectangle(((x, y), (x+w, y+h)), outline=tuple(color), width=2)
    draw.text((int((x + x + w) / 2) , y-2),str(i[1]), font=font, fill=tuple(color),)

plt.figure(figsize=(50,50))
plt.imshow(img)
plt.show()


