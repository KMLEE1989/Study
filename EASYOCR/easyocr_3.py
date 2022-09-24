import os
import easyocr
from PIL import Image, ImageDraw, ImageFont

reader = easyocr.Reader(['en', 'ko'], gpu=True)
DATAPATH = "C:/Users/User/Desktop/raw"

font_size = 30
font = ImageFont.truetype("fonts/gulim.ttc", font_size, encoding="UTF-8")

for image in os.listdir(DATAPATH):
	image_path = os.path.join(DATAPATH, image)
	if os.path.isfile(image_path):
		result = reader.readtext(image_path)
		img = Image.open(image_path)
		for r in result:
			x, y = min(list(_[0] for _ in r[0])), min(list(_[1] for _ in r[0]))
			text_data = r[1]
			d = ImageDraw.Draw(img)
			d.text((x-10,y-10), text_data, font=font, fill=(255,0,0))
		img.save(image_path+"-result.png")
