from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = '../_data/cat_dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print("=================================imge.img_to_array(img)=========================================")
print(x, '\n', x.shape) #  (224, 224, 3)

x = np.expand_dims(x, axis=0)
print("=================================np.expand_dims(x,axis=0)=========================================")
print(x, '\n', x.shape) #  (1, 224, 224, 3)

x = preprocess_input(x)
print("=================================np.expand_dims(x,axis=0)=========================================")
print(x, '\n', x.shape) #  (1, 224, 224, 3)

preds = model.predict(x)

print(preds, '\n', preds.shape)  # (1, 1000)

print('결과는 : ', decode_predictions(preds, top=10)[0])

# [('n04239074', 'sliding_door', 0.21902743), ('n02123045', 'tabby', 0.13198864), 
#  ('n02808304', 'bath_towel', 0.045052547), ('n04589890', 'window_screen', 0.042799547), ('n04493381', 'tub', 0.04112369), 
#  ('n02123159', 'tiger_cat', 0.03896884), ('n02099601', 'golden_retriever', 0.027294837), ('n03223299', 'doormat', 0.02722764),
#  ('n02124075', 'Egyptian_cat', 0.021267276), ('n03887697', 'paper_towel', 0.021090858)]



