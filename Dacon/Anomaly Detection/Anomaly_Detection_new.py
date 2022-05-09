import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time

from cgi import test
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM
import os
import sys
import platform
import random
import math
from typing import List, Dict,Tuple 
import sklearn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
from catboost import Pool, CatBoostClassifier
from xgboost import XGBClassifier
from tensorflow.keras.applications import VGG16, VGG19, ResNet101, DenseNet121, MobileNet, InceptionResNetV2
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time

from sklearn.model_selection import StratifiedKFold
device = torch.device('cuda')


warnings.filterwarnings('ignore')

print(f"-os:{platform.platform()}") #os:Windows-10-10.0.19043-SP0
print(f"-python:{sys.version}") #python:3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
print(f"-pandas:{pd.__version__}") #pandas:1.3.5
print(f"-numpy:{np.__version__}") #numpy:1.20.3
print(f"-sklearn: {sklearn.__version__}") #sklearn: 1.0.2

path = "D:/Study/Hackarthon/dacon/Anomaly_Detection/open/open/"

train_png = sorted(glob(path + 'train/train/*.png'))
test_png = sorted(glob(path + 'test/test/*.png'))

print(len(train_png), len(test_png))

# 4277 2154


train_y = pd.read_csv(path +"train_df.csv")

train_labels = train_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (384, 384),interpolation = cv2.INTER_AREA)
    return img

train_imgs = [img_load(m) for m in tqdm(train_png)]
test_imgs = [img_load(n) for n in tqdm(test_png)]

np.save(path + 'train_imgs_384', np.array(train_imgs))
np.save(path + 'test_imgs_384', np.array(test_imgs))

train_imgs = np.load(path + 'train_imgs_384.npy')
test_imgs = np.load(path + 'test_imgs_384.npy')

meanRGB = [np.mean(x, axis=(0,1)) for x in train_imgs]
stdRGB = [np.std(x, axis=(0,1)) for x in train_imgs]

meanR = np.mean([m[0] for m in meanRGB])/255
meanG = np.mean([m[1] for m in meanRGB])/255
meanB = np.mean([m[2] for m in meanRGB])/255

stdR = np.mean([s[0] for s in stdRGB])/255
stdG = np.mean([s[1] for s in stdRGB])/255
stdB = np.mean([s[2] for s in stdRGB])/255

print("train 평균",meanR, meanG, meanB)
print("train 표준편차",stdR, stdG, stdB)

meanRGB = [np.mean(x, axis=(0,1)) for x in test_imgs]
stdRGB = [np.std(x, axis=(0,1)) for x in test_imgs]

meanR = np.mean([m[0] for m in meanRGB])/255
meanG = np.mean([m[1] for m in meanRGB])/255
meanB = np.mean([m[2] for m in meanRGB])/255

stdR = np.mean([s[0] for s in stdRGB])/255
stdG = np.mean([s[1] for s in stdRGB])/255
stdB = np.mean([s[2] for s in stdRGB])/255

print("test 평균",meanR, meanG, meanB)
print("test 표준편차",stdR, stdG, stdB)

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode == 'train':
          train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.433038, 0.403458, 0.394151],
                                     std = [0.181572, 0.174035, 0.163234]),
                transforms.RandomAffine((-45, 45)),
                
            ])
          img = train_transform(img)
        if self.mode == 'test':
          test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.418256, 0.393101, 0.386632],
                                     std = [0.195055, 0.190053, 0.185323])
            ])
          img = test_transform(img)

        
        label = self.labels[idx]
        return img, label
    
class Network(nn.Module):
    def __init__(self,mode = 'train'):
        super(Network, self).__init__()
        self.mode = mode
        if self.mode == 'train':
          self.model = timm.create_model('resnet101', pretrained=True, num_classes=88, drop_path_rate = 0.2)
        if self.mode == 'test':
          self.model = timm.create_model('resnet101', pretrained=True, num_classes=88, drop_path_rate = 0)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

def main(seed = 2022):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
main(2022)

import gc

cv = StratifiedKFold(n_splits = 5, random_state = 2022,shuffle=True)
batch_size = 8
epochs = 70
pred_ensemble = []


for idx, (train_idx, val_idx) in enumerate(cv.split(train_imgs, np.array(train_labels))):
  print("----------fold_{} start!----------".format(idx))
  t_imgs, val_imgs = train_imgs[train_idx],  train_imgs[val_idx]
  t_labels, val_labels = np.array(train_labels)[train_idx], np.array(train_labels)[val_idx]

  # Train
  train_dataset = Custom_dataset(np.array(t_imgs), np.array(t_labels), mode='train')
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

  # Val
  val_dataset = Custom_dataset(np.array(val_imgs), np.array(val_labels), mode='test')
  val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

  gc.collect()
  torch.cuda.empty_cache()
  best=0

  model = Network().to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay = 2e-2)
  criterion = nn.CrossEntropyLoss()
  scaler = torch.cuda.amp.GradScaler()  

  best_f1 = 0
  early_stopping = 0
  for epoch in range(epochs):
    start=time.time()
    train_loss = 0
    train_pred=[]
    train_y=[]
    model.train()
    for batch in (train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()/len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()
    train_f1 = score_function(train_y, train_pred)
    state_dict= model.state_dict()
    model.eval()
    with torch.no_grad():
      val_loss = 0 
      val_pred = []
      val_y = []
      

      for batch in (val_loader):
        x_val = torch.tensor(batch[0], dtype = torch.float32, device = device)
        y_val = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred_val = model(x_val)
        loss_val = criterion(pred_val, y_val)

        val_loss += loss_val.item()/len(val_loader)
        val_pred += pred_val.argmax(1).detach().cpu().numpy().tolist()
        val_y += y_val.detach().cpu().numpy().tolist()
      val_f1 = score_function(val_y, val_pred)

      if val_f1 > best_f1:
        best_epoch = epoch
        best_loss = val_loss
        best_f1 = val_f1
        early_stopping = 0

        torch.save({'epoch':epoch,
                    'state_dict':state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
             }, path +'best_model_{}.pth'.format(idx))
        print('-----------------SAVE:{} epoch----------------'.format(best_epoch+1))
      else:
          early_stopping += 1

            # Early Stopping
      if early_stopping == 20:
        TIME = time.time() - start
        print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
        print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
        print(f'Val    loss : {val_loss:.5f}    f1 : {val_f1:.5f}')
        break

    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
    print(f'Val    loss : {val_loss:.5f}    f1 : {val_f1:.5f}')
    
pred_ensemble = []
batch_size = 32
# Test
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

for i in range(5):
  model_test = Network(mode = 'test').to(device)
  model_test.load_state_dict(torch.load((path+'best_model_{}.pth'.format(i)))['state_dict'])
  model_test.eval()
  pred_prob = []
  with torch.no_grad():
      for batch in (test_loader):
          x = torch.tensor(batch[0], dtype = torch.float32, device = device)
          with torch.cuda.amp.autocast():
              pred = model_test(x)
              pred_prob.extend(pred.detach().cpu().numpy())
      pred_ensemble.append(pred_prob)
      
      
print(len(pred_ensemble))

pred = (np.array(pred_ensemble[0])+ np.array(pred_ensemble[1])+ np.array(pred_ensemble[3]) + np.array(pred_ensemble[4]) )/4
f_pred = np.array(pred).argmax(1).tolist()

label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in f_pred]

submission = pd.read_csv(path + "sample_submission.csv")

submission["label"] = f_result

submission

submission.to_csv(path + "kookhwa77", index = False)

batch_size = 32
epochs = 30

# Train
train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Test
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

import gc
gc.collect()
torch.cuda.empty_cache()

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

model = Network().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay = 1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() 

batch_size = 32
epochs = 30

best=0
for epoch in range(epochs):
    start=time.time()
    train_loss = 0
    train_pred=[]
    train_y=[]
    model.train()
    for batch in (train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()/len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()
        
    
    train_f1 = score_function(train_y, train_pred)

    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
    
model.eval()
f_pred = []
pred_prob = []

with torch.no_grad():
    for batch in (test_loader):
        x = torch.tensor(batch[0], dtype = torch.float32, device = device)
        with torch.cuda.amp.autocast():
            pred = model(x)
            pred_prob.extend(pred.detach().cpu().numpy())
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())
        
        
label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in f_pred]

submission = pd.read_csv(path + "sample_submission.csv")

submission["label"] = f_result

submission

submission.to_csv(path + "somal1.csv", index = False)