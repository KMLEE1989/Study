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
import gc
gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

print(f"-os:{platform.platform()}") #-os:Windows-10-10.0.19043-SP0
print(f"-python:{sys.version}") #-python:3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
print(f"-pandas:{pd.__version__}") #-pandas:1.3.5
print(f"-numpy:{np.__version__}") #-numpy:1.20.3
print(f"-sklearn: {sklearn.__version__}") #-sklearn: 1.0.2

path= "D:/Study/Hackarthon/dacon/Anomaly Detection/open/"

device = torch.device('cuda')

train_png = sorted(glob('train/*.png'))
test_png = sorted(glob('test/*.png'))

train_y = pd.read_csv(path + "train_df.csv")

train_labels = train_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img

train_imgs = [img_load(m) for m in tqdm(train_png)]
test_imgs = [img_load(n) for n in tqdm(test_png)]

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode=='train':
            augmentation = random.randint(0,2)
            if augmentation==1:
                img = img[::-1].copy()
            elif augmentation==2:
                img = img[:,::-1].copy()
        img = transforms.ToTensor()(img)
        if self.mode=='test':
            pass
        
        label = self.labels[idx]
        return img, label
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
batch_size = 32
epochs = 25

# Train
train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Test
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

model = Network().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() 



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

with torch.no_grad():
    for batch in (test_loader):
        x = torch.tensor(batch[0], dtype = torch.float32, device = device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())
        

label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in f_pred]

submission = pd.read_csv(path + "sample_submission.csv")

submission["label"] = f_result

submission

submission.to_csv("sookook1.csv", index = False)
