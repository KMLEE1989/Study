import random, os
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import Compose, OneOf, Resize, Normalize
from albumentations.pytorch import ToTensor

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device = {DEVICE}')

def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Seed set = {seed}')
    
seed_everything()


