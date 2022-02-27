import matplotlib
import pandas as pd
import numpy as np

import platform
import matplotlib.pyplot as plt

import matplotlib_inline

path="c:/Windows/Fonts/malgun.ttf"

from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows' :
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
    
else:
    print('Unknown ststem.... sorry about the issue')
    
#plt.rcParams['axes.unicode_minus'] = False

from bs4 import BeautifulSoup 
from urllib.request import urlopen
import urllib
import time

tmp1 = 'https://search.naver.com/search.naver?where=kin'
html = tmp1+ '&sm=tab_jum&ie=utf8&query={key_word}&start={num}'

response = urlopen(html.format(num=1, key_word=urllib.parse.quote('여친 선물')))

soup = BeautifulSoup(response, 'html.parser')
tmp = soup.find_all('dl')

tmp_list = []
for line in tmp:
    tmp_list.append(line.text)
tmp_list

