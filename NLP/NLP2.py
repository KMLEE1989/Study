import urllib.request

import matplotlib

raw = urllib.request.urlopen('https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt').readlines()
# print(raw[:5])

raw=[x.decode() for x in raw[1:]]

reviews = []
for i in raw:
    reviews.append(i.split('\t')[1])
    
# print(reviews[:5])

# [b'id\tdocument\tlabel\n', b'8112052\t\xec\x96\xb4\xeb\xa6\xb4\xeb\x95\x8c\xeb\xb3\xb4\xea\xb3\xa0 \xec\xa7\x80\xea\xb8\x88\xeb\x8b\xa4\xec\x8b\x9c\xeb\xb4\x90\xeb\x8f\x84 \xec\x9e\xac\xeb\xb0\x8c\xec\x96\xb4\xec\x9a\x94\xe3\x85\x8b\xe3\x85\x8b\t1\n', b'8132799\t\xeb\x94\x94\xec\x9e\x90\xec\x9d\xb8\xec\x9d\x84 \xeb\xb0\xb0\xec\x9a\xb0\xeb\x8a\x94 \xed\x95\x99\xec\x83\x9d\xec\x9c\xbc\xeb\xa1\x9c, \xec\x99\xb8\xea\xb5\xad\xeb\x94\x94\xec\x9e\x90\xec\x9d\xb4\xeb\x84\x88\xec\x99\x80 \xea\xb7\xb8\xeb\x93\xa4\xec\x9d\xb4 \xec\x9d\xbc\xea\xb5\xb0 \xec\xa0\x84\xed\x86\xb5\xec\x9d\x84 \xed\x86\xb5\xed\x95\xb4 \xeb\xb0\x9c\xec\xa0\x84\xed\x95\xb4\xea\xb0\x80\xeb\x8a\x94 \xeb\xac\xb8\xed\x99\x94\xec\x82\xb0\xec\x97\x85\xec\x9d\xb4 \xeb\xb6\x80\xeb\x9f\xac\xec\x9b\xa0\xeb\x8a\x94\xeb\x8d\xb0. \xec\x82\xac\xec\x8b\xa4 \xec\x9a\xb0\xeb\xa6\xac\xeb\x82\x98\xeb\x9d\xbc\xec\x97\x90\xec\x84\x9c\xeb\x8f\x84 \xea\xb7\xb8 \xec\x96\xb4\xeb\xa0\xa4\xec\x9a\xb4\xec\x8b\x9c\xec\xa0\x88\xec\x97\x90 \xeb\x81\x9d\xea\xb9\x8c\xec\xa7\x80 \xec\x97\xb4\xec\xa0\x95\xec\x9d\x84 \xec\xa7\x80\xed\x82\xa8 \xeb\x85\xb8\xeb\x9d\xbc\xeb\x85\xb8 \xea\xb0\x99\xec\x9d\x80 \xec\xa0\x84\xed\x86\xb5\xec\x9d\xb4\xec\x9e\x88\xec\x96\xb4 \xec\xa0\x80\xec\x99\x80 \xea\xb0\x99\xec\x9d\x80 \xec\x82\xac\xeb\x9e\x8c\xeb\x93\xa4\xec\x9d\xb4 \xea\xbf\x88\xec\x9d\x84 \xea\xbe\xb8\xea\xb3\xa0 \xec\x9d\xb4\xeb\xa4\x84\xeb\x82\x98\xea\xb0\x88 \xec\x88\x98 \xec\x9e\x88\xeb\x8b\xa4\xeb\x8a\x94 \xea\xb2\x83\xec\x97\x90 \xea\xb0\x90\xec\x82\xac\xed\x95\xa9\xeb\x8b\x88\xeb\x8b\xa4.\t1\n', b'4655635\t\xed\x8f\xb4\xeb\xa6\xac\xec\x8a\xa4\xec\x8a\xa4\xed\x86\xa0\xeb\xa6\xac \xec\x8b\x9c\xeb\xa6\xac\xec\xa6\x88\xeb\x8a\x94 1\xeb\xb6\x80\xed\x84\xb0 \xeb\x89\xb4\xea\xb9\x8c\xec\xa7\x80 \xeb\xb2\x84\xeb\xa6\xb4\xea\xbb\x98 \xed\x95\x98\xeb\x82\x98\xeb\x8f\x84 \xec\x97\x86\xec\x9d\x8c.. \xec\xb5\x9c\xea\xb3\xa0.\t1\n', b'9251303\t\xec\x99\x80.. \xec\x97\xb0\xea\xb8\xb0\xea\xb0\x80 \xec\xa7\x84\xec\xa7\x9c \xea\xb0\x9c\xec\xa9\x94\xea\xb5\xac\xeb\x82\x98.. \xec\xa7\x80\xeb\xa3\xa8\xed\x95\xa0\xea\xb1\xb0\xeb\x9d\xbc\xea\xb3\xa0 \xec\x83\x9d\xea\xb0\x81\xed\x96\x88\xeb\x8a\x94\xeb\x8d\xb0 \xeb\xaa\xb0\xec\x9e\x85\xed\x95\xb4\xec\x84\x9c \xeb\xb4\xa4\xeb\x8b\xa4.. \xea\xb7\xb8\xeb\x9e\x98 \xec\x9d\xb4\xeb\x9f\xb0\xea\xb2\x8c \xec\xa7\x84\xec\xa7\x9c \xec\x98\x81\xed\x99\x94\xec\xa7\x80\t1\n']
# ['어릴때보고 지금다시봐도 재밌어요ㅋㅋ', '디자인을 배우는 학생으로, 외국디자이너와 그들이 일군 전통을 통해 발전해가는 문화산업이 부러웠는데. 사실 우리나라에서도 그 어려운시절에 끝까지 열정을 지킨 노라
# 노 같은 전통이있어 저와 같은 사람들이 꿈을 꾸고 이뤄나갈 수 있다는 것에 감사합니다.', '폴리스스토리 시리즈는 1부터 뉴까지 버릴께 하나도 없음.. 최고.', '와.. 연기가 진짜 개쩔구나.. 지루할거라고 생각했 
# 는데 몰입해서 봤다.. 그래 이런게 진짜 영화지', '안개 자욱한 밤하늘에 떠 있는 초승달 같은 영화.']

from konlpy.tag import Mecab
tagger = Mecab('C:\mecab\mecab-ko-dic')

nouns = []
for review in reviews:
    for noun in tagger.nouns(review):
        nouns.append(noun)
        
# print(nouns[:10])

# ['때', '디자인', '학생', '외국', '디자이너', '그', '전통', '발전', '문화', '산업']

stop_words = "영화 전 난 일 걸 뭐 줄 만 건 분 개 끝 잼 이거 번 중 듯 때 게 내 말 나 수 거 점 것"
stop_words = stop_words.split(' ')
# print(stop_words)

# ['영화', '전', '난', '일', '걸', '뭐', '줄', '만', '건', '분', '개', '끝', '잼', '이거', '번', '중', '듯', '때', '게', '내', '말', '나', '수', '거', '점', '것']

nouns = []
for review in reviews:
    for noun in tagger.nouns(review):
        if noun not in stop_words:
            nouns.append(noun)
            
# print(nouns[:10])

# ['디자인', '학생', '외국', '디자이너', '그', '전통', '발전', '문화', '산업', '우리']

from collections import Counter

nouns_counter = Counter(nouns)
top_nouns = dict(nouns_counter.most_common(50))
# print(top_nouns)

# {'연기': 9175, '최고': 8813, '평점': 8514, '스토리': 7163, '생각': 6943, '드라마': 6896, '사람': 6742, '감동': 6489, '배우': 5893, '내용': 5731, '감독': 5629, '재미': 5479, '시간': 5320, '년': 4936, '사랑': 4741, '쓰레기': 4585, '작품': 3985, '하나': 3923, '정도': 3656, '이건': 3650, '마지막': 3647, '액션': 3568, '기대': 3465, '장면': 3262, '이게': 3046, '편': 3044, '최악': 3019, '돈': 2980, '이야
# 기': 2947, '이해': 2745, '애': 2730, '명작': 2685, '여자': 2678, '이상': 2676, '처음': 2673, '한국': 2640, '주인공': 2553, '우리': 2531, '연출': 2376, '때문': 2371, '기억': 2364, '현실': 2193, '마음': 2128, '굿': 2110, '남자': 2078, '결말': 2066, '인생': 2060, '공포': 2048, '전개': 2035, '짜증': 2011}

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.font_manager as fm

# font_path = "C:/Windows/Fonts/HMFMMUEX.TTC"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
# fontprop=fm.FontProperties(fname=font_path, size=5)

# y_pos = np.arange(len(top_nouns))

# plt.figure(figsize=(12,12))
# plt.barh(y_pos, top_nouns.values())
# plt.title('Word Count', font=fontprop)
# plt.yticks(y_pos, top_nouns.keys(), font=fontprop)
# plt.show()

# findfont: Font family ['MoeumT R'] not found. Falling back to DejaVu Sans.

from wordcloud import WordCloud

# wc = WordCloud(background_color='white', font_path="C:/Windows/Fonts/HMFMMUEX.TTC")
# print(wc.generate_from_frequencies(top_nouns))

# <wordcloud.wordcloud.WordCloud object at 0x0000027804B17CF8>

# figure =plt.figure(figsize=(12,12))
# ax = figure.add_subplot(1,1,1)
# ax.axis('off')
# ax.imshow(wc)
# plt.show()

import squarify
import matplotlib as mpl
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/HMFMMUEX.TTC"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
fontprop=fm.FontProperties(fname=font_path, size=5)

norm = mpl.colors.Normalize(vmin=min(top_nouns.values()),
                            vmax=max(top_nouns.values()))

colors = [mpl.cm.Blues(norm(value)) for value in top_nouns.values()]

squarify.plot(label=top_nouns.keys(), sizes=top_nouns.values(),color=colors,alpha=.7);
# rc('font', font=fontprop)
plt.show()

                    