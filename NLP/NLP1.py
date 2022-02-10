# ################################NLP1#########################################

# # s= 'No pain no gain'
# # 'pain' in s
# # print('pain' in s)

# # True

# # print(s.split())

# # ['No', 'pain', 'no', 'gain']

# # s.split().index('gain')
# # print(s.split().index('gain'))

# # 3

# # s[-4:]
# # print(s[-4:])
# # # gain

# # s.split()[1]
# # print(s.split()[1])
# # # pain

# # s.split()[2][::-1]
# # print(s.split()[2][::-1])
# # # on

# # s="한글도 처리 가능"
# # print('처리' in s)
# # True

# # s.split()
# # print(s.split())
# # # ['한글도', '처리', '가능']

# # s.split()[0]
# # print(s.split()[0])
# # # 한글도

# # s='AbCdEfGh'
# # str_lower = s.lower()
# # str_upper = s.upper()
# # print(str_lower, str_upper)
# # # abcdefgh ABCDEFGH

# # s = "I visited UK from Us on 22-09-20"
# # print(s)
# # # I visited UK from Us on 22-09-20

# # new_s = s.replace("UK", "United Kingdom").replace("US", "United States").replace("-20" , "-2020")
# # print(new_s)
# # # I visited United Kingdom from Us on 22-09-2020

# import re

# check = 'ab.'

# # print(re.match(check, 'abc'))
# # print(re.match(check, 'c'))
# # print(re.match(check, 'ab'))

# # <re.Match object; span=(0, 3), match='abc'>
# # None
# # None

# import time

# normal_s_time = time.time()
# r='ab.'

# for i in range(1000):
#     re.match(check, 'abc')
# print('일반 사용시 소요 시간 : ' , time.time()- normal_s_time)

# compile_s_time = time.time()

# r= re.compile('ab.')

# for i in range(1000):
#     r.match(check)
# print('컴파일 사용시 소요시간: ', time.time()-compile_s_time)

# # 일반 사용시 소요 시간 :  0.000997304916381836
# # 컴파일 사용시 소요시간:  0.0009691715240478516

# check = 'ab?'

# print(re.search('a', check))
# print(re.match('kkkab', check))
# print(re.search('kkkab', check))
# print(re.match('ab', check))

# # <re.Match object; span=(0, 1), match='a'>
# # None
# # None
# # <re.Match object; span=(0, 2), match='ab'>

# r=re.compile(' ')
# print(r.split('abc abbc abcbab'))

# # ['abc', 'abbc', 'abcbab']

# r=re.compile('c')
# print(r.split('abc abbc abcbab'))

# # ['ab', ' abb', ' ab', 'bab']

# r= re.compile('[1-9]')
# print(r.split('s1abc 2v3s 4sss 5a'))

# # ['s', 'abc ', 'v', 's ', 'sss ', 'a']

# print(re.sub('[a-z]', 'abcdefg', '1'))
# # 1
# print(re.sub('[^a-z]', 'abc defg', '1'))
# # abc defg

# # 1
# # abc defg

# print(re.findall('[\d]', '1ab 2cd 3ef 4g'))
# # ['1', '2', '3', '4']
# print(re.findall('[\W]', '!abcd@@#'))
# # ['!', '@', '@', '#']

# iter1 = re.finditer('[\d]', '1ab 2cd 3ef 4g')
# print(iter1)

# for i in iter1:
#     print(i)
# # <callable_iterator object at 0x000001F78E359F70>
# # <re.Match object; span=(0, 1), match='1'>
# # <re.Match object; span=(4, 5), match='2'>
# # <re.Match object; span=(8, 9), match='3'>
# # <re.Match object; span=(12, 13), match='4'>
# iter2 = re.finditer('[\W]', '!abcd@@#')
# print(iter2)
# for i in iter2:
#     print(i)

# # <callable_iterator object at 0x000001F78E359E50>
# # <re.Match object; span=(0, 1), match='!'>
# # <re.Match object; span=(5, 6), match='@'>
# # <re.Match object; span=(6, 7), match='@'>
# # <re.Match object; span=(7, 8), match='#'>

# sentence = 'Time is gold'
# tokens = [x for x in sentence.split(' ')]
# print(tokens)
# # ['Time', 'is', 'gold']

# import nltk 
# # nltk.download('punkt')

# from nltk.tokenize import word_tokenize

# tokens = word_tokenize(sentence)
# print(tokens)
# # ['Time', 'is', 'gold']

# sentences = 'The world is a beatiful book.\nBut of little use to him who cannot read it.'
# print(sentences)
# # The world is a beatiful book.
# # But of little use to him who cannot read it.
# tokens = [x for x in sentences.split('\n')]
# print(tokens)

# ['The world is a beatiful book.', 'But of little use to him who cannot read it.']

# from nltk.tokenize import sent_tokenize

# tokens = sent_tokenize(sentences)
# ['The world is a beatiful book.', 'But of little use to him who cannot read it.']
# print(tokens)

# from nltk.tokenize import RegexpTokenizer

# sentence = 'Where there\'s a will, there\'s a way'

# tokenizer = RegexpTokenizer("[\w]+")
# tokens = tokenizer.tokenize(sentence)
# print(tokens)
# # ['Where', 'there', 's', 'a', 'will', 'there', 's', 'a', 'way']

# tokenizer = RegexpTokenizer("[\s]+", gaps=True)
# tokens = tokenizer.tokenize(sentence)
# print(tokens)
# # ['Where', "there's", 'a', 'will,', "there's", 'a', 'way']

# from keras.preprocessing.text import text_to_word_sequence
# sentence = 'Where there\'s a will, there\'s a way'
# text_to_word_sequence(sentence)

# # ['Where', "there's", 'a', 'will,', "there's", 'a', 'way']

# from textblob import TextBlob
# sentence = 'Where there\'s a will, there\'s a way'

# blob = TextBlob(sentence)
# print(blob.words)
# # ['Where', 'there', "'s", 'a', 'will', 'there', "'s", 'a', 'way']

# from nltk import ngrams

# sentence = 'There is no royal road to learning'
# bigram = list(ngrams(sentence.split(),2))
# print(bigram)

# # [('There', 'is'), ('is', 'no'), ('no', 'royal'), ('royal', 'road'), ('road', 'to'), ('to', 'learning')]

# trigram = list(ngrams(sentence.split(), 3))
# print(trigram)

# # [('There', 'is', 'no'), ('is', 'no', 'royal'), ('no', 'royal', 'road'), ('royal', 'road', 'to'), ('road', 'to', 'learning')]

# from textblob import TextBlob

# blob = TextBlob(sentence)
# blob.ngrams(n=2)
# print(blob.ngrams(n=2))
# # [WordList(['There', 'is']), WordList(['is', 'no']), WordList(['no', 'royal']), WordList(['royal', 'road']), WordList(['road', 'to']), WordList(['to', 'learning'])]
# print(blob.ngrams(n=3))
# # [WordList(['There', 'is', 'no']), WordList(['is', 'no', 'royal']), WordList(['no', 'royal', 'road']), WordList(['royal', 'road', 'to']), WordList(['road', 'to', 'learning'])]

# import nltk
# nltk.download('punkt')

# from nltk import word_tokenize

# words = word_tokenize("Think like man of action and act like man of thought.")
# print(words)
# # ['Think', 'like', 'man', 'of', 'action', 'and', 'act', 'like', 'man', 'of', 'thought', '.']

# nltk.download('averaged_perceptron_tagger')

# nltk.pos_tag(words)

# print(nltk.pos_tag(words))
# # [('Think', 'VBP'), ('like', 'IN'), ('man', 'NN'), ('of', 'IN'), ('action', 'NN'), ('and', 'CC'), ('act', 'NN'), ('like', 'IN'), ('man', 'NN'), ('of', 'IN'), ('thought', 'NN'), ('.', '.')]

# nltk.pos_tag(word_tokenize("A rolling stone gathers no moss"))
# print(nltk.pos_tag(word_tokenize("A rolling stone gathers no moss")))
# # [('A', 'DT'), ('rolling', 'VBG'), ('stone', 'NN'), ('gathers', 'NNS'), ('no', 'DT'), ('moss', 'NN')]

# stop_words = "on in the"
# stop_words = stop_words.split(' ')
# print(stop_words)
# # ['on', 'in', 'the']

# sentence = 'singer on the stage'
# sentence = sentence.split(' ')
# nouns = []
# for noun in sentence :
#     if noun not in stop_words:
#         nouns.append(noun)
        
# print(nouns)
# # ['singer', 'stage']

# import nltk
# nltk.download('stopwords')

# from nltk import word_tokenize
# from nltk.corpus import stopwords 

# stop_words = stopwords.words('english')
# print(stop_words)

# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
#  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
#  'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
# 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
# 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
# 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
# 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
# "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
# "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
# "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# s = "If you do not walk today, you will have to run tomorrow."
# words = word_tokenize(s)
# print(words)

# # ['If', 'you', 'do', 'not', 'walk', 'today', ',', 'you', 'will', 'have', 'to', 'run', 'tomorrow', '.']

# no_stopwords = []
# for w in words:
#     if w not in stop_words : 
#         no_stopwords.append(w)
        
# print(no_stopwords)
# # ['If', 'walk', 'today', ',', 'run', 'tomorrow', '.']

# from autocorrect import Speller

# spell = Speller('en')

# print(spell('peoplle'))
# print(spell('peope'))
# print(spell('peopae'))

# # people
# # people
# # people

# s = word_tokenize("Earlly biird catchess the womm.")

# print(s)

# ss=' '.join([spell(s) for s in s])
# print(ss)

# # ['Earlly', 'biird', 'catchess', 'the', 'womm', '.']
# # Early bird catches the worm .

# from textblob import TextBlob
# words = 'apples bananas oranges'
# tb=TextBlob(words)

# print(tb.words)
# print(tb.words.singularize())

# # ['apples', 'bananas', 'oranges']
# # ['apple', 'banana', 'orange']

# words = 'car train airplane'
# tb = TextBlob(words)

# print(tb.words)
# print(tb.words.pluralize())

# # ['car', 'train', 'airplane']
# # ['cars', 'trains', 'airplanes']

# import nltk

# stemmer = nltk.stem.PorterStemmer()

# print(stemmer.stem('application'))
# # applic

# print(stemmer.stem('beginning'))
# # begin

# print(stemmer.stem('catches'))
# # catch

# print(stemmer.stem('education'))
# # educ

# import nltk
# nltk.download('wordnet')
# from nltk.stem.wordnet import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

# print(lemmatizer.lemmatize('application'))
# # application

# print(lemmatizer.lemmatize('beginning'))
# # beginning

# print(lemmatizer.lemmatize('catches'))
# # catch

# print(lemmatizer.lemmatize('education'))
# # education

# import nltk
# from nltk import word_tokenize
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# s="Rome was not built in a day."
# print(s)

# tags = nltk.pos_tag(word_tokenize(s))
# print(tags)
# # [('Rome', 'NNP'), ('was', 'VBD'), ('not', 'RB'), ('built', 'VBN'), ('in', 'IN'), ('a', 'DT'), ('day', 'NN'), ('.', '.')]

# entities = nltk.ne_chunk(tags, binary=True)
# print(entities)

# # (S (NE Rome/NNP) was/VBD not/RB built/VBN in/IN a/DT day/NN ./.)

# import nltk
# from nltk.wsd import lesk

# s="I saw bats."

# # print(word_tokenize(s))
# # print(lesk(word_tokenize(s), 'saw'))
# # print(lesk(word_tokenize(s), 'bats'))

# # ['I', 'saw', 'bats', '.']
# # Synset('saw.v.01')
# # Synset('squash_racket.n.01')

# import re

# # check = '[ㄱ-ㅎ]+'

# # print(re.match(check, 'ㅎ 안녕하세요.'))
# # print(re.match(check, '안녕하세요. ㅎ'))

# # <re.Match object; span=(0, 1), match='ㅎ'>
# # None

# #check = '[ㄱ-ㅎ|ㅏ-ㅣ]+'

# # print(re.search(check, 'ㄱㅏ 안녕하세요'))
# # print(re.match(check, '안 ㄱㅏ'))
# # print(re.search(check, '안 ㄱㅏ')
      
# # <re.Match object; span=(0, 2), match='ㄱㅏ'>
# # None
# # <re.Match object; span=(2, 4), match='ㄱㅏ'>

# # print(re.sub('[가-힣]', '가나다라마바사', '1'))
# # print(re.sub('[^가-힣]', '가나다라마바사', '1'))

# # 1
# # 가나다라마바사

# from konlpy.tag import Mecab
# tagger = Mecab('C:\mecab\mecab-ko-dic')

# sentence = '언제나 현재에 집중할 수 있다면 행복할것이다.'
# print(tagger.pos(sentence))

# print(tagger.morphs(sentence))
# ['언제나', '현재', '에', '집중', '할', '수', '있', '다면', '행복', '할', '것', '이', '다', '.']

# print(tagger.nouns(sentence))
# ['현재', '집중', '수', '행복', '것']

# import kss

# text = '진짜? 내일 뭐하지. 이렇게 애매모호한 문장도? 밥은 먹었어? 나는...'

# print(kss.split_sentences(text))

# from nltk.tokenize import RegexpTokenizer
# from textblob import Sentence

# sentence = '안녕하세요 ㅋㅋ 저는 자연어 처리(Natural Language Processing)를ㄹ!! 배우고 있습니다.'

# tokenizer = RegexpTokenizer("[가-힣]+")
# tokens = tokenizer.tokenize(sentence)
# print(tokens)

# tokenizer = RegexpTokenizer("[ㄱ-ㅎ]+", gaps=True)
# tokens = tokenizer.tokenize(sentence)
# print(tokens)

# from keras.preprocessing.text import text_to_word_sequence

# sentence = '성공의 비결은 단 한 가지, 잘할 수 있는 일에 광적으로 집중하는 것이다.'

# # text_to_word_sequence(sentence)

# from textblob import TextBlob

# blob = TextBlob(sentence)
# print(blob.words)
# ['성공의', '비결은', '단', '한', '가지', '잘할', '수', '있는', '일에', '광적으로', '집중하는', '것이다']

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Think like a man of action and act like man of thought.']

vector = CountVectorizer()
bow = vector.fit_transform(corpus)

# print(bow.toarray())
# print(vector.vocabulary_)

# [[1 1 1 2 2 2 1 1]]
# {'think': 6, 'like': 3, 'man': 4, 'of': 5, 'action': 1, 'and': 2, 'act': 0, 'thought': 7}

vector = CountVectorizer(stop_words='english')
bow = vector.fit_transform(corpus)

print(bow.toarray())
print(vector.vocabulary_)

# [[1 1 2 2 1 1]]
# {'think': 4, 'like': 2, 'man': 3, 'action': 1, 'act': 0, 'thought': 5}

# corpus = ["평생 살 것처럼 꿈을 꾸어라. 그리고 내일 죽을 것처럼 오늘을 살아라."]

# vector = CountVectorizer()
# bow = vector.fit_transform(corpus)

# print(bow.toarray())
# print(vector.vocabulary_)
# [[2 1 1 1 1 1 1 1 1]]
# {'평생': 8, '것처럼': 0, '꿈을': 3, '꾸어라': 2, '그리고': 1, '내일': 4, '죽을': 7, '오늘을': 6, '살아라': 5}

import re 
from konlpy.tag import Mecab
tagger = Mecab('C:\mecab\mecab-ko-dic')

corpus = "평생 살 것처럼 꿈을 꾸어라. 그리고 내일 죽을 것처럼 오늘을 살아라."
tokens = tagger.morphs(re.sub("(\.)","",corpus))

vocab={}
bow=[]

for tok in tokens:
  if tok not in vocab.keys():
    vocab[tok]=len(vocab)
    bow.insert(len(vocab)-1,1)
  else:
    index = vocab.get(tok)
    bow[index]=bow[index]+1

print(bow)
print(vocab)        
    
# [1, 2, 2, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1]
# {'평생': 0, '살': 1, '것': 2, '처럼': 3, '꿈': 4, '을': 5, '꾸': 6, '어라': 7, '그리고': 8, '내일': 9, '죽': 10, '오늘': 11, '아라': 12}

from sklearn.feature_extraction.text import CountVectorizer

corpus = ["Think like a man of action and act like man of thought.", "Try not to become a man of success but rather try to become a man of value.", "Give me liberty, of give me death"]

vector = CountVectorizer(stop_words='english')
bow = vector.fit_transform(corpus)

print(bow.toarray())
print(vector.vocabulary_)

# [[1 1 0 0 2 2 0 1 1 0 0]
#  [0 0 0 0 0 2 1 0 0 2 1]
#  [0 0 1 1 0 0 0 0 0 0 0]]
# {'think': 7, 'like': 4, 'man': 5, 'action': 1, 'act': 0, 'thought': 8, 'try': 9, 'success': 6, 'value': 10, 'liberty': 3, 'death': 2}
import pandas as pd

columns = []
for k, v in sorted(vector.vocabulary_.items(), key=lambda item:item[1]):
    columns.append(k)
    
df = pd.DataFrame(bow.toarray(), columns=columns)
print(df)
#    act  action  death  liberty  like  man  success  think  thought  try  value
# 0    1       1      0        0     2    2        0      1        1    0      0
# 1    0       0      0        0     0    2        1      0        0    2      1
# 2    0       0      1        1     0    0        0      0        0    0      0

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english').fit(corpus)

print(tfidf.transform(corpus).toarray())
print(tfidf.vocabulary_)

# [[0.311383   0.311383   0.         0.         0.62276601 0.4736296
#   0.         0.311383   0.311383   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.52753275
#   0.34682109 0.         0.         0.69364217 0.34682109]
#  [0.         0.         0.70710678 0.70710678 0.         0.
#   0.         0.         0.         0.         0.        ]]
# {'think': 7, 'like': 4, 'man': 5, 'action': 1, 'act': 0, 'thought': 8, 'try': 9, 'success': 6, 'value': 10, 'liberty': 3, 'death': 2}

columns = []
for k, v in sorted(tfidf.vocabulary_.items(), key=lambda item:item[1]):
    columns.append(k)
    
df = pd.DataFrame(tfidf.transform(corpus).toarray(), columns=columns)

print(df)