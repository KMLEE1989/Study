import nltk
nltk.download('punkt')

from nltk import word_tokenize, bigrams

sentence = 'I love data science and deep learning'
tokens = word_tokenize(sentence)

bgram = bigrams(tokens)
bgram_list = [x for x in bgram]
# print(bgram_list)
# [('I', 'love'), ('love', 'data'), ('data', 'science'), ('science', 'and'), ('and', 'deep'), ('deep', 'learning')]

from nltk.util import ngrams

tgram = ngrams(tokens, 3)
qgram = ngrams(tokens, 4)

tgram_list = [x for x in tgram]
qgram_list = [x for x in qgram]

#print(tgram_list) #[('I', 'love', 'data'), ('love', 'data', 'science'), ('data', 'science', 'and'), ('science', 'and', 'deep'), ('and', 'deep', 'learning')]
#print(qgram_list) #[('I', 'love', 'data', 'science'), ('love', 'data', 'science', 'and'), ('data', 'science', 'and', 'deep'), ('science', 'and', 'deep', 'learning')]

from nltk import ConditionalFreqDist

sentence = ['I love data science and deep learning', 'I love science', 'I know this code']

tokens = [word_tokenize(x) for x in sentence]
bgrams = [bigrams(x) for x in tokens]

token = []
for i in bgrams:
    token += ([x for x in i])
cfd = ConditionalFreqDist(token)
# print(cfd.conditions())
# ['I', 'love', 'data', 'science', 'and', 'deep', 'know', 'this']

# print(cfd['I'])

# <FreqDist with 2 samples and 3 outcomes>

# print(cfd['I']['love'])

# 2

# print(cfd['I'].most_common(1))
# [('love', 2)]

import numpy as np

freq_matrix = []

for i in cfd.keys():
    temp = []
    for j in cfd.keys():
        temp.append(cfd[i][j])
    freq_matrix.append(temp)
freq_matrix=np.array(freq_matrix)

# print(cfd.keys())
# print(freq_matrix)

# dict_keys(['I', 'love', 'data', 'science', 'and', 'deep', 'know', 'this'])
# [[0 2 0 0 0 0 1 0]
#  [0 0 1 1 0 0 0 0]
#  [0 0 0 1 0 0 0 0]
#  [0 0 0 0 1 0 0 0]
#  [0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 1]
#  [0 0 0 0 0 0 0 0]]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame(freq_matrix, index=cfd.keys(), columns=cfd.keys())

df.style.background_gradient(cmap='coolwarm')


import networkx as nx

G = nx.from_pandas_adjacency(df)

# print(G.nodes())
# print(G.edges())

#['I', 'love', 'data', 'science', 'and', 'deep', 'know', 'this']
#[('I', 'love'), ('I', 'know'), ('love', 'data'), ('love', 'science'), ('data', 'science'), ('science', 'and'), ('and', 'deep'), ('know', 'this')]

# print(G.edges()[('I', 'love')])
# print(G.edges()[('I', 'know')])

# {'weight': 2}
# {'weight': 1}

# nx.draw(G, with_labels=True)
# plt.show()

from nltk.probability import ConditionalProbDist, MLEProbDist

cpd = ConditionalProbDist(cfd, MLEProbDist)
# print(cpd.conditions())

# ['I', 'love', 'data', 'science', 'and', 'deep', 'know', 'this']

prob_matrix = []

for i in cpd.keys():
    prob_matrix.append([cpd[i].prob(j) for j in cpd.keys()])
    
prob_matrix = np.array(prob_matrix)

# print(cpd.keys())
# print(prob_matrix)

# dict_keys(['I', 'love', 'data', 'science', 'and', 'deep', 'know', 'this'])
# [[0.         0.66666667 0.         0.         0.         0.
#   0.33333333 0.        ]
#  [0.         0.         0.5        0.5        0.         0.
#   0.         0.        ]
#  [0.         0.         0.         1.         0.         0.
#   0.         0.        ]
#  [0.         0.         0.         0.         1.         0.
#   0.         0.        ]
#  [0.         0.         0.         0.         0.         1.
#   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.
#   0.         0.        ]
#  [0.         0.         0.         0.         0.         0.
#   0.         1.        ]
#  [0.         0.         0.         0.         0.         0.
#   0.         0.        ]]

df = pd.DataFrame(prob_matrix, index=cpd.keys(), columns=cpd.keys())
df.style.background_gradient(cmap = 'coolwarm')

prob_G = nx.from_pandas_adjacency(df)

# print(prob_G.nodes())
# print(prob_G.edges())

# ['I', 'love', 'data', 'science', 'and', 'deep', 'know', 'this']
# [('I', 'love'), ('I', 'know'), ('love', 'data'), ('love', 'science'), ('data', 'science'), ('science', 'and'), ('and', 'deep'), ('know', 'this')]

# print(G.edges()[('I', 'love')])
# print(G.edges()[('I', 'know')])

# {'weight': 2}
# {'weight': 1}

# print(prob_G.edges()[('I', 'love')])
# print(prob_G.edges()[('I', 'know')])

# {'weight': 0.6666666666666666}
# {'weight': 0.3333333333333333}

# nx.draw(prob_G, with_labels=True)
# plt.show()

# print(nx.degree_centrality(G))
# {'I': 0.2857142857142857, 'love': 0.42857142857142855, 'data': 0.2857142857142857, 
#  'science': 0.42857142857142855, 'and': 0.2857142857142857, 'deep': 0.14285714285714285, 'know': 0.2857142857142857, 'this': 0.14285714285714285}

# print(nx.eigenvector_centrality(G, weight='weight'))
# {'I': 0.5055042648573065, 'love': 0.6195557831651917, 'data': 0.35703593885196566, 
#  'science': 0.39841035839294925, 'and': 0.15933837227495717, 'deep': 0.055886131430398216, 'know': 0.20216573350291445, 'this': 0.07090581134630142}

# print(nx.closeness_centrality(G, distance='weight'))
# {'I': 0.35, 'love': 0.4375, 'data': 0.3684210526315789, 'science': 0.4117647058823529, 'and': 0.3333333333333333, 
#  'deep': 0.25925925925925924, 'know': 0.2916666666666667, 'this': 0.23333333333333334}

nx.betweenness_centrality(G)

# print(nx.betweenness_centrality(G))
# {'I': 0.47619047619047616, 'love': 0.5714285714285714, 'data': 0.0, 'science': 0.47619047619047616, 'and': 0.2857142857142857, 'deep': 0.0, 'know': 0.2857142857142857, 'this': 0.0}

nx.pagerank(G)
print(nx.pagerank(G))

# {'I': 0.1536831077679558, 'love': 0.19501225218917406, 'data': 0.10481873412175656, 'science': 0.15751225722745082, 'and': 0.12417333539164832, 
#  'deep': 0.07152392879557615, 'know': 0.1224741813421488, 'this': 0.07080220316428934}


def get_node_size(node_values):
    nsize = np.array([v for v in node_values])
    nszie = 1000 * (nsize - min(nsize)) / (max(nsize)- min(nsize))
    
    return nsize

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

dc = nx.degree_centrality(G).values()
ec = nx.eigenvector_centrality(G, weight='weight').values()
cc = nx.closeness_centrality(G, distance='weight').values()
bc = nx.betweenness_centrality(G).values()
pr = nx.pagerank(G).values()

plt.figure(figsize=(14, 20))
plt.axis('off')

plt.subplot(321)
plt.title('Normal', fontsize=16)
nx.draw_networkx(G, font_size=16, alpha=0.7, cmap=plt.cm.Blues)

plt.subplot(322)
plt.title('Degree Centrality', fontsize=16)
nx.draw_networkx(G, font_size=16, 
                 node_color=list(dc), node_size=get_node_size(dc),
                 alpha=0.7, cmap=plt.cm.Blues)

plt.subplot(323)
plt.title('Eigenvector Centrality', fontsize=16)
nx.draw_networkx(G, font_size=16, 
                 node_color=list(ec), node_size=get_node_size(ec),
                 alpha=0.7, cmap=plt.cm.Blues)

plt.subplot(325)
plt.title('Betweenness Centrality', fontsize=16)
nx.draw_networkx(G, font_size=16, 
                 node_color=list(bc), node_size=get_node_size(bc),
                 alpha=0.7, cmap=plt.cm.Blues)

plt.subplot(326)
plt.title('PageRank', fontsize=16)
nx.draw_networkx(G, font_size=16, 
                 node_color=list(pr), node_size=get_node_size(pr),
                 alpha=0.7, cmap=plt.cm.Blues)

# plt.show()

pl = nx.planar_layout(G)
frl = nx.fruchterman_reingold_layout(G)
sl = nx.spectral_layout(G)
nl = nx.random_layout(G)
sl = nx.shell_layout(G)
bl = nx.bipartite_layout(G, G.nodes())
cl = nx.circular_layout(G)
sl = nx.spring_layout(G)
kkl = nx.kamada_kawai_layout(G)

option = {
    'font_size' : 16,
    'node_color' : list(pr),
    'node_size' : get_node_size(pr),
    'alpha': 0.7, 
    'cmap': plt.cm.Blues
}

plt.figure(figsize=(15,15))
plt.axis('off')

plt.subplot(331)
plt.title('panner_layout', fontsize=16)
nx.draw_networkx(G, pos=pl, **option)

plt.subplot(332)
plt.title('fruchterman_reingold_layout', fontsize=16)
nx.draw_networkx(G, pos=frl, **option)

plt.subplot(333)
plt.title('spectral_layout', fontsize=16)
nx.draw_networkx(G, pos=sl, **option)

plt.subplot(334)
plt.title('random_layout', fontsize=16)
nx.draw_networkx(G, pos=nl, **option)

plt.subplot(335)
plt.title('shell_layout', fontsize=16)
nx.draw_networkx(G, pos=sl, **option)

plt.subplot(336)
plt.title('bipartite_layout', fontsize=16)
nx.draw_networkx(G, pos=bl, **option)

plt.subplot(337)
plt.title('circular_layout', fontsize=16)
nx.draw_networkx(G, pos=cl, **option)

plt.subplot(338)
plt.title('spring_layout', fontsize=16)
nx.draw_networkx(G, pos=sl, **option)

plt.subplot(339)
plt.title('kamada_kawai_layout', fontsize=16)
nx.draw_networkx(G, pos=kkl, **option)

plt.show()





