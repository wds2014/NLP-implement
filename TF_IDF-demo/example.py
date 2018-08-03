# -*- coding: utf-8 -*-
####################################################
#         ╔══╗                                     #
#         ╚╗╔╝                                     #
#         ╔╝(¯`v´¯)                                #
#         ╚══`.¸.Coding ~                          #
#                                                  #
# @Author: wang.dongsheng                          #
# @E-mail: hellowds2014@gmail.com                  #
# @Date:   2018-07-22 23:47:02                     #
# @Last Modified by:   wang.dongsheng              #
# @Last Modified time: 2018-08-03 10:26:11        #
####################################################
import numpy as np
import lda
from tfidf import tfidf
X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

X = tfidf(X)
x_1 = np.vsplit(X, 395)

n_top_words = 8
index_y = []

for row in x_1:
    index = []
    for j in range(0,n_top_words):
         index_max = np.where(row == np.max(row))
         if index_max[0].shape[0] > 1:
              index_max = index_max[1][0]
              index.append(index_max)
              row[0][index_max] = 0
         else:
              index.append(int(index_max[1]))
              row[index_max] = 0
    index_y.append(index)

for i in range(20):
   topic_words_index = index_y[i]
   print('Topic:%d' %i, end=' ')
   for word_index in topic_words_index:
        print(vocab[word_index], end=' ')
   print('\n')


# print(X.shape)
# print("%%%%%%%%%%%%%%%%%%%%%%")
# model = lda.LDA(n_topics=20,n_iter=1500,random_state=1)
# model.fit(X)
# topic_word = model.topic_word_
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
#     topic_words=np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
