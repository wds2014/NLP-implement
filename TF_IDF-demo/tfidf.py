# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:28:32 2018

@author: Hu.Haozhou
@E-mail:
@Last Modified by:   Hu.haozhou
@Last Modified time: 2018-07-29 13:43:42

"""
import numpy as np

def tfidf(x):
    N = x.shape[0]
    temp = np.sum(np.minimum(x,1), axis=0)
    idf_x = np.log(N/(1+temp))
    tfidf = idf_x * x
    return tfidf