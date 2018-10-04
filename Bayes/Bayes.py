# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:52:45 2018

@author: tf
"""

from numpy import *

def trainNB(trainMatrix: ndarray, trainCategory: ndarray) -> [ndarray, ndarray, float]:
    '''
    train naive Bayes
    trainMatrix:    m*n
    trainCategory:  m*1
    '''
    m = trainMatrix.shape[0]
    n = trainMatrix.shape[1]
    pAbusive = trainCategory.sum() / m   # the prob of abusive docement (abusive document: class = 1)
    
    positionAbusive = argwhere(trainCategory.reshape(trainCategory.size) == 1)
    abusiveMat = trainMatrix[positionAbusive.reshape(positionAbusive.size)]
    p1Num = abusiveMat.sum(axis=0)
    p0Num = trainMatrix.sum(axis=0) - p1Num
    p1Denom = abusiveMat.sum()
    p0Denom = trainMatrix.sum() - p1Denom.sum()
    p1Vec = log((p1Num+1) / (p1Denom+2))
    p0Vec = log((p0Num+1) / (p0Denom+2))
    return p0Vec, p1Vec, pAbusive

def classifyNB(vec: ndarray, p0Vec: ndarray, p1Vec: ndarray, pClass1: float) -> int:
    '''
    classify with naive Bayes
    vec:    1*n
    p0Vec:  1*n
    p1Vec:  1*n
    '''
    vec = vec.reshape((1, vec.size))
    p0Vec = p0Vec.reshape((1, p0Vec.size))
    p1Vec = p1Vec.reshape((1, p1Vec.size))
    p1 = dot(vec, p1Vec.T) + log(pClass1);    # log[p(w|c1)p(c1)] = log(p(w|c1)) + log(p(c1))
    print('p1',p1)
    p0 = dot(vec, p0Vec.T) + log(1 - pClass1);
    print('p0',p0)
    if p1 > p0:
        return 1
    else:
        return 0
