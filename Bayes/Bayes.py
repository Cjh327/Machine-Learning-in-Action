# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:52:45 2018

@author: tf
"""

from numpy import *

def trainNB0(trainMatrix: ndarray, trainCategory: ndarray) -> [ndarray, ndarray, int]:
    '''
    a naive Bayes
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
    p1Vec = p1Num / p1Denom
    p0Vec = p0Num / p0Denom
    return p1Vec, p0Vec, pAbusive

