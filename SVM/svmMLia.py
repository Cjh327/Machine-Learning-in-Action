# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:20:50 2018

@author: tf
"""

import numpy as np

def loadDataSet(filename: str):
    '''
    load data set from file
    '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)

def selectJrand(i: float, m: float) -> float:
    '''
    generate random number in [i, m)
    '''
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L >aj:
        aj = L
    return aj

def smoSImple(dataMat: np.ndarray, labelMat: np.ndarray, C, toler, maxIter):
    '''
    a simple implementation of smo algrithm
    dataMat:        m * n
    labelMat:    m * 1 
    '''
    m, n = dataMat.shape
    labelMat = labelMat.reshape(labelMat.size, 1)
    b = 0
    alphas = np.zeros((m ,1))
    iterCnt = 0
    while iterCnt < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
