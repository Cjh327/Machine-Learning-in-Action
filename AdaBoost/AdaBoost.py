# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:22:09 2018

@author: tf
"""

import numpy as np
from math import *

def loadDataSet() -> [np.ndarray, np.ndarray]:
    dataMat = np.array([[1, 2.1],
                        [2, 1.1],
                        [1.3, 1],
                        [1, 1],
                        [2, 1]])
    labelMat = np.array([[1],[1],[-1],[-1],[1]])
    return dataMat, labelMat

def stumpClassify(dataMat: np.ndarray, dimen: int, threshVal: float, threshIneq: str) -> np.ndarray:
    '''
    Function: 通过对阈值比较对数据进行分类
    dataMat:    m * n
    dimen:      int, 第dimen个特征
    threshVal:  float, 阈值
    threshIneq: str, 'lt': less than 'gt': greater than
    '''
    m, n = dataMat.shape
    retArr = np.ones((m, 1))
    if threshIneq == 'lt':
        retArr[dataMat[:, dimen] <= threshVal] = -1
    else:
        retArr[dataMat[:, dimen] > threshVal] = -1
    return retArr

def buildStump(dataMat: np.ndarray, labelMat: np.ndarray, D: np.ndarray):
    '''
    Function: 遍历stumpClassify()函数所有可能输入值，并找到数据集上最佳的单层决策树
    dataMat:    m * n
    labelMat:   m * 1
    D:          m * 1   权重向量
    '''
    m, n = dataMat.shape
    assert labelMat.shape == (m, 1)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.zeros((m, 1))
    minErr = np.inf
    for i in range(n):  # 遍历数据集的所有特征
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps     # 计算步长 
        for j in range(-1, int(numSteps) + 1):      # 遍历阈值
            for inequal in ['lt', 'gt']:            # 遍历大小关系
                threshVal = rangeMin + float(j) * stepSize
                predVals = stumpClassify(dataMat, i, threshVal, inequal)
                errMat = np.ones((m, 1))
                errMat[predVals == labelMat] = 0
                weightedErr = D.T.dot(errMat)
                # print('split: dim %d, thresh %.2f ,thresh inequal: %s, the weighted error is %.3f' \
                #      % (i, threshVal, inequal, weightedErr))
                if weightedErr < minErr:
                    minErr = weightedErr
                    bestClassEst = predVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClassEst
    
def adaBoostTrainDS(dataMat: np.ndarray, labelMat: np.ndarray, numIter: int=40):
    '''
    Function: AdaBoost算法（DS: decision stump 单层决策树，AdaBoost中较为流行的弱分类器）
    dataMat:    m * n
    labelMat:   m * 1
    '''
    m, n = dataMat.shape
    assert labelMat.shape == (m, 1)
    weakClassArr = []
    D = np.ones((m ,1)) / m
    aggClassEst = np.zeros((m, 1))  # 记录每个数据点的类别估计累计值
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataMat, labelMat, D)
        print('D:', D.T)
        alpha = 0.5 * log((1 - error) / max(error, 1e-16))    # max(error, 1e-16)确保不会产生除零错误
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:', classEst.T)
        expon = np.multiply(-1 * alpha * labelMat, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / np.sum(D)
        aggClassEst += alpha * classEst
        print('aggClassEst:', aggClassEst.T)
        aggErr = np.multiply(np.sign(aggClassEst) != labelMat, np.ones((m, 1)))   # sign(): 大于0的返回1.0, 小于0的返回-1.0, 等于0的返回0.0
        errRate = np.sum(aggErr) / m
        print('total error: ', errRate, '\n')
        if errRate == 0:
            break
    return weakClassArr

    
    
    
    
    
    
    
    
    