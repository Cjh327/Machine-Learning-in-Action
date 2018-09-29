# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:19:45 2018

@author: tf
"""

from math import log
import numpy as np
import operator

def calcShannonEnt(dataSet):
    '''
    calculate shannon entropy
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, col, value):
    '''
    split dataSet by value on col
    '''
    m = dataSet.shape[0]
    isDeleted = np.argwhere(dataSet[:, col] == value)
    reducedMat = np.delete(dataSet, col, axis=1)
    reducedMat = np.delete(reducedMat, np.reshape(isDeleted, isDeleted.size), axis=0)
    return reducedMat

def chooseBestFeatureToSplit(dataSet):
    '''
    choose best feature to split
    '''
    numFeat = dataSet.shape[1]
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeat = -1
    for i in range(numFeat):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = subDataSet.shape[0] / dataSet.shape[0]
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat
    
    
def majorityCnt(classList):
    '''
    sort class
    '''
    classCnt = {}
    for vote in classList:
        if vote not in classCnt.keys():
            classCnt[vote] = 0
        classCnt[vote] += 1
    sortedClassCnt = sorted(classCnt.items(), key=operator.itemgetter(1), reversed=True)
    return sortedClassCnt[0][0]

def createDecisionTree(dataSet, labels):
    '''
    create decision tree
    '''
    classList = [example[-1] for example in dataSet]
    # stop classifying when all the classes are same
    print(dataSet)
    print(classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if dataSet.shape[1] == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    tree = {bestFeatLabel: {}}
    np.delete(labels, bestFeat)
    featVals = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVals)
    # for every value of bestFeat, build a branch of the tree
    for val in uniqueVals:
        subLabels = labels[:]
        tree[bestFeatLabel][val] = createDecisionTree(splitDataSet(dataSet, bestFeat, val), subLabels)
    return tree
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    