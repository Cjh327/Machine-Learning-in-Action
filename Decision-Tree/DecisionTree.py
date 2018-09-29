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


def splitDataSet(dataSet, axis, value):
    '''
    split dataSet by value on axis
    '''
    retDataSet = np.array([])
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = np.concatenate((featVec[:axis], featVec[axis+1:]),axis=0)
            np.append(retDataSet, reducedFeatVec)
    return retDataSet


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
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    