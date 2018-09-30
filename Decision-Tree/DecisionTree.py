# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:19:45 2018

@author: tf
"""

from math import log
import numpy as np
import operator
import matplotlib.pyplot as plt

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
    isDeleted = np.argwhere(dataSet[:, col] != value)
    reducedMat = np.delete(dataSet, col, axis=1)
    reducedMat = np.delete(reducedMat, np.reshape(isDeleted, isDeleted.size), axis=0)
    return reducedMat

def chooseBestFeatureToSplit(dataSet):
    '''
    choose best feature to split
    '''
    numFeat = dataSet.shape[1] - 1
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
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if dataSet.shape[1] == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[0, bestFeat]
    tree = {bestFeatLabel: {}}
    labels =  np.delete(labels, bestFeat, axis=1)
    featVals = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVals)
    # for every value of bestFeat, build a branch of the tree
    for val in uniqueVals:
        subLabels = labels[:]
        reducedDataSet = splitDataSet(dataSet, bestFeat, val);
        tree[bestFeatLabel][val] = createDecisionTree(reducedDataSet, subLabels)
    return tree
    
def classify(tree, featLabels, testVec):
    '''
    classify using decision tree
    '''
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    featIdx = np.argwhere(featLabels[0, :] == firstStr)[0][0]
    for key in secondDict.keys():
        if testVec[featIdx] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(tree, filename):
    '''
    store decision tree on disk
    '''
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(tree, fw)
    fw.close()
    
def grabTree(filename):
    '''
    grab tree stored on disk
    '''
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

    
    
    
    
    
    
    
    
    
    
    
    
    