# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:42:03 2018

@author: tf
"""

from numpy import *
import operator
import os, sys


#2.1 a simple kNN classifier
def classify0(inX, dataSet, labels, k):
    '''
    a simple kNN classifier
    '''
    dataSetSize = dataSet.shape[0];
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#2.2 Helen dating
def file2matrix(filename):
    '''
    convert a txt file to matrix
    '''
    fr = open(filename)
    lineArray = fr.readlines()
    lineNum = len(lineArray)
    retMat = zeros((lineNum, 3))
    classLabelVector = []
    idx = 0
    for line in lineArray:
        line = line.strip()
        listFromLine = line.split('\t')
        retMat[idx, :] = listFromLine[0:3]
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'didntLike':
            classLabelVector.append(3)
        else:
            classLabelVector.append(0)
        idx += 1
    return retMat, classLabelVector


def autoNorm(dataSet):
    '''
    data normolization
    normval = (oldval - min) / (max - min)
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    '''
    test kNN classifier on dating data
    '''
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCnt = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCnt += 1
    errorRate = errorCnt / numTestVecs
    print('the total error rate is: %f' %(errorRate))
    

#2.3 handwriting recognizing
def img2vector(filename):
    '''
    convert a 32*32-size image file to a 1*1024-size vector
    '''
    retVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            retVect[0, 32*i+j] = int(lineStr[j])
    return retVect

def handwritingClassTest():
    '''
    test kNN classifier on handwriting data
    '''
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    mTrain = len(trainingFileList)
    trainingMat = zeros((mTrain, 1024))
    for i in range(mTrain):
        filename = trainingFileList[i]
        classNum = int(filename.split('.')[0].split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % filename)
    testFileList = os.listdir('testDigits')
    errorCnt = 0
    mTest = len(testFileList)
    for i in range(mTest):
        filename = testFileList[i]
        classNum = int(filename.split('.')[0].split('_')[0])
        vecUnderTest = img2vector('trainingDigits/%s' % filename)
        classifierResult = classify0(vecUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult, classNum))
        if classifierResult != classNum:
            errorCnt += 1
    errorRate = errorCnt / mTest
    print('\nthe total number of error rate is: %f' % errorRate)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    