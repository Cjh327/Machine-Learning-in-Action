# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 19:12:13 2018

@author: tf
"""
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet() -> [np.ndarray, np.ndarray]:
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)

def sigmoid(inX: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-inX))

def gradAscent(dataMat: np.ndarray, classLabels: np.ndarray) -> np.ndarray:
    '''
    gradient ascent
    dataMat:        m * n
    classLabels:    m * 1
    '''
    m, n = dataMat.shape
    classLabels = classLabels.reshape(classLabels.size, 1)
    alpha = 0.001
    maxIter = 500
    weights = np.random.random((n, 1))
    for k in range(maxIter):
        h = sigmoid(dataMat.dot(weights))
        error = classLabels - h     # error: m * 1
        weights = weights + alpha * dataMat.T.dot(error)
    return weights

def plotBestFit(weights):
    '''
    plot best fit
    '''
    dataMat, labelMat = loadDataSet()
    m, n = dataMat.shape
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
        else:
            xcord2.append(dataMat[i, 1])
            ycord2.append(dataMat[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def classifyVector(inX: np.ndarray, weights: np.ndarray) -> int:
    prob = sigmoid(inX.dot(weights))
    if prob > 0.5:
        return 1
    else:
        return 0
    
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscent(np.array(trainingSet), np.array(trainingLabels))
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ('the error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests)))