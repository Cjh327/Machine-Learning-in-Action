# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:38:02 2018

@author: tf
"""

import AdaBoost
import numpy as np

#dataMat, labelMat = AdaBoost.loadDataSet()
#print(dataMat, '\n', labelMat)

#D = np.ones((5, 1)) / 5
#bestStump, minErr, bestClassEst = AdaBoost.buildStump(dataMat, labelMat, D)
#print(bestStump, '\n', minErr, '\n', bestClassEst)

#classifierArr = AdaBoost.adaBoostTrainDS(dataMat, labelMat)
#print(classifierArr)
#print(max(0.1,0.2))

#clas = AdaBoost.adaClassify(np.array([[5, 5], [0, 0]]), classifierArr)
#print(clas)

dataMat, labelMat = AdaBoost.loadFileDataSet('horseColicTraining2.txt')
classifierArr = AdaBoost.adaBoostTrainDS(dataMat, labelMat)
#print(classifierArr)

testDataMat, testLabelMat = AdaBoost.loadFileDataSet('horseColicTest2.txt')
errRate = AdaBoost.adaClassify(testDataMat, classifierArr, testLabelMat)
print(errRate)