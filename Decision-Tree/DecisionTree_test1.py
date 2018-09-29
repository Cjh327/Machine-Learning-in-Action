# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:31:56 2018

@author: tf
"""

import DecisionTree
import numpy as np

def createDataSet():
    dataSet = np.array([[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']])
    labels = np.array(['no surfcing', 'flippers'])
    return dataSet, labels

myDat, labels = createDataSet()
tmp = [example[2] for example in myDat]
print(tmp)
print(myDat)
ent = DecisionTree.calcShannonEnt(myDat)
print(ent)
feat = DecisionTree.chooseBestFeatureToSplit(myDat)
print(feat)