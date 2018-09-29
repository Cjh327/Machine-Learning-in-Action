# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:31:56 2018

@author: tf
"""

import DecisionTree
import numpy as np

def createDataSet():
    dataSet = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0]])
    labels = np.array([['no surfcing', 'flippers']])
    print(labels.shape)
    return dataSet, labels

myDat, labels = createDataSet()
mat = DecisionTree.splitDataSet(myDat, 2, 1)
print(mat)
tree = DecisionTree.createDecisionTree(myDat, labels)
print(tree)