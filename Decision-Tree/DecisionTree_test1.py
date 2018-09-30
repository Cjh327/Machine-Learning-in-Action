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
    return dataSet, labels


myDat, labels = createDataSet()
tree = DecisionTree.createDecisionTree(myDat, labels)
print('tree', tree)
print(DecisionTree.classify(tree, labels, [1,1]))
DecisionTree.storeTree(tree, 'tree.txt')
newtree = DecisionTree.grabTree('tree.txt')
print(newtree)


fr = open('lenses.txt')
lenses = np.array([inst.strip().split('\t') for inst in fr.readlines()])
lensesLabels = np.array([['age', 'prescript', 'astigmatic', 'tearRate']])
lensesTree = DecisionTree.createDecisionTree(lenses, lensesLabels)
print(lensesTree)