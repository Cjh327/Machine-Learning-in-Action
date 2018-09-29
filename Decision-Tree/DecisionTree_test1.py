# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:31:56 2018

@author: tf
"""

import DecisionTree

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfcing', 'flippers']
    return dataSet, labels

myDat, labels = createDataSet()
print(myDat)
ent = DecisionTree.calcShannonEnt(myDat)
print(ent)