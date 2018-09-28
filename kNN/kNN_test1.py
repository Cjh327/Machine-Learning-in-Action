# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:45:40 2018

@author: tf
"""

import kNN

def createDataSet():
    group = array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = createDataSet()
res = kNN.classify0([1,1], group, labels, 3)
print(res)