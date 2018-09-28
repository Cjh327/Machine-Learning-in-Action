# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:13:47 2018

@author: tf
"""

import kNN

'''
# data reading
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')
#print(datingDataMat)
#print(datingLabels)
print(datingDataMat.shape)
print(len(datingLabels))

import matplotlib
import matplotlib.pyplot as plt

# data plotting
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15*array(datingLabels), 15*array(datingLabels))

# data normolization
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print('normMat', normMat)
print('ranges', ranges)
print('minVals', minVals)
'''

kNN.datingClassTest()