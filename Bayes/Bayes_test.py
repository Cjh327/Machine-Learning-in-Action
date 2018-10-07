# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:54:05 2018

@author: tf
"""
 
import Bayes
import numpy as np

def loadDataSet() -> [list, list]:
    postingList =  [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'], 
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [[0], [1], [0], [1], [0], [1]]
    return postingList, classVec


'''
listOPosts, listClasses = loadDataSet()
vocabList = createVocabList(listOPosts)
print(vocabList)
vec = setOfWords2Vec(vocabList, listOPosts[0])
print(vec)
trainMat = words2Mat(vocabList, listOPosts)
print(trainMat.shape)
p0V, p1V, pAb = Bayes.trainNB(trainMat, np.array(listClasses))
print('p1V', p1V)
print('p0V', p0V)
print('pAb', pAb)
'''

def testingNB():
    listPosts, listClasses = loadDataSet()
    vocabList = Bayes.createVocabList(listOPosts)
    trainMat = Bayes.words2Mat(vocabList, listOPosts)
    p0V, p1V, pAb = Bayes.trainNB(trainMat, np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = Bayes.setOfWords2Vec(vocabList, testEntry)
    print(testEntry, 'classified as: ', Bayes.classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'my', 'garbage']
    thisDoc = Bayes.setOfWords2Vec(vocabList, testEntry)
    print(testEntry, 'classified as: ', Bayes.classifyNB(thisDoc, p0V, p1V, pAb))
    
#testingNB()
    
Bayes.spamTest()