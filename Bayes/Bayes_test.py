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


def createVocabList(dataSet: list) -> list:
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList: list, inputSet: list) -> np.ndarray:
    retVec = np.zeros((1, len(vocabList)))
    for word in inputSet:
        if word in vocabList:
            retVec[0, vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return retVec

def words2Mat(vocabList: list, inputSets: list) -> np.ndarray:
    retMat = np.zeros((len(inputSets), len(vocabList)))
    i = 0
    for wordsSet in inputSets:
        retMat[i] = setOfWords2Vec(vocabList, wordsSet)
        i += 1
    return retMat

listOPosts, listClasses = loadDataSet()
vocabList = createVocabList(listOPosts)
print(vocabList)
vec = setOfWords2Vec(vocabList, listOPosts[0])
print(vec)
trainMat = words2Mat(vocabList, listOPosts)
print(trainMat.shape)
p1V, p0V, pAb = Bayes.trainNB0(trainMat, np.array(listClasses))
print('p1V', p1V)
print('p0V', p0V)
print('pAb', pAb)