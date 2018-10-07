# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:52:45 2018

@author: tf
"""

from numpy import *
import numpy as np
import re

def createVocabList(dataSet: list) -> list:
    '''
    create vocabulary list
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList: list, inputSet: list) -> ndarray:
    retVec = zeros((1, len(vocabList)))
    for word in inputSet:
        if word in vocabList:
            retVec[0, vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return retVec

def words2Mat(vocabList: list, inputSets: list) -> ndarray:
    retMat = zeros((len(inputSets), len(vocabList)))
    i = 0
    for wordsSet in inputSets:
        retMat[i] = setOfWords2Vec(vocabList, wordsSet)
        i += 1
    return retMat

def trainNB(trainMatrix: ndarray, trainCategory: ndarray) -> [ndarray, ndarray, float]:
    '''
    train naive Bayes
    trainMatrix:    m*n
    trainCategory:  m*1
    '''
    m = trainMatrix.shape[0]
    n = trainMatrix.shape[1]
    pAbusive = trainCategory.sum() / m   # the prob of abusive docement (abusive document: class = 1)
    
    positionAbusive = argwhere(trainCategory.reshape(trainCategory.size) == 1)
    abusiveMat = trainMatrix[positionAbusive.reshape(positionAbusive.size)]
    p1Num = abusiveMat.sum(axis=0)
    p0Num = trainMatrix.sum(axis=0) - p1Num
    p1Denom = abusiveMat.sum()
    p0Denom = trainMatrix.sum() - p1Denom.sum()
    p1Vec = log((p1Num+1) / (p1Denom+2))
    p0Vec = log((p0Num+1) / (p0Denom+2))
    return p0Vec, p1Vec, pAbusive

def classifyNB(vec: ndarray, p0Vec: ndarray, p1Vec: ndarray, pClass1: float) -> int:
    '''
    classify with naive Bayes
    vec:    1*n
    p0Vec:  1*n
    p1Vec:  1*n
    '''
    vec = vec.reshape((1, vec.size))
    p0Vec = p0Vec.reshape((1, p0Vec.size))
    p1Vec = p1Vec.reshape((1, p1Vec.size))
    p1 = dot(vec, p1Vec.T) + log(pClass1);    # log[p(w|c1)p(c1)] = log(p(w|c1)) + log(p(c1))
    print('p1',p1)
    p0 = dot(vec, p0Vec.T) + log(1 - pClass1);
    print('p0',p0)
    if p1 > p0:
        return 1
    else:
        return 0

def textParse(bigString) -> list:
    '''
    take a big string and parses out the text into a list of strings
    '''
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''
    take a big string and parses out the text into a list of strings
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainList = docList
    trainClasses = classList
    testList = []
    testClasses = []
    for i in range(10):
        randIdx = int(random.uniform(0, len(trainList)))
        testList.append(trainList[randIdx])
        testClasses.append(trainClasses[randIdx])
        del(trainList[randIdx])
        del(trainClasses[randIdx])
    trainMat = words2Mat(vocabList, trainList)
    p0V, p1V, pSpam = trainNB(trainMat, np.array(trainClasses))
    errorCnt = 0
    for idx, doc in enumerate(testList):
        wordVec = setOfWords2Vec(vocabList, doc)
        if classifyNB(wordVec, p0V, p1V, pSpam) != testClasses[idx]:
            errorCnt += 1
    errorRate = errorCnt / len(testClasses)
    print('the error rate is: ', errorRate)
    

























