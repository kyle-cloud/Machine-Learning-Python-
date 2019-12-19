from math import log
import operator

def calcShannonEnt(dataSet):
    rowNumber = len(dataSet)
    LabelCounts = {}
    for rowVec in dataSet:
        currentLabel = rowVec[-1]
        if currentLabel not in LabelCounts:
            LabelCounts[currentLabel] = 0
        LabelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in LabelCounts:
        prob = float(LabelCounts[key]) / rowNumber
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for rowVec in dataSet:
        if rowVec[axis] == value:
            reducedVec = rowVec[: axis]
            reducedVec.extend(rowVec[axis+1 :])
            retDataSet.append(reducedVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numberFeature = len(dataSet[0]) - 1
    numberRow = len(dataSet)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numberFeature):
        colList = [example[i] for example in dataSet]
        uniqueValues = set(colList)
        newEntropy = 0.0
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(numberRow)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList.key():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClass = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClass[0][0]

def createTree(dataSet, labels):
    subLabels = labels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet) #
    bestFeatureLabel = labels[bestFeature]          #
    myTree = {bestFeatureLabel:{}}                  #
    del(subLabels[bestFeature])
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subDataSet = splitDataSet(dataSet, bestFeature, value)
        myTree[bestFeatureLabel][value] = createTree(subDataSet, subLabels)
    return myTree

def classify(inputTree, featureLables, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLables.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel