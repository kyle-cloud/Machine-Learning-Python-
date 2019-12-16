from numpy import*
from matplotlib import*
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#实例1（分类约会对象）：读文件-->画图分析数据-->归一化特征值（不同特征值权重一样，但数值范围不一样，归一化到0-1之间）
#             -->测试-->运行
def filetoMatrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines, 3))
    index = 0
    classLabelVector = []
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def plotMap(datingDataMat, datingLabels):
    fig = plt.figure()
    ax =  fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    
    normDataSet = zeros(shape(dataSet))
    normDataSet = dataSet - tile(minVals, (normDataSet.shape[0], 1))
    normDataSet = normDataSet / tile(ranges, (normDataSet.shape[0], 1))
    return normDataSet

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2Matrix('../book_sourceCode/Ch02/datingTestSet2.txt')
    normMat = autoNorm(datingDataMat)

    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifyResult = classify0(datingDataMat[i, :], datingDataMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the result is %d, the answer is %d" % (classifyResult, datingLabels[i]))
        if classifyResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("persentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2Matrix('../book_sourceCode/Ch02/datingTestSet2.txt')
    normMat = autoNorm(datingDataMat)

    inArr = array([ffMiles, percentTats, iceCream])
    classifyResult = classify0((inArr-datingDataMat.min(0))/(datingDataMat.max(0)-datingDataMat.min(0)), normMat, datingLabels, 3)
    print("you will probably like this person ", resultList[classifyResult-1])


#实例2（识别1-9图像）：
def imgtoVector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec

def handwritingClassTest():
    trainingFileList = os.listdir('../book_sourceCode/Ch02/trainingDigits')
    numberofFiles = len(trainingFileList)
    hwLabels = []
    trainingMat = zeros((numberofFiles, 1024))
    for i in range(numberofFiles):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = imgtoVector('../book_sourceCode/Ch02/trainingDigits/%s' % fileNameStr)
    
    errorCount = 0.0
    testFileList = os.listdir('../book_sourceCode/Ch02/testDigits')
    numberofTestFiles = len(testFileList)
    for i in range(numberofTestFiles):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        returnTestVec = imgtoVector('../book_sourceCode/Ch02/testDigits/%s' % fileNameStr)
        returnResult = classify0(returnTestVec, trainingMat, hwLabels, 3)
        print("the classifier came back with %d, the real answer is %d" % (returnResult, classNumber))
        if returnResult != classNumber:
            errorCount += 1.0
    print("the total error rate is %f" % (errorCount/float(numberofTestFiles)))
