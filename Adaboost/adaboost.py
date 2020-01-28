from numpy import *

def loadSimData():
    dataMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix, demen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    今天太der了\
        移动要好好学英语啊
        明天开始以前的工作作息
        明天大姐姐夫还有大姨回去了，嗯，今天实在不想写。不过明天就正式开始魔鬼学习了
def buildStump(dataArr, classLabells, D):

