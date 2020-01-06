from numpy import*

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open("../book_sourceCode/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split('/t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    b = 0
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            FXi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b
            Ei = FXi - float(labelMat[i])
            if (labelMat[i]*Ei < -toler and alphas[i] < C) or (labelMat[i]*Ei > toler and alphas[i] > 0):
                

    return b, alphas