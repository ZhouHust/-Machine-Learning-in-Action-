import logRegres
import numpy as np

def classifyVector(inX, weights): 
    prob = logRegres.sigmoid(np.sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

def loadData(filename):
    X = []; y = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): lineArr.append(float(curLine[i]))
        X.append(lineArr); y.append(float(curLine[21]))
    return X, y

def colicTest():
    trainingSet, trainingLabels = loadData('horseColicTraining.txt')
    testSet, testLabels = loadData('horseColicTest.txt')
    trainWeights = logRegres.stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = len(testLabels)
    for i in range(numTestVec):
        if int(classifyVector(testSet[i], trainWeights)) != int(testLabels[i]): errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print('the error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest(numTests=10):
    errorSum = 0.0
    for k in range(numTests): errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % (numTests, errorSum / float(numTests)))

if __name__ == '__main__':
    multiTest()

