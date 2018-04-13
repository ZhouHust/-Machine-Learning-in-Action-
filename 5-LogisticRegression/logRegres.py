import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():         # 载入数据
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)

def sigmoid(inX):     # 激活函数
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatrix, classLabels):   # 优化算法：梯度上升
    m, n = dataMatrix.shape
    labelMat = np.array(classLabels).reshape(m, 1)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix.dot(weights))
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.T.dot(error)
    return weights

def plotBestFit(weights):    # 画出决策边界
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]
    cord = {
        0: { 'x': [], 'y': []},
        1: { 'x': [], 'y': []},
    }
    for i in range(n):
        id = int(labelMat[i] == 1)
        cord[id]['x'].append(dataArr[i, 1])
        cord[id]['y'].append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(cord[0]['x'], cord[0]['y'], s=30, c='red', marker='s')
    ax.scatter(cord[1]['x'], cord[1]['y'], s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):  # 随机梯度上升
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = np.ones((n, 1))
    for i in range(m):
        x = dataMatrix[i].reshape(n, 1)
        h = sigmoid(np.sum(x * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * x
    return weights

def stocGradAscent1(dataMatrix, classLebels, numIter=150, plot=False): # 改进的随机梯度上升
    m, n = dataMatrix.shape
    weights = np.ones(n)
    weights_history = [[], [], []]
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = np.random.randint(0, len(dataIndex))
            x = dataMatrix[randIndex]
            h = sigmoid(np.sum(x * weights))
            error = classLebels[randIndex] - h
            weights = weights + alpha * error * x
            del(dataIndex[randIndex])
            #added for recording weights' history
            for k in range(0, 3): weights_history[k].append(weights[k])
    #plot x0, x1, x2
    if plot:
        y_labels = ['x0', 'x1', 'x2']; col = ['red', 'blue', 'green']
        fig = plt.figure()
        for i in range(3):
            ax = fig.add_subplot(311 + i)
            ax.scatter(range(numIter * m), weights_history[i], s=1, c=col[i], marker='s')
            ax.set_xlabel('iteration'); ax.set_ylabel(y_labels[i])
        plt.show()

    return weights

if __name__ == '__main__':

    dataArr, labelMat = loadDataSet()

    weights = {
        0: gradAscent(dataArr, labelMat),
        1: stocGradAscent0(dataArr, labelMat),
        2: stocGradAscent1(dataArr, labelMat)
    }

    for i in range(3): plotBestFit(weights[i])
