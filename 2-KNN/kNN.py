import numpy as np
import matplotlib.pyplot as plt
import operator
import os

# def createDataSet():
#     group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels

#--------DataingPerson Classification------------

def classify0(X, dataSet, labels, k):   # 最近邻分类，X为单行向量
    diffMat = X - dataSet               # X减去dataSet每一行，得到与dataSet同尺寸的数组
    sqDiffMat = diffMat**2
    sqDistances = np.sum(sqDiffMat, axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()   # argsort()按值排序，返回对应索引
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1   # dict.get(key,default=None),返回key对
                                                                   # 应的value，若不存在则返回default
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # sorted(iterable, cmp=None, key=None, reverse=False), dict.items()将dict分解为元组列表，
    # operator.itemgetter(1)按第二个元素对元组进行排序,reverse=True降序排列,返回排序后的元组列表
    return sortedClassCount[0][0] # 返回出现次数最多的标签

def file2matrix(filename):  #将训练集数据从文本转成np数组/矩阵
    fr = open(filename)
    text = fr.readlines()    # 每行视作一个字符串，返回字符串列表
    n = len(text)          # 行数
    X = np.zeros((n, 3))   # 初始化 X
    y = []
    index = 0
    for line in text:
        line = line.strip()     # 去除首尾空格tab或换行符等空字符，返回剩余字符
        listFromLine = line.split('\t')  # 以tab分割字符串，返回分割后的字符串列表
        X[index, :] = listFromLine[0:3]  # 将训练集数据写入 X
        y.append(int(listFromLine[-1]))  # 将训练集标签写入 y
        index += 1
    return X, y  # 返回训练集数据

def autoNorm(X):       # 均值归一化
    minVals = np.min(X, axis=0)  # 对数组 X 的每一列求最小值，返回一个与X列数相等的单行np数组
    maxVals = np.max(X, axis=0)
    ranges = maxVals - minVals
    normX = (X - minVals) / ranges 
    return normX, ranges, minVals

def datingClassTest():  # 验证数据分类
    X, y = file2matrix('datingTestSet2.txt')
    X, ranges, minVals = autoNorm(X)
    p = 0.1             # 选取测试集的比例
    n = X.shape[0]
    testSize = int(n * p)  # 测试集的大小
    errorCount = 0.0
    for i in range(testSize):
        res = classify0(X[i, :], X[testSize:n, :], y[testSize:n], 3)  #对测试进行分类验证
        print('the classifier came back with %d, the real answer is: %d' % (res, y[i]))
        if (res != y[i]): errorCount += 1.0  # 统计错误分类数量
    print('the total error rate is: %f' % (errorCount / float(testSize)))   # 输出错误率

def classifyPerson():   # 输入参数进行分类预测
    X, y = file2matrix('datingTestSet2.txt')
    X, ranges, minVals = autoNorm(X)
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice Cream consumed per year?'))
    inArr = np.array([ffMiles, percentTats, iceCream])
    res = classify0((inArr - minVals) / ranges, X, y, 3)
    print('You will probably like this person:', resultList[res - 1])

def draw():   # 对训练数据进行可视化
    X, y = file2matrix('datingTestSet2.txt')
    X, ranges, minVals = autoNorm(X)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], 15.0 * np.array(y), 15.0 * np.array(y))
    plt.show()


#--------Handwriting Number Recognition---------

def img2vec(filename):  
    vec = np.zeros((1, 1024))   # 32*32 img --> 1*1024 vec
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vec[0, 32 * i + j] = int(line[j])
    return vec

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')  # 返回该路径下的文件(夹)名称列表
    n = len(trainingFileList)  # 训练集样品数量
    trainingMat = np.zeros((n, 1024))  # 初始化训练集数据
    for i in range(n):
        fileNameStr = trainingFileList[i]   # 获得文件名(含扩展名),'3_20.txt'中3表示实际标签,20表示该标签下的训练数据编号
        fileStr = fileNameStr.split('.')[0]   # 文件名(不含扩展名)
        classNumStr = int(fileStr.split('_')[0]) # 实际标签标号
        hwLabels.append(classNumStr)  # 按顺序对实际标签进行储存
        trainingMat[i, :] = img2vec('digits/trainingDigits/%s' % fileNameStr) #按顺序将训练数据文本转成矩阵存入数组

    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    nTest = len(testFileList)
    for i in range(nTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vec('digits/testDigits/%s' % fileNameStr)  # 按顺序将测试数据文本转成输入向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  #按顺序进行分类预测
        print('the classifier came back with: %d. the real answer is: %d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0   # 统计错误数量
            # print('filename: %s %d' % (fileNameStr, classifierResult))

    print('the total number of errors is: %d' % errorCount)
    print('the total error rate is: %f' % (errorCount / float(nTest)))


if __name__ == '__main__':
    datingClassTest()
    draw()
    classifyPerson()
    handwritingClassTest()
    

