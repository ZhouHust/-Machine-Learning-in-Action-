from math import log
import pickle
import operator
import treePlotter

def calcShannonEnt(dataSet):   # 计算香农熵(Entropy)
    n = len(dataSet)           # 数据集包含的实例总数
    labelCounts = {}           # 创建字典储存标签出现次数
    for featVec in dataSet:    # 对数据集中的实例(特征)
        currentLabel = featVec[-1]    # 取实例的最后一个特征
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 统计此特征出现次数存入字典
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / n  # 以频率作概率
        shannonEnt -= prob * log(prob, 2)   # 循环求和计算香农熵
    return shannonEnt

def createDataSet():  # 创造简单鱼鉴定数据集
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value): # 按给定特征划分数据集
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] # 在分类同时去除此特征
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet): # 选择划分数据集的最佳特征
    numFeatures = len(dataSet[0]) - 1  # 分类前数据集包含的特征数
    baseEntroy = calcShannonEnt(dataSet)  # 分类前的香农熵
    bestInfoGain = 0.0          # 最佳信息增益初始化
    bestFeature = -1            # 最佳特征序号初始化
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 将所有特征的所有可能值存入列表
        uniqueVals = set(featList)   # 通过建立set去重
        newEntroy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # 对每个特征可能值进行分类
            prob = len(subDataSet) / float(len(dataSet)) # 计算该值的概率(频率)
            newEntroy += prob * calcShannonEnt(subDataSet) # 计算该值分类后的熵
        infoGain = baseEntroy - newEntroy #计算该分类的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i       # 返回信息增益最大的分类特征
    return bestFeature

# 如果已处理所有可能值分类，类标签依然不唯一，则采用出现次数最多的分类定义该叶子节点
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels): # 创建决策树，输入参数(训练集，和标签集)
    classList = [example[-1] for example in dataSet] 
    if classList.count(classList[0]) == len(classList): return classList[0] # 为同一类
    if len(dataSet[0]) == 1: return majorityCnt(classList) # 数据集中只含有一个特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, 
                        bestFeat, value), subLabels)  # 递归构建决策树
    return myTree

def classify(inputTree, featLabels, testVec): # 使用决策树进行分类
    tempList = list(inputTree.keys())
    firstStr = tempList[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)  # 储存tree
    fw.close()

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)  # 载入tree

if __name__ == '__main__':
#----------简单鱼鉴定数据集--------------
    # myDat, labels = createDataSet()
    # entropy = calcShannonEnt(myDat)
    # print(entropy)
    # print(splitDataSet(myDat, 0, 1))
    # print(splitDataSet(myDat, 0, 0))
    # bestFeature = chooseBestFeatureToSplit(myDat)
    # print(bestFeature)
    # myTree = createTree(myDat, labels)
    # print(myTree)

    # #test
    # myDat, labels = createDataSet()
    # myTree = treePlotter.retrieveTree(0)
    # result = classify(myTree, labels, [1, 0])
    # print(result)
    # result = classify(myTree, labels, [1, 1])
    # print(result)

    # #store
    # storeTree(myTree, 'classifierStorage.txt')
    # mt = grabTree('classifierStorage.txt')
    # print(mt)
#----------------------------------------------------

    # lenses
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)
