import numpy as np
import feedparser   # RSS(Really Simple Syndication)程序库
import operator
import re

def loadDataSet():   # 创建简单实验样本
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec   # 词条切分后的文档集合，文档的类别标签

def createVocabList(dataSet): # 创建文档中不重复的词条列表
    vocabSet = set([])  # 通过set去重
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # set取并集
    return list(vocabSet)  # 返回去重之后的词条列表

#---词集模型(出现即记为 1，不考虑出现次数)
def setofWords2Vec(vocabList, inputSet):  # 输入为(词汇表,某个文本)
    returnVec = [0] * len(vocabList)  # 初始化置 0 返回值,与词汇表等长向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 # 文本中出现词汇表中的词则设为 1
        else:
            print('the word: %s is not in my vocabulary!', word)
    return returnVec  # 返回文本中是否含有词汇表中对应词条的结果

#---词袋模型(考虑出线次数，返回次数)
def bagofWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 出现一次加 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory): # 训练朴素贝叶斯模型,输入为(训练文本词条矩阵，训练数据的分类向量)
    numTrainDocs = len(trainMatrix)  # 训练文本总数
    numWords = len(trainMatrix[0])   # 第一个文本的词条向量包含的词条数目
    pAbusive = np.sum(trainCategory) / float(numTrainDocs) # 计算训练集中 1 分类的概率

    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0

    # 拉普拉斯平滑，见《统计学习方法》P51
    p0Num = np.ones(numWords)  # 为避免单个 0 次出现导致总概率为 0
    p1Num = np.ones(numWords)  # 统一初始化为 1
    p0Denom = 2.0 # 总词数相应初始化为 2
    p1Denom = 2.0
    for i in range(numTrainDocs): # 遍历所有训练文本
        if trainCategory[i] == 1: # 若分类为 1
            p1Num +=trainMatrix[i] # 相应词的次数累积
            p1Denom += np.sum(trainMatrix[i]) # 出现的总词数
        else:  # 同理，分类 0
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    # p0Vect = p0Num / p0Denom   # 条件概率,P(词出现|分类为0)
    # p1Vect = p1Num / p1Denom   # 分类为 1 的条件下，各词出现的概率
    p0Vect = np.log(p0Num / p0Denom)  # 避免下溢出改进为取对数
    p1Vect = np.log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive  # 返回词汇表对应的概率向量及1分类的概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1): # 输入为(待分类的文本词向量，训练(TrainNB0)返回值)
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1) # 表征P(分类为1|当前词向量情况) <-- P(当前词向量情况|分类为1) * P(分类为1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1 - pClass1) # 表征P(分类为0|当前词向量情况) <-- P(当前词向量情况|分类为0) * P(分类为0)
    return 1 if p1 > p0 else 0                              # 均应除以P(当前词向量)，故只比较分子

def testingNB(vec2Classify): # 创建简单实验样本测试NB模型
    listOPosts, listClasses = loadDataSet()  # 简单样本,返回训练数据矩阵和标签向量
    myVocabList = createVocabList(listOPosts) # 获取去重之后的词汇表
    trainMat = []  # 初始化训练矩阵
    for postinDoc in listOPosts:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc)) # 准备训练矩阵  
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)  # 训练
    thisDoc = np.array(setofWords2Vec(myVocabList, vec2Classify)) # 输入文本词条进行测试
    print(vec2Classify, 'classified as ', classifyNB(thisDoc, p0V, p1V, pAb))

# --------使用朴素贝叶斯进行交叉验证------
def textParse(bigString):  # 文本解析,接收大字符串并解析为字符串列表
    listOfTokens = re.split('\\W*', bigString)   # 正则匹配
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():  # 垃圾邮件测试函数
    docList = []     # 文本列表(数组)
    classList = []   # 标签列表
    fullText = []    # 全部文本列表
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)   # 文本数组
        fullText.extend(wordList)  # 全部文本信息
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 从文本数组创建词汇表
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):   # 随机选择10个文本作为测试集
        randIndex = np.random.randint(0, len(trainingSet))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []      # 训练集矩阵
    trainClasses = []  # 训练集的标签向量
    for docIndex in trainingSet:
        trainMat.append(setofWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 训练

    errorCount = 0  # 初始化错误率，后面计算错误率
    for docIndex in testSet:
        wordVector = setofWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('the error document is:', docList[docIndex])
    print('the error rate is :', float(errorCount) / len(testSet))
'''
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = np.random.randint(0, len(trainingSet))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagofWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagofWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF: print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY: print(item[0])
'''
if __name__ == '__main__':

    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # vec1 = setofWords2Vec(myVocabList, listOPosts[0])
    # print(vec1)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print('p0V = ', p0V)
    # print('p1V = ', p1V)
    # print('pAb = ', pAb)
    #
    # #test
    testingNB(['love', 'my', 'dalmation'])
    testingNB(['stupid', 'garbage'])
    #
    # #SpamTest
    spamTest()
'''
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    # print(ny['entries'])
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocabList, pSF, pNY = localWords(ny, sf)
    getTopWords(ny, sf)
'''
