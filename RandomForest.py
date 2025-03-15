import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from math import log
import operator

# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#定义结构树
def CreatTree(dataset, labels, featlabels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):  # 所有类别相同
        return classlist[0]
    if len(dataset[0]) == 1:  # 到叶子了
        return majorityCnt(classlist)
    
    bestfeat = BestFeature(dataset)
    bestfeatlabel = labels[bestfeat]
    featlabels.append(bestfeatlabel)
    myTree = {bestfeatlabel: {}} 
    sublabels = labels[:]
    np.delete(sublabels,bestfeat)
    featvalues = [example[bestfeat] for example in dataset]
    uniquevalues = set(featvalues)  
    for value in uniquevalues:
        myTree[bestfeatlabel][value] = CreatTree(
            splitdataset(dataset, bestfeat, value), sublabels, featlabels)
    
    return myTree

#计算多数类别
def majorityCnt(classlist):
    classCount={}
    for value in classlist:
        if value not in classCount.keys():
            classCount[value]=0
        classCount[value]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#计算熵值得出最佳节点
def BestFeature(dataset):
    numFeatures=len(dataset[0])-1
    #计算当前熵值
    baseEntroy=calculateEntroy(dataset)
    #最大的信息增益
    bestInformationGain=0
    bestFeature=-1
    for i in range( numFeatures) :
        featlist=[examples[i] for examples in dataset]
        uniquevalues=set(featlist)
        newEntroy=0
        for value in uniquevalues:
            subDataset=splitdataset(dataset,i,value)
            prob=len(subDataset)/float(len(dataset))
            newEntroy+=prob*calculateEntroy(subDataset)
        informationGain=baseEntroy-newEntroy
        if informationGain>bestInformationGain:
            bestInformationGain=informationGain
            bestFeature=i
    return bestFeature



#取出当前特征每种取值的数据
def splitdataset(dataset,i,value):
    subdataset=[]
    for example in dataset:
        if example[i]==value:
            subexample=np.concatenate((example[:i], example[i+1:]))
            subdataset.append(subexample)
    return subdataset
#计算当前节点的熵值
def calculateEntroy(dataset):
    numexamples=len(dataset)
    labelCounts={}
    for example in dataset:
        if example[-1] not in labelCounts.keys():
            labelCounts[example[-1]]=0
        labelCounts[example[-1]]+=1

    Entroy=0
    for key in labelCounts:
        prop=labelCounts[key]/numexamples
        Entroy-=prop*log(prop,2)
    return Entroy


# 定义决策树
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree= None

    def fit(self, X, y):
        self.tree= self.BuildTree(X, y)

    def BuildTree(self, X, y):
        featlabels = []
        return CreatTree(X, y,  featlabels)

    def predict(self, X):
        return [self.Predict(inputs) for inputs in X]
    
    # 预测类别
    def Predict(self, inputs): 
        tree = self.tree
        while isinstance(tree, dict):
            feature = next(iter(tree))
            value = inputs[feature]
            if value in tree[feature]:
                tree = tree[feature][value]
            else:
                return None
        return tree

# 定义随机森林
class RandomForest:
    def __init__(self, num_trees=10, max_depth=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []
    #随机取样和随机选取特征
    def fit(self, X, y):
        for _ in range(self.num_trees):
            sample_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_sample, y_sample = X[sample_indices], y[sample_indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        predictions = np.array(predictions).T  # 每一列代表一个样本

        final_predictions = []
        for pred in predictions:
            # 去除 None 值
            pred = pred[pred != None]
            # 如果预测结果为空，使用默认值填充（例如，使用最常见的类别）
            if len(pred) == 0:
                final_predictions.append(0)  
                continue
            # 如果预测结果是非整数类型（如字符串），将其映射为整数
            if not np.issubdtype(pred.dtype, np.integer):
                unique_labels, pred = np.unique(pred, return_inverse=True)
            counts = np.bincount(pred)
            most_common_label = np.argmax(counts)
            # 将整数索引转换回原始标签
            if not np.issubdtype(pred.dtype, np.integer):
                most_common_label = unique_labels[most_common_label]

            final_predictions.append(most_common_label)

        return np.array(final_predictions)

#开始训练
train = RandomForest(num_trees=10, max_depth=3)
train.fit(X_train, y_train)

# 开始检验
test= train.predict(X_test)
accuracy = accuracy_score(y_test, test)
print(f'Accuracy: {accuracy:.2f}')
       