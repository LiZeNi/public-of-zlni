import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from math import log
import operator

# 加载 Iris 数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#定义决策树
def CreatTree(dataset, labels, featlabels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):  # 所有类别相同
        return classlist[0]
    if len(dataset[0]) == 1:  # 到叶子了
        return majorityCnt(classlist)
    
    bestfeat = BestFeature(dataset)  # 返回的是特征索引
    bestfeatlabel = labels[bestfeat]  # 获取对应的特征名称
    featlabels.append(bestfeatlabel)
    myTree = {bestfeatlabel: {}}  # 使用特征名称作为键
    sublabels = labels[:]
    np.delete(sublabels,bestfeat)  # 删除已使用的特征
    featvalues = [example[bestfeat] for example in dataset]
    uniquevalues = set(featvalues)  
    for value in uniquevalues:
        myTree[bestfeatlabel][value] = CreatTree(
            splitdataset(dataset, bestfeat, value), sublabels, featlabels)
    
    return myTree


# 计算多数类别
def majorityCnt(classlist):
    classCount = {}
    for value in classlist:
        if value not in classCount.keys():
            classCount[value] = 0
        classCount[value] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 计算熵值得出最佳节点
def BestFeature(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntroy = calculateEntroy(dataset)
    bestInformationGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featlist = [example[i] for example in dataset]
        uniquevalues = set(featlist)
        newEntroy = 0
        for value in uniquevalues:
            subDataset = splitdataset(dataset, i, value)
            prob = len(subDataset) / float(len(dataset))
            newEntroy += prob * calculateEntroy(subDataset)
        informationGain = baseEntroy - newEntroy
        if informationGain > bestInformationGain:
            bestInformationGain = informationGain
            bestFeature = i
    return bestFeature

# 取出当前特征每种取值的数据
def splitdataset(dataset, i, value):
    subdataset = []
    for example in dataset:
        if example[i] == value:
            subexample = np.concatenate((example[:i], example[i+1:]))
            subdataset.append(subexample)
    return subdataset

# 计算当前节点的熵值
def calculateEntroy(dataset):
    numexamples = len(dataset)
    labelCounts = {}
    for example in dataset:
        if example[-1] not in labelCounts.keys():
            labelCounts[example[-1]] = 0
        labelCounts[example[-1]] += 1

    Entroy = 0
    for key in labelCounts:
        prop = labelCounts[key] / numexamples
        Entroy -= prop * log(prop, 2)
    return Entroy

# 定义决策树
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.BuildTree(X, y)

    def BuildTree(self, X, y):
        featlabels = list(iris.feature_names)  # 使用特征名称列表
        return CreatTree(X, y, featlabels)



    def predict(self, X):
        return [self.Predict(inputs) for inputs in X]
    
    def Predict(self, inputs): 
        tree = self.tree
        while isinstance(tree, dict):
            feature = int(next(iter(tree)))  # 获取特征名称
            if feature>3:
                feature=3
            if feature==0:
                name='sepal length (cm)'
            elif feature ==1:
                name='sepal width (cm)'
            elif feature==2:
                name= 'petal length (cm)'
            else :
                name='petal width (cm)'

            value = inputs[iris.feature_names.index(name)]  # 根据特征名称获取输入值
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
        self.feature_importances_ = None
        self.feature_name_to_index = {name: idx for idx, name in enumerate(iris.feature_names)}  # 特征名称到索引的映射

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])  # 根据特征数量初始化特征重要性数组
        for _ in range(self.num_trees):
            sample_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_sample, y_sample = X[sample_indices], y[sample_indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        # 计算特征重要性
        self.compute_feature_importances()

    def compute_feature_importances(self):
        for tree in self.trees:
            self.update_feature_importances(tree.tree)

    def update_feature_importances(self, tree, importance=1.0):
        if isinstance(tree, dict):
            feature_index = int(next(iter(tree)))  # 获取特征名称
            if feature_index>3:
                feature_index=3
            self.feature_importances_[feature_index] += importance
            for subtree in tree.values():
                self.update_feature_importances(subtree, importance=importance / len(tree))

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

# 开始训练
train = RandomForest(num_trees=10, max_depth=3)
train.fit(X_train, y_train)

# 开始检验
y_pred = train.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 特征重要性评估
feature_importances = train.feature_importances_
print("Feature Importances:", feature_importances)

# 特征重要性可视化
plt.figure(figsize=(10, 6))
plt.bar(iris.feature_names, feature_importances, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45)
plt.show()

# 验证集预测结果可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
