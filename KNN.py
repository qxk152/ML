import numpy as np
from math import sqrt
from collections import Counter
from metrics import accuracy_score
#x是新的点集
def kNN_classify(k,X_train,y_train,x):
    assert 1 <= k <= X_train.shape[0],"k must be valid "
    assert X_train.shape[0] == y_train.shape[0] ,\
    "the size of X_train must equall y_train"
    assert X_train.shape[1] == x.shape[0], \
    "the features of X_train must equall x"
    dis = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    #排序
    nearest = np.argsort(dis)
    #找到距离最小的k个 遍历每一个节点的索引并找到其对应的 类别
    topK_y = [y_train[i] for i in nearest[:k]]
    return Counter(topK_y).most_common(1)[0][0]

class KNNClassifier:
    def __init__(self,k):
        # 初始化knn分类器
        assert 1 <= k, "k must be valid "
        self.k = k
        #需要接受一个数据训练集
        self.__X_train = None
        self.__ytrain = None #  _X 前的_ 代表私有
    def fit(self,X_train,y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."
        self.__X_train = X_train
        self.__y_train = y_train  # _ 代表私有

    def predict(self,X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.__X_train is not None and self.__y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self.__X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        y_predict = [self.__predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)
    #私有方法
    def __predict(self,x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] == self.__X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        dis = [sqrt(np.sum((x - x_train)**2)) for x_train in self.__X_train ]

        #排序
        nearest = np.argsort(dis)
        #找到前k个最小的索引 根据索引 得到对应的值的列表
        topK_y = [self.__y_train[i] for i in nearest[:self.k]]
        return  Counter(topK_y).most_common(1)[0][0]
    def score(self,X_test,y_test):
        #根据传入的X_test y_test 计算模型预测的准确率
        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    # 将对象转化为供解释器读取的形式
    def __repr__(self):
        return "KNN(k=%d)" % self.k