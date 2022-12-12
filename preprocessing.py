import numpy as np
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None;
    def fit(self,X):
        #求出X的每一个特征的均值和方差
        assert X.ndim == 2,"The dimension of X must be 2"
        self.mean_ = [np.mean(X[:,i]) for i in range(X.shape[1])]
        self.scale_ = [np.std(X[:, i]) for i in range(X.shape[1])]

    def transform(self,X):
        """进行均值方差归一化"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None ,\
        "must be fit first"
        assert X.shape[1] == len(self.mean_)," X.shape[1] == len(self.mean_)"
        #一个空的矩阵
        resX = np.empty(shape= X.shape,dtype =float)
        for col in range(resX.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col])/self.scale_[col]
        return resX
