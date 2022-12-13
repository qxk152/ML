import numpy as np
from metrics import r2_score
class LinearRegression:
    # X_b = Θ0 * 1 + Θ1X1 + ... ΘmXn = X_b * Θ
    def __int__(self):
        """初始化 多元线性回归 模型"""
        self.coeffient_ = None #系数
        self.intercept_ = None #截距
        self._theta = None # 系数

    def fit_normal(self,X_train,y_train):
        """根据训练集 训练 线性回归模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # 1.首先把 X_train 变为 第一列为1的 矩阵X_b
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        # 2根据公式计算shita
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T.dot(y_train))
        self.intercept_ = self._theta[0]
        self.coeffient_ = self._theta[1:]
        return self
    def predict(self,X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coeffient_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coeffient_), \
            "the feature number of X_predict must be equal to X_train"
        # 1.首先把 X_train 变为 第一列为1的 矩阵X_b
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        y_predict = X_b.dot(self._theta)
        return y_predict

    def score(self,X_test,y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        #执行预测
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"