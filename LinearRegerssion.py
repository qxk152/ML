import numpy as np
from metrics import r2_score

from sklearn.preprocessing import StandardScaler
class LinearRegression:
    # X_b = Θ0 * 1 + Θ1X1 + ... ΘmXn = X_b * Θ
    def __int__(self):
        """初始化 多元线性回归 模型"""
        self.coeffient_ = None #系数
        self.intercept_ = None #截距
        self._theta = None # 系数

    def fit_normal(self,X_train,y_train):
        """多元线性回归根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # 1.首先把 X_train 变为 第一列为1的 矩阵X_b
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        # 2根据公式计算shita
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T.dot(y_train))
        self.intercept_ = self._theta[0]
        self.coeffient_ = self._theta[1:]
        return self
    def fit_gd(self,X_train,y_train,eta = 0.01,n_iters = 1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        #计算损失函数
        def J(theta,X_b,y):
            try:
                return np.sum((y - X_b.dot(theta))**2)/len(y)
            except:
                return float('inf')
        #计算损失函数对theta的梯度
        def dJ(theta,X_b,y):
            #res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1,len(theta)):
            #     res[i] = (X_b.dot(theta)-y).dot(X_b[:,i])
            res = X_b.T.dot(X_b.dot(theta) - y)
            return res * 2. / len(X_b)
        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon=1e-8):
            """沿着负梯度的方向不断进行搜索 直到两次的损失函数差小于epsilon"""
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        # (1 x1 x2... xn )(theta0)
        #                  theta1
        theta = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        self._theta = theta
        self.coeffient_ = theta[1:]
        self.intercept_ = theta[0]
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