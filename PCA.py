import numpy as np
class PCA:
    def __init__(self,n_components):
        """初始化PCA"""
        assert n_components >= 1,"n_component must be vaild"
        # 几个主成分
        self.n_components = n_components
        # 用户传来的数据计算出来的结果 存储主成分
        self.components_ = None
    #用来计算pca的主成分
    def fit(self,X,eta = 0.01,n_iters = 1e4):
        """获得数据集X的前n个主成分"""
        #X的特征数必须大于 想要求得的主成分个数
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            cur_iters = 0
            w = direction(initial_w)
            while cur_iters < n_iters:
                gradient = df(w,X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w,X) - f(last_w,X)) < epsilon):
                    break
                cur_iters+=1
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            init_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca,init_w,eta,n_iters)
            self.components_[i,:] = w
            # reshape(-1, 1) 就是每行一个数 * w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self
    def transform(self,X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    #自定义输出实例化对象时的信息
    def __repr__(self):
        #  %是占位符 类似于c语言 print
        return "PCA(n_compnents = %d)" % self.n_components
