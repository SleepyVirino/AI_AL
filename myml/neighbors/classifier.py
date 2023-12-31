import numpy as np
from metric_learn import LMNN

class KnnClassifier():

    def __init__(self, k=5, distance="L2"):
        self.k = k
        self.distance = distance
        self.lmnn = LMNN() if distance=="mahala" else None

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.lmnn:
            self.lmnn.fit(X,y)
        if len(self.X) < self.k:
            raise Exception("k must be less than or equal to the number of training examples")

    def predict(self, X):

        test_size = len(X)
        pred = np.zeros(test_size,dtype=int)

        if self.lmnn:
            X_embedded = self.lmnn.transform(X)
            X_embedded_train = self.lmnn.transform(self.X)
            dist = self.l2_distance(X_embedded,X_embedded_train)
        elif self.distance=="L2":
            dist = self.l2_distance(X,self.X)
        else:
            raise Exception("Please use mahala or L2 distance")
        sorted_index = np.argsort(dist, axis=1)
        k_nearest = sorted_index[:, :self.k]
        print(k_nearest)
        print(self.y[k_nearest])
        for t in range(test_size):
            max_index = np.argmax(np.bincount(self.y[k_nearest][t]))
            pred[t] = max_index

        return pred

    def l2_distance(self, X1, X2):
        X1_2 = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_2 = np.sum(X2**2, axis=1).reshape(1, -1)
        X1_X2 = np.dot(X1, X2.T)
        return np.sqrt(X1_2 + X2_2 - 2*X1_X2)


