import numpy as np


class KnnRegressor():

    def __init__(self, k=5, distance="L2"):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X = X
        self.y = y

        if len(self.X) < self.k:
            raise Exception("k must be less than or equal to the number of training examples")

    def predict(self, X):

        test_size = len(X)
        pred = np.zeros(test_size,dtype=int)


        if self.distance=="L2":
            dist = self.l2_distance(X,self.X)
        else:
            raise Exception("Please use L2 distance")
        sorted_index = np.argsort(dist, axis=1)
        k_nearest = sorted_index[:, :self.k]
        for t in range(test_size):
            val = np.mean(self.y[k_nearest][t])
            pred[t] = val

        return pred

    def l2_distance(self, X1, X2):
        X1_2 = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_2 = np.sum(X2**2, axis=1).reshape(1, -1)
        X1_X2 = np.dot(X1, X2.T)
        return np.sqrt(X1_2 + X2_2 - 2*X1_X2)