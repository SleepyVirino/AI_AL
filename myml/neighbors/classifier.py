import numpy as np

class KnnClassifier():

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        if len(self.X) < self.k:
            raise Exception("k must be less than or equal to the number of training examples")

    def predict(self, X):

        test_size = len(X)
        pred = np.zeros(test_size,dtype=int)
        dist = self.l2_distance(X)
        sorted_index = np.argsort(dist, axis=1)
        k_nearest = sorted_index[:, :self.k]
        for t in range(test_size):
            max_index = np.argmax(np.bincount(self.y[k_nearest][t]))
            pred[t] = max_index

        return pred

    def l2_distance(self, X1):
        X1_2 = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_2 = np.sum(self.X**2, axis=1).reshape(1, -1)
        X1_X2 = np.dot(X1, self.X.T)
        return np.sqrt(X1_2 + X2_2 - 2*X1_X2)

if __name__ == "__main__":
    knn = KnnClassifier(2)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([7, 8, 9])
    knn.fit(X, y)
    X1 = np.array([[1, 2], [3, 4]])
    y = knn.predict(X1)
    print(y)
