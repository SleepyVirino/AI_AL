import numpy as np

FLT_EPSILON = np.finfo(float).eps
PYTHON_MAX_RECURSION_DEPTH = 1000

class DecisionTreeRegressor:
    def __init__(self, max_depth=PYTHON_MAX_RECURSION_DEPTH, criterion="mse"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None
        self.n_features = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self.build_tree(X, y, 0)

    def predict(self, X):
        predictions = np.array([self.predict_one(x) for x in X])
        return predictions

    def predict_one(self, x):
        node = self.tree
        while isinstance(node, dict):
            if x[node["feature"]] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        return node

    def sample_equal(self, X):
        if len(X) == 0:
            return True
        temp = X[0]
        for x in X:
            if any(x != temp):
                return False
        else:
            return True

    def build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or self.sample_equal(X):
            return np.mean(y)
        if depth >= self.max_depth:
            return np.mean(y)

        best_feature, best_thresh = self.find_best_split(X, y)
        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature, best_thresh)

        if len(left_X) > 0 and len(right_X) > 0:
            left_tree = self.build_tree(left_X, left_y, depth+1)
            right_tree = self.build_tree(right_X, right_y, depth + 1)
        else:
            left_tree = np.mean(y)
            right_tree = np.mean(y)


        return {"feature": best_feature, 'thresh': best_thresh, 'left': left_tree, "right": right_tree}

    def find_best_split(self, X, y):

        best_feature = None
        best_thresh = None

        if self.criterion == "mse":
            best_mse = np.inf
            for feature in range(self.n_features):
                feature_vals = np.unique(X[:, feature])
                for val in feature_vals:
                    left_indices = np.where(X[:, feature] <= val)
                    right_indices = np.where(X[:, feature] > val)
                    mse_result = self.mse_delta(y, left_indices, right_indices)

                    if mse_result < best_mse:
                        best_mse = mse_result
                        best_feature = feature
                        best_thresh = val
        else:
            raise Exception("Unknown criterion! Please use 'mse'")
        return best_feature, best_thresh

    def mse_delta(self, y, left_indices, right_indices):
        left_indices = left_indices[0]
        right_indices = right_indices[0]
        p = len(left_indices) / len(y)
        return p * self.mse(y[left_indices]) + (1 - p) * self.mse(y[right_indices])

    def mse(self, y):
        if len(y) == 0:
            return 0
        else:
            return np.var(y)

    def split_data(self, X, y, feat, thresh):
        left_indices = np.where(X[:, feat] <= thresh)[0]
        right_indices = np.where(X[:, feat] > thresh)[0]
        left_X = X[left_indices]
        left_y = y[left_indices]
        right_X = X[right_indices]
        right_y = y[right_indices]
        return left_X, left_y, right_X, right_y

