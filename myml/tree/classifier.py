import numpy as np

FLT_EPSILON = np.finfo(float).eps
PYTHON_MAX_RECURSION_DEPTH = 1000

class DecisionTreeClassifier:

    def __init__(self, criterion='gini',max_depth=PYTHON_MAX_RECURSION_DEPTH):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.tree = self.build_tree(X, y, 1)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

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
            return np.argmax(np.bincount(y))
        if depth >= self.max_depth:
            return np.argmax(np.bincount(y))

        best_feature, best_thresh = self.find_best_split(X, y)
        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature, best_thresh)

        if len(left_X) > 0 and len(right_X) > 0:
            left_tree = self.build_tree(left_X, left_y, depth+1)
            right_tree = self.build_tree(right_X, right_y, depth + 1)
        else:
            left_tree = np.argmax(np.bincount(y))
            right_tree = np.argmax(np.bincount(y))

        return {"feature": best_feature, 'thresh': best_thresh, 'left': left_tree, "right": right_tree}

    def find_best_split(self, X, y):

        best_feature = None
        best_thresh = None

        if self.criterion == "gini":
            best_gini = 1
            for feature in range(self.n_features):
                feature_vals = np.unique(X[:, feature])
                for val in feature_vals:
                    left_indices = np.where(X[:, feature] <= val)
                    right_indices = np.where(X[:, feature] > val)
                    gini_result = self.gini_index(y, left_indices, right_indices)

                    if gini_result < best_gini:
                        best_gini = gini_result
                        best_feature = feature
                        best_thresh = val

        elif self.criterion == "gain":
            best_gain = 0
            for feature in range(self.n_features):
                feature_vals = np.unique(X[:, feature])
                for val in feature_vals:
                    left_indices = np.where(X[:, feature] <= val)
                    right_indices = np.where(X[:, feature] > val)
                    gain_result = self.gain(y, left_indices, right_indices)

                    if gain_result > best_gain:
                        best_gain = gain_result
                        best_feature = feature
                        best_thresh = val

        elif self.criterion == "gain_ratio":
            best_gain_ratio = 0
            for feature in range(self.n_features):
                feature_vals = np.unique(X[:, feature])
                for val in feature_vals:
                    left_indices = np.where(X[:, feature] <= val)
                    right_indices = np.where(X[:, feature] > val)
                    gain_ratio_result = self.gain_ratio(y, left_indices, right_indices)

                    if gain_ratio_result > best_gain_ratio:
                        best_gain_ratio = gain_ratio_result
                        best_feature = feature
                        best_thresh = val
        else:
            raise Exception("Unknown criterion! Please use 'gini', 'gain' or 'gain_ratio'")

        return best_feature, best_thresh

    def gain_ratio(self, y, left_indices, right_indices):
        return self.gain(y, left_indices, right_indices) / self.iv(y, left_indices, right_indices)

    def iv(self, y, left_indices, right_indices):
        left_indices = left_indices[0]
        right_indices = right_indices[0]
        p1 = len(left_indices) / len(y) + FLT_EPSILON
        p2 = len(right_indices) / len(y) + FLT_EPSILON
        return -(p1 * np.log2(p1) + p2 * np.log2(p2))

    def gain(self, y, left_indices, right_indices):
        left_indices = left_indices[0]
        right_indices = right_indices[0]
        p = len(left_indices) / len(y)

        entropy_left = self.entropy(y[left_indices])
        entropy_right = self.entropy(y[right_indices])
        entropy_root = self.entropy(y)
        return entropy_root - (p * entropy_left + (1 - p) * entropy_right)

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y) + FLT_EPSILON
        return -np.sum(probabilities * np.log2(probabilities))

    def gini_index(self, y, left_indices, right_indices):
        left_indices = left_indices[0]
        right_indices = right_indices[0]
        p = len(left_indices) / len(y)
        gini_left = self.gini(y[left_indices])
        gini_right = self.gini(y[right_indices])
        return p * gini_left + (1 - p) * gini_right

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y) + FLT_EPSILON
        return 1 - np.sum(probabilities ** 2)

    def split_data(self, X, y, feat, thresh):
        left_indices = np.where(X[:, feat] <= thresh)[0]
        right_indices = np.where(X[:, feat] > thresh)[0]
        left_X = X[left_indices]
        left_y = y[left_indices]
        right_X = X[right_indices]
        right_y = y[right_indices]
        return left_X, left_y, right_X, right_y





