import numpy as np


class DecisionTreeClassifier:

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.tree = self.build_tree(X, y)

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

    def build_tree(self, X, y):
        if len(np.unique(y)) == 1 or len(X[0]) == 0:
            return np.argmax(np.bincount(y))

        best_feature, best_thresh = self.find_best_split(X, y)
        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature, best_thresh)

        left_tree = self.build_tree(left_X, left_y)
        right_tree = self.build_tree(right_X, right_y)

        return {"feature": best_feature, 'thresh':best_thresh, 'left':left_tree, "right":right_tree}

    def find_best_split(self, X, y, criterion='gini'):


        best_feature = None
        best_thresh = None

        if criterion == "gini":
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
            
            pass
        elif criterion == "gain":
            pass
        elif criterion == "gain_ratio":
            pass
        else:
            raise Exception("Unknown criterion! Please use 'gini', 'gain' or 'gain_ratio'")


        return best_feature, best_thresh



    def gini_index(self, y, left_indices, right_indices):
        p = len(left_indices) / len(y)
        gini_left = self.gini(y[left_indices])
        gini_right = self.gini(y[right_indices])
        return p * gini_left + (1 - p) * gini_right

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def split_data(self, X, y, feat, thresh):
        left_indices = np.where(X[:, feat] <= thresh)[0]
        right_indices = np.where(X[:, feat] > thresh)[0]
        left_X = X[left_indices]
        left_y = y[left_indices]
        right_X = X[right_indices]
        right_y = y[right_indices]
        return left_X, left_y, right_X, right_y

