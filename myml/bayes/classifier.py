import numpy as np

FLT_EPSILON = 1.192092896e-07


class GaussianBayesClassfier():

    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.feature_means = None
        self.feature_variances = None

    def fit(self, X, y):
        # Get all unique class labels
        self.classes = np.unique(y)
        num_classes = len(self.classes)


        self.feature_means = np.zeros((num_classes, X.shape[1]))
        self.feature_variances = np.zeros((num_classes, X.shape[1], X.shape[1]))
        self.class_priors = np.zeros(num_classes)


        # For every class, compute feature_means and feature_variances
        for i, c in enumerate(self.classes):
            X_C = X[y==c]

            # Compute the feature_means
            self.feature_means[i] = np.average(X_C, axis=0)
            # Compute the feature_variances
            self.feature_variances[i] =np.cov(X_C, rowvar=False)
            # Compute the prior probability
            self.class_priors[i] = (np.sum(y==c)+1)/(len(y)+num_classes)


    def predict(self, X):
        num_samples, num_features = X.shape
        num_classes = len(self.classes)

        posterior_probs = np.zeros((num_samples,num_classes))

        for i, c in enumerate(self.classes):

            means_c = self.feature_means[i]
            variance_c = self.feature_variances[i]
            variance_c_1 = np.linalg.pinv(variance_c)
            variance_val_c = np.linalg.det(variance_c)
            if variance_val_c <= 0:
                variance_val_c = np.finfo(float).eps
            for s in range(num_samples):
                bias = X[s]-means_c
                tmp = np.dot(bias,variance_c_1)
                tmp = np.dot(tmp,bias.T)
                result = np.log(variance_val_c)+tmp+FLT_EPSILON

                posterior_probs[s][i] = result
        return self.classes[np.argmin(posterior_probs, axis=1)]






