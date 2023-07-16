import numpy as np

FLT_EPSILON = np.finfo(float).eps


class NaiveBayesClassifier():
    def __init__(self, discrete_features=False):
        self.classes = None
        self.class_priors = None
        self.discrete_features = discrete_features

    def fit(self, X, y):
        # Get all unique class labels
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # Compute the prior probability
        self.class_priors = np.zeros(num_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.sum(y==c)/len(y)

        # Compute the likelihood
        self.likelihood = dict()
        for i, c in enumerate(self.classes):
            X_C = X[y==c]
            likelihood_c = dict()
            for j in range(X.shape[1]):
                likelihood_c_i = dict()
                if self.discrete_features:
                    vals = np.unique(X_C[:,j])
                    for v in vals:
                        likelihood_c_i[v] = (np.sum(X_C[:,j]==v)+1)/(len(X_C)+len(vals))
                else:
                    likelihood_c_i['mean'] = np.average(X_C[:,j])
                    likelihood_c_i['variance'] = np.var(X_C[:,j])
                likelihood_c[j] = likelihood_c_i
            self.likelihood[c] = likelihood_c

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


    def predict_one(self, x):
        posterior_probs = np.zeros(len(self.classes))
        for i, c in enumerate(self.classes):
            posterior_probs[i] = self.class_priors[i]
            for j in range(len(x)):
                if self.discrete_features:
                    posterior_probs[i] *= self.likelihood[c][j].get([x[j]],1)
                else:
                    posterior_probs[i] *= self.gaussian_pdf(x[j], self.likelihood[c][j]['mean'], self.likelihood[c][j]['variance'])
        return self.classes[np.argmax(posterior_probs)]

    def gaussian_pdf(self, x, mean, variance):
        return 1/np.sqrt(2*np.pi*variance+FLT_EPSILON)*np.exp(-(x-mean)**2/(2*variance+FLT_EPSILON))



class GaussianBayesClassfier():

    def __init__(self, class_prob_eq=True):
        self.classes = None
        self.class_priors = None
        self.feature_means = None
        self.feature_variances = None
        self.class_prob_eq = class_prob_eq

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
                variance_val_c = FLT_EPSILON
            for s in range(num_samples):
                bias = X[s]-means_c
                tmp = np.dot(bias,variance_c_1)
                tmp = np.dot(tmp,bias.T)
                result = np.log(variance_val_c)+tmp
                if not self.class_prob_eq:
                    result -= np.log(self.class_priors[i])
                posterior_probs[s][i] = result
        return self.classes[np.argmin(posterior_probs, axis=1)]






