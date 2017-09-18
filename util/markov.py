"""
Markov Predictors:
Input/Output: 0,1,2,3,4,...
"""
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels, _check_partial_fit_first_call
from sklearn.metrics import accuracy_score
import numpy as np

class BaseMarkovPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, order):
        self.order = order
        self.encoder = None
        self.transition_ = None
        self._validate_params()

    def _validate_params(self):
        if not self.order > 0:
            raise ValueError('order muse be >0')

    def _validate_input(self, X, y):
        X, y = check_X_y(X, y, ensure_min_features=self.order)
        if not X.shape[1] == self.order:
            raise ValueError('Columns of X must be equal to order')
        return X, y

    def _validate_input_X(self, X):
        X = check_array(X, ensure_min_features=self.order)
        if not X.shape[1] == self.order:
            raise ValueError('Columns of X must be equal to order')
        return X

    def _transform(self, np_array):
        shape = np_array.shape
        return self.encoder.transform(np_array.reshape(-1, )).reshape(shape)

    def _inverse_transform(self, np_array):
        shape = np_array.shape
        return self.encoder.inverse_transform(np_array.reshape(-1, )).reshape(shape)

    def _partial_fit(self, X, y, classes=None, _refit=False):
        X, y = self._validate_input(X, y)

        if _refit:
            self.classes_ = None

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            n_classes = len(self.classes_)
            self.transition_counts_ = np.zeros(
                [n_classes for i in range(self.order + 1)])
            encoder = LabelEncoder().fit(self.classes_)
            self.encoder = encoder

        # convert any classess to int labels: 0,1,2,...
        X = self._transform(X)
        y = self._transform(y)

        n_samples = X.shape[0]

        for i in range(n_samples):
            self.transition_counts_[tuple(X[i])][y[i]] += 1.

        return self._build_transition()

    def _build_transition(self):
        raise NotImplemented('Should Implement Build Transition Method!')

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        labels_X = np.unique(X.reshape(-1, ))
        labels_y = np.unique(y)
        classes = np.union1d(labels_X, labels_y)
        return self._partial_fit(X, y, _refit=True, classes=classes)

    def partial_fit(self, X, y, classes=None):
        return self._partial_fit(X, y, classes=classes)

    def predict(self, X):
        X = self._validate_input_X(X)
        probabilities = self._predict_likelihood(X)
        n_samples = probabilities.shape[0]
        y = [None for i in range(n_samples)]
        for i in range(n_samples):
            counts = probabilities[i, :]
            if np.sum(counts) > 0:
                y[i] = np.argmax(counts)
            else:
                x = self._transform(X[i, :])
                y[i] = x[-1]

        y = np.array(y)
        y = self._inverse_transform(y)
        return y

    def _predict_likelihood(self, X):
        X = self._validate_input_X(X)

        X = self._transform(X)
        probabilities = []
        for i in range(X.shape[0]):
            pos = tuple(X[i])
            probabilities.append(self.transition_[pos])

        return np.array(probabilities)

    def dynamic_score(self, X, y):
        X, y = self._validate_input(X, y)
        n_samples = X.shape[0]
        y_predict = [None for i in range(n_samples)]
        for i in range(n_samples):
            y_predict[i] = self.predict([X[i]])[0]
            self.partial_fit([X[i]], [y[i]])
        return accuracy_score(y_predict, y)


class MaximumLikelihoodMarkovPredictor(BaseMarkovPredictor):
    def _build_transition(self):
        self.transition_ = self.transition_counts_
        # print self.transition_
        return self

def timeseries2XY(time_series, order, successive=True, sep=None):
    X = []
    Y = []

    N = len(time_series)
    for i in range(order, N):
        if not successive and (i % sep) < order:
            continue
        x = time_series[i - order:i]
        y = time_series[i]
        X.append(x)
        Y.append(y)
    return X, Y