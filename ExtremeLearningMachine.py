import numpy as np


class ExtremeLearningMachine(object):
    def __init__(self, n_unit, activation=None):
        self._activation = self._sig if activation is None else activation
        self._n_unit = n_unit
        self.inputX = None
        self.X = None
        self.a = None
        self.b = None
        self.beta = None

    @staticmethod
    def _sig(x):
        return 1. / (1 + np.exp(-x))

    def fit(self, inputX, inputY):
        self.X = np.hstack((np.ones((inputX.shape[0],1)), inputX))
        self.a = np.random.random((self._n_unit, self.X.shape[1]))
        self.b = np.random.random((self._n_unit, 1))
        H = self._activation(self.X.dot(self.a.T) + self.b.T)
        self.beta = np.linalg.pinv(H).dot(inputY)
        return self.beta

    def transform(self, inputX):
        if not hasattr(self, 'a'):
            raise UnboundLocalError('must fit before transform')
        self.X = np.hstack((np.ones((inputX.shape[0],1)), inputX))
        H = self._activation(self.X.dot(self.a.T) + self.b.T)
        return H.dot(self.beta)


class EnsembleELM(object):
    def __init__(self, ensemble, n_unit, activation=None):
        self.elm = [ExtremeLearningMachine(n_unit, activation=activation) for i in range(ensemble)]
        self.beta = np.zeros(ensemble)
        self.y_train = np.zeros(ensemble)