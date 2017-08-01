import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging

class GaussianProcess:
    def __init__(self, numFidelities, dim):
        # Use an anisotropic kernel
        # (independent length scales for each dimension)
        sqrdExp = ConstantKernel() ** 2. * RBF(length_scale=dim*[1.])
        numHyperParams = dim + 1
        self.models, self.isFit, self.xValues, self.yValues = [], [], [], []
        for _ in range(numFidelities):
            self.models.append(GaussianProcessRegressor(
                                    kernel=sqrdExp,
                                    n_restarts_optimizer=numHyperParams*10))
            self.isFit.append(False)
            self.xValues.append([])
            self.yValues.append([])

    def isValid(self, fidelity):
        return len(self.xValues[fidelity]) >= 2

    def fitModel(self, fidelity):
        if self.isValid(fidelity) and not self.isFit[fidelity]:
            x = np.atleast_2d(self.xValues[fidelity])
            y = np.array(self.yValues[fidelity]).reshape(-1, 1)
            self.models[fidelity].fit(x, y)
            self.isFit[fidelity] = True

    def addSample(self, x, y, fidelity):
        self.xValues[fidelity].append(x)
        self.yValues[fidelity].append(y)
        self.isFit[fidelity] = False

    def getPrediction(self, x, fidelity):
        self.fitModel(fidelity)
        mean, std = self.models[fidelity].predict(x.reshape(1, -1),
                                                  return_std=True)
        return mean[0][0], std[0]
