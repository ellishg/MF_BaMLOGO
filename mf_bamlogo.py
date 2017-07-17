import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from copy import deepcopy
import logging

class ObjectiveFunction:
    def __init__(self, fn, costs, lows, highs):
        assert len(lows) == len(highs)

        self.dim = len(lows)
        self.maxFidelity = len(costs) - 1
        self.fn = fn
        self.costs = np.array(costs)
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.numObservations = np.zeros_like(self.costs)

    def evaluate(self, x, fidelity):
        self.numObservations[fidelity] = self.numObservations[fidelity] + 1
        args = tuple(x * (self.highs - self.lows) + self.lows)
        y = self.fn(args, fidelity)
        logging.debug('Evaluating function at fidelity {0}: f{1} = {2}'
                            .format(fidelity, args, y))
        return y

    def totalCost(self):
        return np.sum(self.numObservations * self.costs)

class MF_BaMLOGO:

    class GaussianProcess:
        def __init__(self, numFidelities, dim):
            # Use an anisotropic kernel
            # (independent length scales for each dimension)
            sqrdExp = ConstantKernel() ** 2. * RBF(length_scale=dim*[1.])
            numHyperParams = dim + 1
            self.models, self.isFit, self.data = [], [], []
            for _ in range(numFidelities):
                self.models.append(GaussianProcessRegressor(
                                    kernel=sqrdExp,
                                    n_restarts_optimizer=numHyperParams*10))
                self.isFit.append(False)
                self.data.append(dict())

        def isValid(self, fidelity):
            return len(self.data[fidelity]) >= 2

        def fitModel(self, fidelity):
            if self.isValid(fidelity):
                x = np.atleast_2d(list(self.data[fidelity].keys()))
                y = np.array(list(self.data[fidelity].values())).reshape(-1, 1)
                self.models[fidelity].fit(x, y)
                self.isFit[fidelity] = True

        def addSample(self, x, y, fidelity):
            self.data[fidelity][tuple(x)] = y
            self.isFit[fidelity] = False

        def getPrediction(self, x, fidelity):
            if not self.isFit[fidelity]:
                self.fitModel(fidelity)
            mean, std = self.models[fidelity].predict(x.reshape(1, -1),
                                                      return_std=True)
            return mean[0][0], std[0]

    class Node:
        def __init__(self, lows, highs, depth):
            self.lows = np.array(lows)
            self.highs = np.array(highs)
            self.center = (self.lows + self.highs) / 2.
            self.value = None
            self.fidelity = None
            self.isFakeValue = False
            self.depth = depth

        def setFidelity(self, value, fidelity):
            self.value = value
            self.fidelity = fidelity
            self.isFakeValue = False

        def setFakeValue(self, fakeValue):
            self.value = fakeValue
            self.fidelity = None
            self.isFakeValue = True

        def split(self):
            lengths = self.highs - self.lows
            longestDimension = np.argmax(lengths)
            logging.debug('Splitting node at x={0} along axis {1}'
                            .format(tuple(self.center), longestDimension))
            t = lengths[longestDimension] / 3.
            lowerThird = self.lows[longestDimension] + t
            upperThird = self.highs[longestDimension] - t
            listOfLows = [deepcopy(self.lows) for _ in range(3)]
            listOfHighs = [deepcopy(self.highs) for _ in range(3)]
            listOfHighs[0][longestDimension] = lowerThird   # Left node
            listOfLows[1][longestDimension] = lowerThird    # Center node
            listOfHighs[1][longestDimension] = upperThird   # Center node
            listOfLows[2][longestDimension] = upperThird    # Right node
            newNodes = [MF_BaMLOGO.Node(listOfLows[i], listOfHighs[i],
                                        self.depth + 1) for i in range(3)]
            newNodes[1].value = self.value
            newNodes[1].fidelity = self.fidelity
            newNodes[1].isFakeValue = self.isFakeValue
            return newNodes

    class Space:
        def __init__(self, dim):
            self.nodes = [MF_BaMLOGO.Node([0.] * dim, [1.] * dim, depth=0)]

        def maxDepth(self):
            return max([n.depth for n in self.nodes])

        def bestNodeInRange(self, level, width):
            depthRange = range(width * level, width * (level + 1))

            def inRange(n):
                return n[1].depth in depthRange

            nodesInLevel = list(filter(inRange, enumerate(self.nodes)))
            if not nodesInLevel:
                return None, None
            return max(nodesInLevel, key=lambda n: n[1].value)

        def expandAt(self, index):
            node = self.nodes.pop(index)
            newNodes = node.split()
            self.nodes.extend(newNodes)

    def __init__(self, objectiveFunction,
                        initNumber=10, algorithm='MF-BaMLOGO'):
        assert algorithm in ['MF-BaMLOGO', 'BaMLOGO', 'LOGO']
        self.algorithm = algorithm
        self.wSchedule = [3, 4, 5, 6, 8, 30]
        self.objectiveFunction = objectiveFunction
        self.dim = objectiveFunction.dim
        self.maxFidelity = objectiveFunction.maxFidelity
        self.numFidelities = self.maxFidelity + 1
        self.numExpansions = 0
        self.wIndex = 0
        self.stepBestValue = -float('inf')
        self.lastBestValue = -float('inf')
        self.bestNode = None
        self.gp = self.GaussianProcess(self.numFidelities, self.dim)

        self.epsilon = 0
        if self.algorithm == 'MF-BaMLOGO':
            samples = []
            for i in range(initNumber):
                x = np.random.uniform([0.] * self.dim, [1.] * self.dim)
                y = self.objectiveFunction.evaluate(x, fidelity=0)
                samples.append(y)
                self.gp.addSample(x, y, fidelity=0)
                if i % 3 == 0:
                    y1 = self.objectiveFunction.evaluate(x, fidelity=1)
                    self.epsilon = max(self.epsilon, abs(y - y1))
                    samples.append(y1)
                    self.gp.addSample(x, y1, fidelity=1)

            r = 1e-6 * (max(samples) - min(samples))
            self.thresholds = (self.numFidelities - 1) * [r]
            self.timeSinceEval = np.zeros((self.numFidelities),)
            logging.debug('Thresholds initialized to {0}'.format(r))
            logging.debug('Epsilon initialzed to {0}'.format(self.epsilon))

        self.space = self.Space(self.dim)
        self.observeNode(self.space.nodes[0])

    def maximize(self, resources=100, ret_data=False):
        costs, values, queryPoints = [], [], []
        while self.objectiveFunction.totalCost() < resources:
            self.stepBestValue = -float('inf')
            self.expandStep()
            self.adjustW()

            if self.bestNode:
                cost = self.objectiveFunction.totalCost()
                x = tuple(self.bestNode.center)
                y = self.bestNode.value
                costs.append(cost)
                queryPoints.append(x)
                values.append(y)
                logging.info('Best value is {0} with cost {1}'.format(y, cost))

        if ret_data:
            return costs, values, queryPoints

    def maxLevel(self):
        depthWidth = self.wSchedule[self.wIndex]
        hMax = math.sqrt(self.numExpansions + 1)
        return math.floor(min(hMax, self.space.maxDepth()) / depthWidth)

    def expandStep(self):
        logging.debug('Starting expand step')
        vMax = -float('inf')
        depthWidth = self.wSchedule[self.wIndex]
        level = 0
        while level <= self.maxLevel():
            logging.debug('Expanding level {0}'.format(level))
            idx, bestNode = self.space.bestNodeInRange(level, depthWidth)
            if idx is not None and bestNode.value > vMax:
                vMax = bestNode.value
                logging.debug('vMax is now {0}'.format(vMax))
                self.space.expandAt(idx)
                self.observeNode(self.space.nodes[-3])  # Left node
                self.observeNode(self.space.nodes[-2])  # Center node
                self.observeNode(self.space.nodes[-1])  # Right node
                self.numExpansions = self.numExpansions + 1
            level = level + 1

    def observeNode(self, node):
        x = node.center
        if node.value is not None and not node.isFakeValue:
            if node.fidelity == self.maxFidelity:
                logging.debug('Already had node at x={0}'.format(tuple(x)))
                return
        lcb, ucb = self.computeLCBUCB(x)

        if ucb is None or self.bestNode is None or ucb >= self.bestNode.value:
            fidelity = self.chooseFidelity(node)
            self.evaluateNode(node, fidelity, offset=self.error(fidelity),
                                updateGP=True, adjustThresholds=True)

        elif self.algorithm == 'MF-BaMLOGO' and 2. * self.error(0) < ucb - lcb:
            logging.debug('Unfavorable region at x={0}; '
                          'Using lowest fidelity'.format(tuple(x)))
            self.evaluateNode(node, fidelity=0, offset=-self.error(0))

        else:
            logging.debug('Unfavorable region at x={0}. '
                          'Using LCB = {1}'.format(tuple(x), lcb))
            node.setFakeValue(lcb)

    def evaluateNode(self, node, fidelity,
                    offset=0., updateGP=False, adjustThresholds=False):
        x = node.center
        if node.value is not None and not node.isFakeValue:
            if fidelity <= node.fidelity:
                logging.debug('Already had node at x={0}'.format(tuple(x)))
                return

        y = self.objectiveFunction.evaluate(x, fidelity)
        node.setFidelity(y + offset, fidelity)

        self.stepBestValue = max(self.stepBestValue, y)
        logging.debug('Step best is now {0}'.format(self.stepBestValue))
        if fidelity == self.maxFidelity:
            if not self.bestNode or self.bestNode.value < y:
                self.bestNode = node

        if self.algorithm == 'MF-BaMLOGO' or self.algorithm == 'BaMLOGO':
            if updateGP:
                self.gp.addSample(x, y, fidelity)

        if self.algorithm == 'MF-BaMLOGO' and adjustThresholds:
            self.timeSinceEval[:fidelity+1] = 0
            self.timeSinceEval[fidelity+1:] =\
                            self.timeSinceEval[fidelity+1:] + 1
            for f in range(len(self.thresholds)):
                c = (self.objectiveFunction.costs[f+1]
                        / self.objectiveFunction.costs[f])
                if self.timeSinceEval[f+1] > c:
                    self.thresholds[f] = self.thresholds[f] * 2
                    logging.debug('Thresholds are now {0}'
                                        .format(self.thresholds))
                    break

        if self.algorithm == 'MF-BaMLOGO' and fidelity > 0:
            mean, _ = self.gp.getPrediction(x, fidelity - 1)
            if mean is None or abs(y - mean) > self.epsilon:
                lowFidelityY = self.objectiveFunction.evaluate(x, fidelity - 1)
                self.epsilon = max(self.epsilon, abs(y - lowFidelityY))
                logging.debug('Epsilon is now {0}'.format(self.epsilon))

    def chooseFidelity(self, node):
        if self.algorithm == 'MF-BaMLOGO':
            x = node.center
            beta = self.beta()

            for fidelity in range(self.numFidelities - 1):
                if not self.gp.isValid(fidelity):
                    logging.debug('Choosing fidelity {0} because of invalid GP'
                                        .format(fidelity))
                    return fidelity

                _, std = self.gp.getPrediction(x, fidelity)
                if beta * std >= self.thresholds[fidelity]:
                    logging.debug('Choosing fidelity {0}'.format(fidelity))
                    return fidelity

            logging.debug('Choosing highest fidelity')
            return self.maxFidelity
        else:
            return self.maxFidelity

    def error(self, f):
        return self.epsilon * (self.maxFidelity - f)

    def beta(self):
        n = 0.5
        return math.sqrt(2. * math.log(
                math.pi ** 2. * (self.numExpansions + 1) ** 2. / (6. * n)))

    def computeLCBUCB(self, x):
        if self.algorithm == 'MF-BaMLOGO' or self.algorithm == 'BaMLOGO':
            beta = self.beta()

            def uncertainty(args):
                f, (_, std) = args
                return beta * std + self.error(f)

            predictions = []
            for fidelity in range(self.numFidelities):
                if self.gp.isValid(fidelity):
                    predictions.append(self.gp.getPrediction(x, fidelity))
            if not predictions:
                return None, None

            f, (mean, std) = min(enumerate(predictions), key=uncertainty)
            lcb, ucb = (mean - beta * std - self.error(f),
                        mean + beta * std + self.error(f))

            logging.debug('LCB/UCB for f{0} (fidelity {1})'.format(tuple(x), f))
            logging.debug('Mean={0}, std={1}, beta={2}'.format(mean, std, beta))
            logging.debug('LCB={0}, UCB={1}'.format(lcb, ucb))

            return lcb, ucb
        else:
            return None, None

    def adjustW(self):
        if self.stepBestValue > self.lastBestValue:
            self.wIndex = min(self.wIndex + 1, len(self.wSchedule) - 1)
        else:
            self.wIndex = max(self.wIndex - 1, 0)
        self.lastBestValue = self.stepBestValue
        logging.debug('Width is now {0}'.format(
                            self.wSchedule[self.wIndex]))

    def bestQuery(self):
        return tuple(self.bestNode.center), self.bestNode.value
