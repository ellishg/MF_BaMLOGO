import numpy as np
import math
import logging

class MFBaMLOGO:

    def __init__(self, fn, costEstimations, lows, highs,
                        initNumber=10, algorithm='MF-BaMLOGO'):
        assert algorithm in ['MF-BaMLOGO', 'BaMLOGO', 'LOGO']
        assert len(lows) == len(highs)
        self.algorithm = algorithm
        self.wSchedule = [3, 4, 5, 6, 8, 30]
        self.fn = fn
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.dim = len(self.lows)
        self.costs = costEstimations
        self.totalCost = 0.
        self.numFidelities = len(self.costs)
        self.maxFidelity = self.numFidelities - 1
        self.numExpansions = 0
        self.wIndex = 0
        self.stepBestValue = -float('inf')
        self.lastBestValue = -float('inf')
        self.bestNode = None
        from .model import GaussianProcess
        self.model = GaussianProcess(self.numFidelities, self.dim)

        self.epsilon = 0
        if self.algorithm == 'MF-BaMLOGO':
            samples = []
            for i in range(initNumber):
                x = np.random.uniform([0.] * self.dim, [1.] * self.dim)
                y = self.evaluate(x, 0)
                samples.append(y)
                self.model.addSample(x, y, 0)
                if i % 3 == 0:
                    y1 = self.evaluate(x, 1)
                    self.epsilon = max(self.epsilon, abs(y - y1))
                    samples.append(y1)
                    self.model.addSample(x, y1, fidelity=1)

            r = 1e-6 * (max(samples) - min(samples))
            self.thresholds = (self.numFidelities - 1) * [r]
            self.timeSinceEval = np.zeros((self.numFidelities),)
            logging.debug('Thresholds initialized to {0}'.format(r))
            logging.debug('Epsilon initialzed to {0}'.format(self.epsilon))

        from .partitiontree import PartitionTree
        self.space = PartitionTree(self.dim)
        self.observeNode(self.space.nodes[0])

    def maximize(self, budget=100., ret_data=False, plot=True):
        costs, bestValues, queryPoints = [], [], []
        while self.totalCost < budget:
            self.stepBestValue = -float('inf')
            self.expandStep()
            self.adjustW()

            if self.bestNode:
                cost = self.totalCost
                x = self.transformToDomain(self.bestNode.center)
                y = self.bestNode.value
                costs.append(cost)
                queryPoints.append(x)
                bestValues.append(y)
                logging.info('Best value is {0} with cost {1}'.format(y, cost))
            if plot and self.dim == 1:
                self.plotInfo()

        if ret_data:
            return costs, bestValues, queryPoints

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
                logging.debug('Already had node at x={0}'
                                .format(self.transformToDomain(x)))
                return
        lcb, ucb = self.computeLCBUCB(x)

        if ucb is None or self.bestNode is None or ucb >= self.bestNode.value:
            fidelity = self.chooseFidelity(node)
            self.evaluateNode(node, fidelity, offset=self.error(fidelity),
                                updateGP=True, adjustThresholds=True)

        elif self.algorithm == 'MF-BaMLOGO' and 2. * self.error(0) < ucb - lcb:
            logging.debug('Unfavorable region at x={0}; Using lowest fidelity'
                            .format(self.transformToDomain(x)))
            self.evaluateNode(node, fidelity=0, offset=-self.error(0))

        else:
            logging.debug('Unfavorable region at x={0}. Using LCB = {1}'
                            .format(self.transformToDomain(x), lcb))
            node.setFakeValue(lcb)

    def evaluateNode(self, node, fidelity,
                    offset=0., updateGP=False, adjustThresholds=False):
        x = node.center
        if node.value is not None and not node.isFakeValue:
            if fidelity <= node.fidelity:
                logging.debug('Already had node at x={0}'
                                .format(self.transformToDomain(x)))
                return

        y = self.evaluate(x, fidelity)
        node.setFidelity(y + offset, fidelity)

        self.stepBestValue = max(self.stepBestValue, y)
        logging.debug('Step best is now {0}'.format(self.stepBestValue))
        if fidelity == self.maxFidelity:
            if not self.bestNode or self.bestNode.value < y:
                self.bestNode = node

        if self.algorithm == 'MF-BaMLOGO' or self.algorithm == 'BaMLOGO':
            if updateGP:
                self.model.addSample(x, y, fidelity)

        if self.algorithm == 'MF-BaMLOGO' and adjustThresholds:
            self.timeSinceEval[:fidelity+1] = 0
            self.timeSinceEval[fidelity+1:] =\
                            self.timeSinceEval[fidelity+1:] + 1
            for f in range(len(self.thresholds)):
                c = (self.costs[f+1]
                        / self.costs[f])
                if self.timeSinceEval[f+1] > c:
                    self.thresholds[f] = self.thresholds[f] * 2
                    logging.debug('Thresholds are now {0}'
                                        .format(self.thresholds))
                    break

        if self.algorithm == 'MF-BaMLOGO' and fidelity > 0:
            mean, _ = self.model.getPrediction(x, fidelity - 1)
            if mean is None or abs(y - mean) > self.epsilon:
                lowFidelityY = self.evaluate(x, fidelity - 1)
                # TODO: Try updating the GP here
                self.epsilon = max(self.epsilon, abs(y - lowFidelityY))
                logging.debug('Epsilon is now {0}'.format(self.epsilon))


    def evaluate(self, x, f):
        args = self.transformToDomain(x)
        logging.debug('Evaluating f{0} at fidelity {1}'.format(args, f))
        y, cost = self.fn(args, f)
        logging.debug('Got y = {0} with cost {1}'.format(y, cost))
        self.totalCost += cost
        return y

    def chooseFidelity(self, node):
        if self.algorithm == 'MF-BaMLOGO':
            x = node.center
            beta = self.beta()

            for fidelity in range(self.numFidelities - 1):
                if not self.model.isValid(fidelity):
                    logging.debug('Choosing fidelity {0} because of invalid GP'
                                        .format(fidelity))
                    return fidelity

                _, std = self.model.getPrediction(x, fidelity)
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
                if self.model.isValid(fidelity):
                    predictions.append(self.model.getPrediction(x, fidelity))
            if not predictions:
                return None, None

            f, (mean, std) = min(enumerate(predictions), key=uncertainty)
            lcb, ucb = (mean - beta * std - self.error(f),
                        mean + beta * std + self.error(f))

            logging.debug('LCB/UCB for f{0} (fidelity {1})'
                            .format(self.transformToDomain(x), f))
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
        logging.debug('Width is now {0}'.format(self.wSchedule[self.wIndex]))

    def transformToDomain(self, x):
        return tuple(x * (self.highs - self.lows) + self.lows)

    def bestQuery(self):
        return self.transformToDomain(self.bestNode.center), self.bestNode.value

    def plotInfo(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2)
        def f(arg):
            x = self.transformToDomain(arg)
            return self.fn(x, self.numFidelities - 1)[0]
        self.model.plotModel(axes[0], f)
        self.space.plotTree(axes[1])
        plt.show()
