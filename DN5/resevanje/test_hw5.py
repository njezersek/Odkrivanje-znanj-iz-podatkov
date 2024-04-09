import unittest
import csv

import numpy as np

from hw5 import SoftMaxLearner, softmax, CA, log_loss, test_cv, test_learning


mirisX = np.array([[5.1, 3.5, 1.4, 0.2],
                   [5.4, 3.9, 1.7, 0.4],
                   [5.4, 3.7, 1.5, 0.2],
                   [5.7, 4.4, 1.5, 0.4],
                   [5.4, 3.4, 1.7, 0.2],
                   [5. , 3. , 1.6, 0.2],
                   [4.8, 3.1, 1.6, 0.2],
                   [5. , 3.2, 1.2, 0.2],
                   [5. , 3.5, 1.3, 0.3],
                   [4.8, 3. , 1.4, 0.3],
                   [7. , 3.2, 4.7, 1.4],
                   [5.7, 2.8, 4.5, 1.3],
                   [5. , 2. , 3.5, 1. ],
                   [6.7, 3.1, 4.4, 1.4],
                   [5.9, 3.2, 4.8, 1.8],
                   [6.6, 3. , 4.4, 1.4],
                   [5.5, 2.4, 3.8, 1.1],
                   [6. , 3.4, 4.5, 1.6],
                   [5.5, 2.6, 4.4, 1.2],
                   [5.7, 3. , 4.2, 1.2],
                   [6.3, 3.3, 6. , 2.5],
                   [7.6, 3. , 6.6, 2.1],
                   [6.5, 3.2, 5.1, 2. ],
                   [6.4, 3.2, 5.3, 2.3],
                   [6.9, 3.2, 5.7, 2.3],
                   [7.2, 3.2, 6. , 1.8],
                   [7.4, 2.8, 6.1, 1.9],
                   [7.7, 3. , 6.1, 2.3],
                   [6.7, 3.1, 5.6, 2.4],
                   [6.7, 3. , 5.2, 2.3]])


mirisY = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2])


class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.X = mirisX
        self.y = mirisY
        self.train = self.X[::2], self.y[::2]
        self.test = self.X[1::2], self.y[1::2]

    def test_softmax(self):
        l = SoftMaxLearner(lambda_=0, intercept=False)
        c = l(self.train[0], self.train[1])
        test = self.test[0][::5]
        probs = softmax(c.parameters, test)
        np.testing.assert_almost_equal(probs, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_range(self):
        l = SoftMaxLearner(lambda_=1)
        c = l(self.train[0], self.train[1])
        prob = c(self.test[0])
        self.assertEqual(prob.shape, (15, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_regularization(self):
        l = SoftMaxLearner(lambda_=1.)
        c = l(self.train[0], self.train[1])
        prob1 = c(self.test[0])
        l = SoftMaxLearner(lambda_=10.)
        c = l(self.train[0], self.train[1])
        prob10 = c(self.test[0])
        l = SoftMaxLearner(lambda_=100.)
        c = l(self.train[0], self.train[1])
        prob100 = c(self.test[0])
        def maxdiff(a):
            return np.max(a, axis=1) - np.min(a, axis=1)
        self.assertTrue(np.all(maxdiff(prob1) > maxdiff(prob10)))
        self.assertTrue(np.all(maxdiff(prob10) > maxdiff(prob100)))


def data1():
    X = np.array([[5.0, 3.6, 1.4, 0.2],
                  [5.4, 3.9, 1.7, 0.4],
                  [4.6, 3.4, 1.4, 0.3],
                  [5.0, 3.4, 1.5, 0.2],
                  [5.6, 2.9, 3.6, 1.3],
                  [6.7, 3.1, 4.4, 1.4],
                  [5.6, 3.0, 4.5, 1.5],
                  [5.8, 2.7, 4.1, 1.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def data2():
    X, y = data1()
    X = X[:6]
    y = y[:6]
    return X[:6], y[:6]


class DummyCVLearner:
    """ For CV testing """
    def __call__(self, X, y):
        return DummyCVClassifier(ldata=X)


class YouTestOnTrainingData(Exception): pass


class FoldsNotEqualSize(Exception): pass


class NotAllTested(Exception): pass


class MixedOrder(Exception): pass


class DummyCVClassifier:
    def __init__(self, ldata):
        self.ldata = list(map(list, ldata))

    def __call__(self, X):
        for x in X:
            if list(x) in self.ldata:
                raise YouTestOnTrainingData()
        else:
            return np.array([[sum(x), len(self.ldata)] for x in X])


class TooManyConsecutiveInstances(Exception): pass


class DummyShuffleLearner:
    """ For CV testing """
    def __call__(self, X, y):
        X = X.T
        notinorder = len(np.flatnonzero(np.abs(np.diff(X)) - 1))
        if notinorder < 5:
            raise TooManyConsecutiveInstances
        return lambda X: np.array([[0.5, 0.5] for x in X])


class TestEvaluation(unittest.TestCase):

    def test_ca(self):
        X, y = data1()
        self.assertAlmostEqual(CA(y, [[1, 0]]*len(y)), 0.5)
        self.assertAlmostEqual(CA(y, [[0.5, 1]]*len(y)), 0.5)
        self.assertAlmostEqual(CA(y[:4], [[0.4, 0.6]]*len(y[:4])), 0.0)
        self.assertAlmostEqual(CA(y[4:], [[0.4, 0.6]]*len(y[4:])), 1.0)
        self.assertAlmostEqual(
            CA(y[:6], [[0.4, 0.6]]*len(y[:6])), 2/6.)
        self.assertAlmostEqual(CA(
            y,
            [[0, 1],
             [0.2, 0.8],
             [0, 1],
             [0, 1],
             [1, 0],
             [0.6, 0.4],
             [1, 0],
             [1, 0]]),
            0.0)

    def test_logloss(self):
        X, y = mirisX, mirisY
        sm = SoftMaxLearner(lambda_=0)
        pred = test_learning(sm, X, y)
        ca = log_loss(y, pred)
        self.assertLess(ca, 1.)

    def test_logreg_noreg_learning_ca(self):
        X, y = data1()
        sm = SoftMaxLearner(lambda_=0)
        pred = test_learning(sm, X, y)
        ca = CA(y, pred)
        self.assertAlmostEqual(ca, 1.)

    def test_cv(self):
        for X, y in [data1(), data2()]:
            X_copy = X.copy()
            pred = test_cv(DummyCVLearner(), X, y, k=4)

            if len(y) == 8:
                # on the first DS training data should have 6 instances
                self.assertEqual(pred[0][1], 6)

                # on the first DS all folds should be of equal size
                if len(set([a for _, a in pred])) != 1:
                    raise FoldsNotEqualSize()

            signatures = [a for a, _ in pred]
            if len(set(signatures)) != len(y):
                raise NotAllTested()

            if signatures != list(map(lambda x: sum(list(x)), X_copy)):
                raise MixedOrder()

    def test_cv_softmax(self):
        y = mirisY
        X = mirisX
        pred = test_cv(SoftMaxLearner(), X, y, k=4)
        self.assertIsNotNone(pred)
        self.assertEqual((len(X), len(set(y))), pred.shape)

    def test_cv_shuffled(self):
        """Do not take folds in order
        - shuffle because data is frequently clustered """
        y = mirisY
        X = np.array([[i] for i in range(len(y))])
        pred = test_cv(DummyShuffleLearner(), X, y, k=4)
        self.assertIsNotNone(pred)

class TestFinalSubmissions(unittest.TestCase):

    def test_format(self):
        """ Tests format of your final predictions. """

        with open("final.txt", "rt") as f:
            content = list(csv.reader(f))
            content = [l for l in content if l]

            ids = []

            for i, l in enumerate(content):
                # each line contains 10 columns
                self.assertEqual(len(l), 10)

                # first line is just a description line
                if i == 0:
                    self.assertEqual(l, ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4',
                                         'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
                else:
                    ids.append(int(l[0]))  # first element is an id
                    probs = np.array([float(f) for f in l[1:]])  # the rest are probabilities
                    self.assertLessEqual(np.max(probs), 1)
                    self.assertGreaterEqual(np.min(probs), 0)

            # ids covered the whole range
            self.assertEqual(set(ids), set(range(1, 11878+1)))

    def test_function(self):
        try:
            from hw5 import create_final_predictions
        except ImportError:
            self.fail("Function create_final_predictions does not exists.")

if __name__ == "__main__":
    unittest.main()
