import time
import numpy as np
import pandas as pd
from pytest import param
from scipy.optimize import minimize


def softmax(parameters, X):
    """
    Args:
        parameters (1d n*r numpy array): r is the number of classes and n is the number of attributes
        X (2d m*n numpy array): m rows where each row is a sample with n attributes

    Returns:
        2d m*r numpy array: each row is a sample with r probabilities
    """
    parameters = parameters.reshape(X.shape[1], -1)
    P = np.exp(X @ parameters)
    return P / np.sum(P, axis=1, keepdims=True)    

def cost(parameters, X, y, lambda_):
    """
    Args:
        parameters (1d n*r numpy array): r is the number of classes and n is the number of attributes
        X (2d m*n numpy array): m rows where each row is a sample with n attributes
        y (1d m numpy array): class labels for each sample
        lambda_ (number): regularization parameter 

    Returns:
        number: cost of the model
    """
    P = softmax(parameters, X)
    return np.sum(np.log(P[np.arange(len(P)), y])) - lambda_ * np.sum(parameters ** 2)


def grad(parameters, X, y, lambda_):
    """
    Args:
        parameters (1d n*r numpy array): r is the number of classes and n is the number of attributes
        X (2d m*n numpy array): m rows where each row is a sample with n attributes
        y (1d m numpy array): class labels for each sample
        lambda_ (number): regularization parameter 

    Returns:
        1d n*r numpy array: gradient of the cost function with respect to the parameters
    """
    P = softmax(parameters, X)
    parameters = parameters.reshape(X.shape[1], -1)
    E = np.zeros_like(P)
    E[np.arange(len(P)), y] = 1
    grad = X.T @ (E-P) - 2 * lambda_ * parameters
    return grad.flatten()

def bfgs(X, y, lambda_):
    # tukaj inicirajte parametere modela
    x0 = np.zeros((X.shape[1], np.max(y) + 1))

    # preostanek funkcije pustite kot je
    res = minimize(lambda pars, X=X, y=y, lambda_=lambda_: -cost(pars, X, y, lambda_),
                   x0,
                   method='L-BFGS-B',
                   jac=lambda pars, X=X, y=y, lambda_=lambda_: -grad(pars, X, y, lambda_),
                   tol=0.00001)
    return res.x


class SoftMaxLearner:

    def __init__(self, lambda_=0, intercept=True):
        self.intercept = intercept
        self.lambda_ = lambda_

    def __call__(self, X, y):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        pars = bfgs(X, y, self.lambda_)
        return SoftMaxClassifier(pars, self.intercept)


class SoftMaxClassifier:

    def __init__(self, parameters, intercept):
        self.parameters = parameters
        self.intercept = intercept

    def __call__(self, X):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        ypred = softmax(self.parameters, X)
        return ypred


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(SoftMaxLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = c(X)
    return results


def test_cv(learner, X, y, k=5):
    """
    Cross-validation prediction
    Primer klica:
        res = test_cv(SoftMaxLearner(lambda_=0.0), X, y)

    Args:
        learner (SoftMaxLearner): learner to use
        X (2d m*n numpy array): m rows where each row is a sample with n attributes
        y (1d m numpy array): class labels for each sample
        k (int): number of folds 
    Returns:
        2d m*r numpy array
    """
    results = []
    indices = np.arange(len(y))
    permutation = []
    for i in range(k):
        X_train = np.delete(X, indices[i::k], axis=0)
        y_train = np.delete(y, indices[i::k])
        X_test = X[indices[i::k]]
        permutation.extend(indices[i::k])
        c = learner(X_train, y_train)
        results.append(c(X_test))

    permuation_inverse = np.argsort(permutation)
    return np.vstack(results)[permuation_inverse]


def CA(real, predictions):
    """
    Args:
        real (1d m numpy array): class label for each sample
        predictions (2d m*r numpy array): each row is a prediction for a sample with r probabilities

    Returns:
        number: classification accuracy
    """
    return np.mean(np.argmax(predictions, axis=1) == real)

def log_loss(real, predictions):
    """
    Args:
        real (1d m numpy array): class label for each sample
        predictions (2d m*r numpy array): each row is a prediction for a sample with r probabilities

    Returns:
        number: log loss
    """
    return np.mean(-np.log(predictions[np.arange(len(predictions)), real]))
