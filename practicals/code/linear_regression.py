import numpy as np
from sklearn.datasets import load_diabetes


def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


    return beta

def y_output(X, beta):
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    y_out = np.dot(X, beta)
    return y_out

def MSE(y, y_out):

    error_mat = (y - y_out)
    error_mat = np.square(error_mat)

    n = error_mat.shape[0]
    sum = np.sum(error_mat)
    mse = (1/n) * sum
    return mse

diabetes = load_diabetes()
X_train = diabetes.data[:300, :]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, :]
y_test = diabetes.target[300:, np.newaxis]

beta = lsq(X_train, y_train)
y_out = y_output(X_test, beta)
MSE = MSE(y_test, y_out)

# print the parameters
print(y_out.shape)
print(y_test.shape)
print(MSE)