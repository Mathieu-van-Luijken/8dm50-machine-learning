import numpy as np

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
    """
    Determining target vector values for a given Input data matrix and parameter matrix
    :param X: Input data matrix
    :param beta: parameter matrix
    :return: Estimated target vector values
    """
    #Add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    #Calculate the target vector values
    y_out = np.dot(X, beta)
    return y_out

def MSE(y, y_out):
    """
    Calculation of the Mean squared error of two matrices
    :param y: true target values
    :param y_out: estimated target values
    :return: Mean squared error
    """

    #Calculate the difference between the two input matrices
    error_mat = (y - y_out)

    #Square the difference
    error_mat = np.square(error_mat)

    #Calculate the MSE by summing and dividing by the number of elements
    n = error_mat.shape[0]
    summation = np.sum(error_mat)
    mse = summation / n

    return mse

