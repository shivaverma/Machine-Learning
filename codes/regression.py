import numpy as np


def predict(feature_data, feature_test):

    from sklearn.linear_model import LinearRegression
    c = LinearRegression()
    c.fit(feature_data, feature_test)
    return c.coef_


def accuracy(out, inp):

    from sklearn.metrics import accuracy_score
    return accuracy_score(out, inp)


if __name__ == '__main__':

    x = np.array([[12], [12], [54], [3], [41]])
    y = np.array([[1], [3], [12], [3], [1]])
    pred = predict(x, y)
    print pred

