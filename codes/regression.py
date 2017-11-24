import numpy as np


def predict(feature_data, feature_test, points):

    from sklearn.linear_model import LinearRegression
    c = LinearRegression()
    c.fit(feature_data, feature_test)
    return c.predict(points)


def accuracy(out, inp):

    from sklearn.metrics import accuracy_score
    return accuracy_score(out, inp)


if __name__ == '__main__':

    x = np.array([[2], [4], [24], [6], [10]])
    y = np.array([[1], [2], [12], [3], [5]])
    z = np.array([[2], [3]])
    pred = predict(x, y, z)
    print pred

