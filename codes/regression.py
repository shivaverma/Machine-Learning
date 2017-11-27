# author: Shiva Verma, Mail: shivajbd@gmail.com
# linear model regression

import numpy as np


def predict(feature_data1, feature_data2, feature_test):

    from sklearn.linear_model import LinearRegression
    c = LinearRegression()
    c.fit(feature_data1, feature_data2)                # fitting data
    print c.score(feature_data1, feature_data2)        # r-square score
    return c.predict(feature_test)                     # predicting output


if __name__ == '__main__':

    x = np.array([[3], [3], [21], [7], [10]])
    y = np.array([[1], [2], [12], [3], [5]])
    z = np.array([[2], [3]])
    pred = predict(x, y, z)
    print pred

