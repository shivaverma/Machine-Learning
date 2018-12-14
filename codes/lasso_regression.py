# author: Shiva Verma, Mail: shivajbd@gmail.com
# lasso regression

import numpy as np
from sklearn.linear_model import Lasso


def predict(feature_data1, feature_data2):

    c = Lasso()
    c.fit(feature_data1, feature_data2)             # fitting data
    # print c.score(feature_data1, feature_data2)   # r-square score
    print(c.coef_)                                   # coefficient of each feature


if __name__ == '__main__':

    x = np.array([[3, 2], [3, 1], [21, 5], [7, 21], [10, 32]])
    y = np.array([[1], [2], [12], [3], [5]])
    predict(x, y)
