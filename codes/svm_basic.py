# author: Shiva Verma, Mail: shivajbd@gmail.com
# svm classifier

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def predict_label(feature_data, label, p):

    c = SVC(kernel='linear')
    c.fit(feature_data, label)                        # fitting the data
    return c.predict(p)                               # predicting new data


def accuracy(out, inp):

    return accuracy_score(out, inp)


if __name__ == '__main__':

    x = np.array([[12, 0], [13, 0], [2, 4], [3, 5]])  # forming an array of points
    y = np.array([1, 1, 2, 2])                        # labeling each point
    point = ([[5, 0], [14, 0]])                       # creating array of points for prediction
    pred = predict_label(x, y, point)                 # result vector is stored in pred
    print(pred)
    test_label = ([2, 1])
    acc = accuracy(pred, test_label)                  # finding the accuracy by comparing to test_label
    print(acc)

