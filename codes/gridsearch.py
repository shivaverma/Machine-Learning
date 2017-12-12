# author: Shiva Verma, Mail: shivajbd@gmail.com
# svm classifier

import numpy as np


def gridsearch(feature_data, label, param):

    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC        # importing gaussian naive bayes
    c = SVC()
    clf = GridSearchCV(c, param)
    clf.fit(feature_data, label)                        # fitting the data
    return clf                               # predicting new data


def accuracy(out, inp):

    from sklearn.metrics import accuracy_score        # to calculate the accuracy
    return accuracy_score(out, inp)


if __name__ == '__main__':

    x = np.array([[12, 0], [13, 0], [2, 4], [3, 5], [6, 7], [6, 23]])
    y = np.array([1, 1, 2, 2, 2, 1])
    point = ([[5, 0], [14, 0]])
    param = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    clf = gridsearch(x, y, param)
    print clf.best_params_


