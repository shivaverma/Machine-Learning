# author: Shiva Verma, Mail: shivajbd@gmail.com
# principal component axis

import numpy as np


def pca(feature_data):

    from sklearn.decomposition import PCA
    c = PCA(n_components=2)
    c.fit(feature_data)
    return c


if __name__ == '__main__':

    x = np.array([[3, 6, 4], [3, 6, 1], [21, 42, 6], [7, 14, 1], [10, 20, 43]])
    # y = np.array([[1], [2], [12], [3], [5]])
    clf = pca(x)
    print clf.explained_variance_ratio_
    print clf.components_
