# author: Shiva Verma, Mail: shivajbd@gmail.com
# k-mean clustering

import numpy as np


def predict_label(feature_data, p):

    from sklearn.cluster import KMeans
    c = KMeans(n_clusters=2)                # initialize with two centers
    c.fit(feature_data)                     # fitting the data
    return c.predict(p)                     # predicting new data


if __name__ == '__main__':

    x = np.array([[12, 0], [13, 0], [2, 4], [3, 5]])
    point = ([[13, 0], [14, 0]])
    pred = predict_label(x, point)
    print pred

