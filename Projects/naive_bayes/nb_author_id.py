from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess

""" 
This is a mini project. It identify email by their authors.
There are two authors Chris and Sara here.
"""

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

acc = accuracy_score(pred, labels_test)
print 'accuracy: ' + str(acc)




