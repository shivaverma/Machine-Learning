#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary

clf_nb = GaussianNB()
clf_nb.fit(features_train, labels_train)
pred_nb = clf_nb.predict(features_test)
acc_nb = accuracy_score(pred_nb, labels_test)
print 'naive bayes accuracy: ' + str(acc_nb*100) + '%'

clf_svm = SVC(kernel='linear')
clf_svm.fit(features_train, labels_train)
pred_svm = clf_svm.predict(features_test)
acc_svm = accuracy_score(pred_svm, labels_test)
print 'support vector machine accuracy: ' + str(acc_svm*100) + '%'

clf_dt = DecisionTreeClassifier(min_samples_split=50)
clf_dt.fit(features_train, labels_train)
pred_dt = clf_dt.predict(features_test)
acc_dt = accuracy_score(pred_dt, labels_test)
print 'decision tree accuracy: ' + str(acc_dt*100) + '%'


clf_knn = KNeighborsClassifier(n_neighbors=1)
clf_knn.fit(features_train, labels_train)
pred_knn = clf_knn.predict(features_test)
acc_knn = accuracy_score(pred_knn, labels_test)
print 'K nearest neighbours accuracy: ' + str(acc_knn*100) + '%'

clf_ab = AdaBoostClassifier()
clf_ab.fit(features_train, labels_train)
pred_ab = clf_ab.predict(features_test)
acc_ab = accuracy_score(pred_ab, labels_test)
print 'adaboost accuracy: ' + str(acc_ab*100) + '%'

clf_rf = RandomForestClassifier(min_samples_split=50)
clf_rf.fit(features_train, labels_train)
pred_rf = clf_rf.predict(features_test)
acc_rf = accuracy_score(pred_rf, labels_test)
print 'random forest accuracy: ' + str(acc_rf*100) + '%'


prettyPicture(clf_nb, features_test, labels_test)
prettyPicture(clf_svm, features_test, labels_test)
prettyPicture(clf_dt, features_test, labels_test)
prettyPicture(clf_knn, features_test, labels_test)
prettyPicture(clf_ab, features_test, labels_test)
prettyPicture(clf_rf, features_test, labels_test)

