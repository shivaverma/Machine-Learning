#!/usr/bin/python

import pickle
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.
    This is the second step toward building your POI identifier!
    Start by loading/formatting the data...
"""

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier()
tree.fit(features_train, labels_train)
pred = tree.predict(features_test)

print "no of poi predicted: " + str(np.count_nonzero(pred == 1))       # count of '1' in pred array

acc = accuracy_score(pred, labels_test)
print "accuracy: " + str(acc)

p_score = precision_score(labels_test, pred)
print "precision score: " + str(p_score)

r_score = recall_score(labels_test, pred)
print "recall score: " + str(r_score)

