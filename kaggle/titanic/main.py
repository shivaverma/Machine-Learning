import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def data_clean(data):

    """cleaning data using pandas library"""

    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Gender'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # data['Family'] = data['Parch'] + data['SibSp']
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data = data.drop(['Sex', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    data['Age'] = (data['Age']-data['Age'].min())/(data['Age'].max() - data['Age'].min())
    data['Fare'] = (data['Fare'] - data['Fare'].min())/(data['Fare'].max() - data['Fare'].min())
    # data['Family'] = (data['Family'] - data['Family'].min())/(data['Family'].max() - data['Family'].min())
    # data['Pclass'] = (data['Pclass'] - data['Pclass'].min())/(data['Pclass'].max() - data['Pclass'].min())
    return data


def fun_knn(feature_train, feature_test, label_train, label_test):

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    acc = accuracy_score(pred, label_test)
    print "accuracy of KNN: " + str(acc)


def fun_nb(feature_train, feature_test, label_train, label_test):

    clf = GaussianNB()
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    acc = accuracy_score(pred, label_test)
    print "accuracy of Naive Bais: " + str(acc)


def fun_rf(feature_train, feature_test, label_train, label_test):

    clf = RandomForestClassifier(min_samples_split=13, min_samples_leaf=4)
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    acc = accuracy_score(pred, label_test)
    print "accuracy of Random Forest: " + str(acc)


def fun_dt(feature_train, feature_test, label_train, label_test):

    clf = DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=4)
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    acc = accuracy_score(pred, label_test)
    print "accuracy of Decision Tree: " + str(acc)


def fun_svm(feature_train, feature_test, label_train, label_test):

    clf = SVC(gamma=.09, C=9)
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    acc = accuracy_score(pred, label_test)
    print "accuracy of svm: " + str(acc)


def write_test(feature_train, label_train, feature_test):

    p_id = feature_test['PassengerId']
    feature_test = feature_test.drop(['PassengerId'], axis=1)
    clf = RandomForestClassifier(min_samples_split=13, min_samples_leaf=4)
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    feature_test['Survived'] = pred
    feature_test['PassengerId'] = p_id
    feature_test = feature_test.drop(['SibSp', 'Parch', 'Fare', 'Pclass', 'Age', 'Gender'], axis=1)

    feature_test.to_csv('my_out.csv', index=False)


if __name__ == '__main__':

    train_data = data_clean(train)
    test_data = data_clean(test)
    x = train_data.drop(['PassengerId', 'Survived'], axis=1)
    y = train_data['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    fun_svm(x_train, x_test, y_train, y_test)
    fun_dt(x_train, x_test, y_train, y_test)
    fun_rf(x_train, x_test, y_train, y_test)
    fun_nb(x_train, x_test, y_train, y_test)
    fun_knn(x_train, x_test, y_train, y_test)
    # print x
    write_test(x, y, test_data)

