import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split


def cleat_data(data):

    # filling 'Nan' values with median of age
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Gender'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

    # dropping useless column
    data = data.drop(['SibSp', 'Parch', 'Sex', 'Name',
                      'Cabin', 'Embarked', 'Ticket'], axis=1)
    return data


if __name__ == '__main__':

    train = pd.read_csv('train.csv', header=0)
    train_data = cleat_data(train)
    print train_data
