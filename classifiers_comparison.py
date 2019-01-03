# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# classifiers_comparison.py - Small script to compare the results of a few
#                             common classifiers on the dataset
# ==============================================================================
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import cross_val_score

import extractor
from dataset import Dataset
from utils import query_input

# warn user of long script run...
q = query_input('[Warning] This script can be a bit long to run. Continue ?')
if not q: exit()

# load data:
# - with balancing using under- and over-sampling
# - only keeping numerical features (for classifiers like the KNN)
dataset = Dataset('covtype.data', debug=True,
    autobalance='both', extractor={'func': extractor.only_numerical})

# define classifiers to test
classifiers = [
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=20)),
    ('Neural Net', MLPClassifier(alpha=1)),
    ('AdaBoost', AdaBoostClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('QDA', QuadraticDiscriminantAnalysis()),
    ('Multinomial Logistic Regression', LogisticRegression(
        C=50. / dataset.N_train, multi_class='multinomial',
        penalty='l1', solver='saga', tol=0.1)),
]

# split train / test datasets into features and matching labels
train_data, train_labels = dataset.train(split=True)
test_data, test_labels = dataset.test(split=True)

# run the various classifiers and print results
k = 5
print('Comparing {} classifiers with a {}-fold '
      'cross-validation.'.format(len(classifiers), k))
for clf_name, clf in classifiers:
    print('\n{}:\n'.format(clf_name) + '-' * (len(clf_name)+1)) 
    scores = cross_val_score(clf, train_data, train_labels.values.ravel(), cv=k)
    score = np.mean(scores)
    print('Mean Score: {}% ({})'.format(int(round(100.*score)), score))
