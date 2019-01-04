# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# rf_gridsearch.py - Hyperparameters optimization for the Random Forest
#                    Classifier with a Grid Search process
# ==============================================================================
# Adapted from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
# ==============================================================================
import numpy as np

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from scripts.dataset import Dataset
from scripts.utils import query_input

# warn user of long script run...
q = query_input('[Warning] This script can be a bit long to run. Continue?')
if not q: exit()

# load data:
# - without balancing
# - with default extraction: get all features + transform one-hot to categorical
dataset = Dataset('covtype.data', debug=True)
# split train / test datasets into features and matching labels
train_data, train_labels = dataset.train(split=True)
test_data, test_labels = dataset.test(split=True)

# MODIFY feature: make "Hillshade_9am", "Hillshade_Noon" and
# "Hillshade_3pm" categorical (with specific thresholds)
# ----------------------------------------------------------
train_data['Hill_9bin'] = train_data['Hillshade_9am'] > 175
test_data['Hill_9bin'] = test_data['Hillshade_9am'] > 175
train_data['Hill_12bin'] = train_data['Hillshade_Noon'] > 200
test_data['Hill_12bin'] = test_data['Hillshade_Noon'] > 200
train_data['Hill_3bin'] = train_data['Hillshade_3pm'] > 150
test_data['Hill_3bin'] = test_data['Hillshade_3pm'] > 150
# + remove old features
train_data.drop(columns=['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], inplace=True)
test_data.drop(columns=['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], inplace=True)

train_data['Hill_9bin'] = train_data['Hill_9bin'].astype('category')
train_data['Hill_12bin'] = train_data['Hill_12bin'].astype('category')
train_data['Hill_3bin'] = train_data['Hill_3bin'].astype('category')
test_data['Hill_9bin'] = test_data['Hill_9bin'].astype('category')
test_data['Hill_12bin'] = test_data['Hill_12bin'].astype('category')
test_data['Hill_3bin'] = test_data['Hill_3bin'].astype('category')

# set the parameters to tune
tuned_parameters = { 
    'n_estimators': [30, 50, 80],
    'max_features': ['auto', None, 'log2'],
    'criterion' : ['gini', 'entropy']
}

# run Grid Search
k = 5
print('Tuning hyper-parameters (with {}-fold cross-validation).\n'.format(k))

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=k, verbose=2,
                   n_jobs=4)
clf.fit(train_data, train_labels)

# print results
print('Best parameters for the development set:\n')
print(clf.best_params_)
print('\nGrid scores on the development set:\n')
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print('{:0.3f} (+/-{:0.03f}) for {}'.format(mean, std * 2, params))

print('\nDetailed classification report:\n')
print('The model is trained on the full development set.')
print('The scores are computed on the full evaluation set.\n')

y_true, y_pred = test_labels, clf.predict(test_data)
print(classification_report(y_true, y_pred))
print()

