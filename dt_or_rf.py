# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# dt_or_rf.py - Simple script to compare the results of a Decision Tree and a
#               Random Forest Classifier on the project's dataset
# ==============================================================================
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from scripts.dataset import Dataset

# load data:
# - without balancing
# - with default extraction: get all features + transform one-hot to categorical
dataset = Dataset('covtype.data', debug=True)
# split train datasets into features and matching labels
train_data, train_labels = dataset.train(split=True)

# MODIFY feature: make "Hillshade_9am", "Hillshade_Noon" and
# "Hillshade_3pm" categorical (with specific thresholds)
# ----------------------------------------------------------
train_data['Hill_9bin'] = train_data['Hillshade_9am'] > 175
train_data['Hill_12bin'] = train_data['Hillshade_Noon'] > 200
train_data['Hill_3bin'] = train_data['Hillshade_3pm'] > 150
# + remove old features
train_data.drop(columns=['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], inplace=True)

train_data['Hill_9bin'] = train_data['Hill_9bin'].astype('category')
train_data['Hill_12bin'] = train_data['Hill_12bin'].astype('category')
train_data['Hill_3bin'] = train_data['Hill_3bin'].astype('category')

# compare the 2 models (with 5-fold cross-validation)
k = 5
print('Comparing Decision Tree and Random Forest Classifier (with {}-fold '
      'cross-validation).'.format(k))
clf_dt = DecisionTreeClassifier()
clf_rf = RandomForestClassifier(n_estimators=20)
scores_dt = cross_val_score(clf_dt, train_data, train_labels, cv=k)
scores_rf = cross_val_score(clf_rf, train_data, train_labels, cv=k)
print('Mean Score of Decision Tree: {}'.format(np.mean(scores_dt)))
print('Mean Score of Random Forest: {}'.format(np.mean(scores_rf)))