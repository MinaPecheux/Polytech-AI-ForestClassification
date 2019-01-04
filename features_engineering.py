# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# features_engineering.py - Script to test various ideas of features engineering
#                           (only for research purpose since it doesn't better
#                           the model's accuracy)
# ==============================================================================
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from scripts.dataset import Dataset
from scripts.displayer import Displayer
from scripts.utils import query_input

# warn user of long script run...
q = query_input('[Warning] This script can be a bit long to run. Continue?')
if not q: exit()

# useful additional data: ELU codes for forest types
ELU_codes = {1: '2702', 2: '2703', 3: '2704', 4: '2705',
			 5: '2706', 6: '2717', 7: '3501', 8: '3502',
			 9: '4201', 10: '4703', 11: '4704', 12: '4744',
			13: '4758', 14: '5101', 15: '5151', 16: '6101',
			17: '6102', 18: '6731', 19: '7101', 20: '7102',
			21: '7103', 22: '7201', 23: '7202', 24: '7700',
			25: '7701', 26: '7702', 27: '7709', 28: '7710',
			29: '7745', 30: '7746', 31: '7755', 32: '7756',
			33: '7757', 34: '7790', 35: '8703', 36: '8707',
			37: '8708', 38: '8771', 39: '8772', 40: '8776' }

# let user choose what features to engineer
print('Choose features to engineer:')
print('(Pick one or more integer code(s) and separate them with a comma (",") - e.g.: "0" or "0,1,3")')
print('(If you don\'t provide any code, there will be no engineering.)')
print('[0] ADD     - Climatic and geologic zones')
print('[1] REPLACE - "VDist_To_Hydrology" to absolute value')
print('[2] MODIFY  - "Slope" to categorical')
print('[3] ADD     - "Hillshade_Mean"')
print('[4] ADD     - Difference "Hillshade_9am" - "Hillshade_3pm"')
print('[5] MODIFY  - "Hillshade_9am", "Hillshade_Noon" and "Hillshade_3pm" to categorical')
print('[6] ADD     - Euclidean norm "Dist_To_Hydrology"')
answer = input()
if len(answer) == 0: queries = []
else: queries = [int(x) for x in answer.split(',')]

print()

# load data:
# - without balancing
# - with default extraction: get all features + transform one-hot to categorical
dataset = Dataset('covtype.data', debug=True)
# split train / test datasets into features and matching labels
train_data, train_labels = dataset.train(split=True)
test_data, test_labels = dataset.test(split=True)

if len(queries) == 0:
	print('No features to engineer. Showing results on raw data.\n')
else:
	print('Starting to engineer features...')

	for q in queries:
		if q == 0:
			# Idea n°1:
			# ADD features: climatic and geoligic zones
			# -----------------------------------------
			climate_getter = lambda x: ELU_codes[x['Soil_Type']][0]
			geology_getter = lambda x: ELU_codes[x['Soil_Type']][1]
			train_data['Climate'] = train_data.apply(climate_getter, axis=1)
			train_data['Geology'] = train_data.apply(geology_getter, axis=1)
			train_data['Climate'] = train_data['Climate'].astype('category')
			train_data['Geology'] = train_data['Geology'].astype('category')
			test_data['Climate'] = test_data.apply(climate_getter, axis=1)
			test_data['Geology'] = test_data.apply(geology_getter, axis=1)
			test_data['Climate'] = test_data['Climate'].astype('category')
			test_data['Geology'] = test_data['Geology'].astype('category')
		elif q == 1:
			# Idea n°2:
			# REPLACE feature: change "VDist_To_Hydrology" to its absolute value
			# ------------------------------------------------------------------
			train_data['VDist_To_Hydrology_abs'] = np.abs(train_data['VDist_To_Hydrology'])
			test_data['VDist_To_Hydrology_abs'] = np.abs(test_data['VDist_To_Hydrology'])
			train_data.drop(columns=['VDist_To_Hydrology'], inplace=True)
			test_data.drop(columns=['VDist_To_Hydrology'], inplace=True)
		elif q == 2:
			# Idea n°3:
			# MODIFY feature: make "Slope" categorical (with 2 bins separated by
			# a specific threshold)
			# ------------------------------------------------------------------
			train_data['Slope_bin'] = train_data['Slope'] > 30
			test_data['Slope_bin'] = test_data['Slope'] > 30
			# + remove old feature
			train_data.drop(columns=['Slope'], inplace=True)
			test_data.drop(columns=['Slope'], inplace=True)

			train_data['Slope_bin'] = train_data['Slope_bin'].astype('category')
			test_data['Slope_bin'] = test_data['Slope_bin'].astype('category')
		elif q == 3:
			# Idea n°4:
			# ADD feature: "Hillshade_mean"
			# -----------------------------
			train_data['Hillshade_Mean'] = train_data.loc[:,['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(axis=1)
			test_data['Hillshade_Mean'] = test_data.loc[:,['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(axis=1)
		elif q == 4:
			# Idea n°5:
			# ADD feature: difference between "Hillshade_9am" and "Hillshade_3pm"
			# -------------------------------------------------------------------
			train_data['Hill_Diff'] = np.abs(train_data['Hillshade_9am'] - train_data['Hillshade_3pm'])
			test_data['Hill_Diff'] = np.abs(test_data['Hillshade_9am'] - test_data['Hillshade_3pm'])
		elif q == 5:
			# Idea n°6:
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
		elif q == 6:
			# Idea n°7:
			# ADD feature: "Dist_To_Hydrology", ie Euclidean norm of both distances
			# ---------------------------------------------------------------------
			train_data['Dist_To_Hydrology'] = np.sqrt(np.square(train_data['HDist_To_Hydrology']) + np.square(train_data['VDist_To_Hydrology']))
			test_data['Dist_To_Hydrology'] = np.sqrt(np.square(test_data['HDist_To_Hydrology']) + np.square(test_data['VDist_To_Hydrology']))
		else:
			print('Unknown feature engineering code: "{}". Ignoring this query.'.format(q))

	print('Finished engineering features.\n')

# make sure categorical columns are treated right
train_data['Wilderness_Area'] = train_data['Wilderness_Area'].astype('category')
train_data['Soil_Type'] = train_data['Soil_Type'].astype('category')
test_data['Wilderness_Area'] = test_data['Wilderness_Area'].astype('category')
test_data['Soil_Type'] = test_data['Soil_Type'].astype('category')
	
print('Current feature types:')
print('----------------------')
print(train_data.dtypes)

# compute and print classification results
n = 10
print('\nEvaluating model\'s accuracy (RFC with n={} estimators).'.format(n))
print('-------------------------------------------------------')
clf_dt = RandomForestClassifier(n_estimators=n)
clf_dt.fit(train_data, train_labels)
score = clf_dt.score(test_data, test_labels)
feat_imps = clf_dt.feature_importances_

print('Score: {}% ({})'.format(int(round(100.*score)), score))
print('\nFeatures importance:\n')
sort_feats = np.argsort(-feat_imps)
fnames = list(train_data.columns)
for f in sort_feats:
    print('{:20}\t{:.2f}%'.format(fnames[f], 100.*feat_imps[f]))

