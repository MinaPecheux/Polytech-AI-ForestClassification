# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# data_analyzer.py - Basic script to load, display and (grossly) analyze the
#                    project's dataset
# ==============================================================================
from scripts.dataset import Dataset
from scripts.displayer import Displayer

# load data:
# - without balancing
# - with default extraction: get all features + transform one-hot to categorical
dataset = Dataset('covtype.data', debug=True)

# extract complete dataframe (train + test sets, with features and labels)
df = dataset.dataframe()
df['Wilderness_Area'] = df['Wilderness_Area'].astype('category')
df['Soil_Type'] = df['Soil_Type'].astype('category')

# display information and basic plots
displayer = Displayer(df)
print(displayer.desc_long)
displayer.plot(['corrmatrix', 'piechart', 'hist', 'boxplot'])
