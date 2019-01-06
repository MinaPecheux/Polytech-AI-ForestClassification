# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# data_analyzer.py - Basic script to load, display and (grossly) analyze the
#                    project's dataset
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scripts.dataset import Dataset
from scripts.displayer import Displayer
from scripts.utils import query_input

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

# ask user before starting PCA
q = query_input('Run a PCA analysis?')
run_pca = q

# PCA analysis
if run_pca:
    # .. create PCA tool (and display parameters)
    print('[PCA] Running a PCA analysis...')
    print('-------------------------------')
    pca = PCA()
    print('With parameters:\n')
    print(pca)
    print()
    # .. scale data + extract only numerical features
    num_features = df.columns[:10]
    Z  = StandardScaler().fit_transform(np.float_(df.loc[:, num_features].values))
    # .. compute PCA and check n components
    coord = pca.fit_transform(Z)
    n_components = pca.n_components_
    print('PCA extracted {} components.\n'.format(n_components))
    print('1. Explained variance ratio depending on the number of components:\n')
    exp_var = np.cumsum(pca.explained_variance_ratio_)
    print('# components\t%age of variance explained')
    print('-' * 42)
    for i in range(n_components):
        s = '{}% ({:.4f})'.format(int(100.*exp_var[i]), exp_var[i])
        print('{:>12}\t{:>26}'.format(i+1, s))
    print()

    n = Z.shape[0]

    # .. broken-stick test
    print('2. Broken-stick test')
    eigval = pca.singular_values_**2/n
    bs = np.zeros(n_components)
    for i in range(n_components):
    	bs[i] = 1./float(n_components-i)
    bs = np.cumsum(bs)
    bs = bs[::-1]
    print(pd.DataFrame({'Eigenvalues': eigval, 'Thresholds': bs}))
    print()

    # .. representation quality
    # contribution of individuals to total inertia
    print('10 most important individuals:')
    print('- with highest contribution to total inertia')
    print('- with best cos^2 contributions\n')
    ind_contributions = np.sum(Z**2, axis=1)
    best_ind_idx = np.argsort(-ind_contributions)[:10]
    print('Individual ID\tContribution')
    print('-' * 28)
    for i in best_ind_idx:
        s = '{:.3f}'.format(ind_contributions[i])
        print('{:>13}\t{:>12}'.format(i, s))
    print()
    # indivuals representation quality
    cos2 = coord**2
    for i in range(n_components): cos2[:,i] = cos2[:,i]/ind_contributions
    print('Individual ID\t   cos^2_1\t   cos^2_2\t   cos^2_3')
    print('-' * 58)
    for i in best_ind_idx:
        s0 = '{:.3f}'.format(cos2[i,0])
        s1 = '{:.3f}'.format(cos2[i,1])
        s2 = '{:.3f}'.format(cos2[i,2])
        print('{:>13}\t{:>10}\t{:>10}\t{:>10}'.format(i, s0, s1, s2))
    print()

	# .. variable correlation with axes
    sqrt_eigval = np.sqrt(eigval)
    var_corr = np.zeros((n_components, n_components))
    for k in range(n_components):
        var_corr[:,k] = pca.components_[k,:] * sqrt_eigval[k]
    
    fig, axes = plt.subplots(figsize=(8,8))
    for j in range(n_components):
        plt.annotate(dataset.feature_names[j], (var_corr[j,0], var_corr[j,1]))
    
    plt.plot([-1,1], [0,0], color='silver', linestyle='-', linewidth=1)
    plt.plot([0,0], [-1,1], color='silver', linestyle='-', linewidth=1)
    axes.add_artist(plt.Circle((0,0), 1, color='blue', fill=False))

    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    plt.show()

