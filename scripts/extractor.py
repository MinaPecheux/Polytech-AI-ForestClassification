# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# extractor.py - Various feature extractors to easily parse CSV data
# ==============================================================================
import sys
import csv
import numpy as np
import pandas as pd

def base(data, refs, one_hot=False):
    """Loads data into a Pandas DataFrame by directly setting numerical data and
    (optionally) converting one-hot categorical (or dummies) to integer
    categorical.
    
    WARNING: even though categorical data is (potentially) converted, columns of
    the Pandas DataFrame are not set as 'categorical'. This must be done in your
    script after loading to insure a correct treatment of the dataset by ML
    algorithms!
    
    Parameters
    ----------
    data : numpy.ndarray
        Data to load.
    refs : list(str or list(str, int))
        Feature names: if it is a string, the feature is numerical; otherwise if
        it is a list of string and int, the feature is categorical (the string
        is the name of the feature and the int is the number of one-hot columns,
        i.e. categorical levels, to consider).
    one-hot : bool, optional
        If true, data is left as is. If false (by default), one-hot values are
        converted to integer categorical variables.
    """
    p        = 0
    features = []
    fnames   = []
    cats     = []
    for i, f in enumerate(refs):
        if not isinstance(f, list):
            features.append(data[i])
            fnames.append(f)
            p += 1
        else:
            if one_hot:
                t, n = f
                for j in range(n):
                    fname = t + '_' + str(j)
                    fnames.append(fname)
                    cats.append(fname)
                    features.append(data[data.columns[p+j]])
                p += n
            else:
                t, n  = f
                fnames.append(t)
                cats.append(t)
                d     = data[data.columns[[i for i in range(p,p+n)]]]
                dcols = [i for i in range(1,n+1)]
                features.append(pd.Series(np.int_(dcols)[np.where(d != 0)[1]]))
                p    += n
    df = pd.DataFrame(np.asarray(features).T, columns=fnames)
    return df

def only_numerical(data, refs):
    """Loads data into a Pandas DataFrame by only keeping numerical variables
    (categorical features are dropped).
    
    Parameters
    ----------
    data : numpy.ndarray
        Data to load.
    refs : list(str or list(str, int))
        Feature names: if it is a string, the feature is numerical; otherwise if
        it is a list of string and int, the feature is categorical and is ignored.
    """
    p        = 0
    features = []
    fnames   = []
    for i, f in enumerate(refs):
        if not isinstance(f, list):
            features.append(data[i])
            fnames.append(f)
            p += 1
    return pd.DataFrame(np.asarray(features).T, columns=fnames)

def to_categorical(data, refs, **kwargs):
    """Loads data into a Pandas DataFrame by converting numerical features to
    categorical bins. If a list of thresholds is passed in for each numerical
    variable, they are used to partition the variables. Else, if an integer
    value is passed, it is understood as the number of bins to create.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data to load.
    refs : list(str or list(str, int))
        Feature names: if it is a string, the feature is numerical; otherwise if
        it is a list of string and int, the feature is categorical (the string
        is the name of the feature and the int is the number of one-hot columns,
        i.e. categorical levels, to consider).
    """
    thresholds = kwargs.get('thresholds', 3)
    df         = base(data, refs)
    fnames     = []
    features   = []
    for i, f in enumerate(refs):
        if not isinstance(f, list):
            if isinstance(thresholds, list): bins = thresholds
            else: bins = thresholds+1
            d = pd.cut(df[f], bins, labels=[i for i in range(bins)]).values
            features.append(d)
            fnames.append(f)
        else:
            t, _ = f
            features.append(df[t].values)
            fnames.append(t)
    return pd.DataFrame(np.asarray(features).T, columns=fnames)
