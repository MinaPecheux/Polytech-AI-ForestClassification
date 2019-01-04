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
    """Reads basically (numerical or categorical) + Converts dummies to categorical"""
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
    # if one_hot:
    #     for c in cats: df[c] = df[c].astype('bool')
    # else:
    #     for c in cats: df[c] = df[c].astype('category')
    # for c in cats: df[c] = df[c].astype('category')
    return df

def only_numerical(data, refs):
    """Only returns numerical features (drops categorical completely)"""
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
    """Converts to only categorical (with thresholds)"""
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
