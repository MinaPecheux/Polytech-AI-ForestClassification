# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# dataset.py - Util class to load data, shuffle it, balance it...
# ==============================================================================
import sys
import csv

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

from .extractor import base as extractor_base

FEATURE_NAMES = [
    'Elevation',
    'Aspect',
    'Slope',
    'HDist_To_Hydrology',
    'VDist_To_Hydrology',
    'HDist_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'HDist_To_Fire_Points',
    'Wilderness_Area',
    'Soil_Type'
]


def ndtype_to_strformat(dtype):
    """Converts a NumPy dtype to common string formatting"""
    name, type = dtype
    if 'u' in type:                                 return '%u'
    elif 'i' in type:                               return '%d'
    elif 'U' in type or 'S' in type or 'a' in type: return '%s'
    elif 'f' in type:                               return '%f'
    else: raise ValueError('No known conversion for NumPy dtype "%s".' % (type))

class Dataset(object):

    """
    Parameters
    ----------
    filename : string
        Path of the file to read data from.
    split_ratio: float, optional
        Percentage of the data to take as test data (the rest is for training).
    shuffle: bool, optional
        If true, data is shuffled before splitting in test/train sets. Otherwise,
        the beginning is used as train set and the rest as test set (according to
        the split ratio).
    """
    def __init__(self, filename, split_ratio=0.2, shuffle=True, extractor=None,
                 autobalance=None, name=None, debug=False):
        self.debug     = debug
        self.name      = 'Dataset' if name is None else name
        self.extractor = extractor_base if extractor is None else extractor['func']
        if extractor is not None:
            kw = extractor
            kw.pop('func')
            self.extractor_kwargs = kw
        else: self.extractor_kwargs = {}
        
        # prepare features array (with specific column names and types)
        self._set_features()
        # read input data
        if self.debug: print('[{}] Reading data...'.format(self.name))
        self.init_data, self.init_labels = self._read_data(filename)
        self.data, self.labels           = self.init_data, self.init_labels
        # create train/test sets
        self.split_ratio = split_ratio
        # if necessary, rebalance set
        if isinstance(autobalance, str): self._rebalance(autobalance)
        # split into smaller parts
        self.split_samples(shuffle)
        if self.debug:
            print('[{}] Created dataset:'.format(self.name))
            print('Train/test ratio: {}%/{}%'.format(int(100.*(1-split_ratio)), int(100.*(split_ratio))))
            print('Total nb of samples: {}'.format(self.N))
            print('\t> Train samples: {}'.format(self.train_data.shape[0]))
            print('\t> Test  samples: {}\n'.format(self.test_data.shape[0]))
                    
    def _set_features(self):
        """
        Defines features name and ``NumPy`` dtype.
        Types are listed here:

        https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html

        Most common are:

        - '?': boolean                          - 'b': (signed) byte
        - 'B': unsigned byte                    - 'i': (signed) integer
        - 'u': unsigned integer                 - 'f': floating-point
        - 'c': complex-floating point           - 'm': timedelta
        - 'M': datetime                         - 'O': (Python) objects
        - 'U': Unicode string                   - 'V': raw data (void)
        - 'S', 'a': zero-terminated bytes (not recommended)
        """
        self.features = [
            'Elevation',
            'Aspect',
            'Slope',
            'HDist_To_Hydrology',
            'VDist_To_Hydrology',
            'HDist_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'HDist_To_Fire_Points',
            ['Wilderness_Area', 4],
            ['Soil_Type', 40]
        ]
        self.feature_names = [x if not isinstance(x, list) else x[0] \
                              for x in self.features]
        
    def _rebalance(self, mode):
        """Rebalances the Dataset to get as many samples in each class. The
        class with the smallest number of samples is the reference (other
        classes will drop samples).
        
        # WARNING: this function should NOT be used after splitting the dataset
        into train/test subsets. Counts will not be right!
        (It must be called during the initialization by passing the flag
        'autobalance=**mode**' to the constructor, where **mode** is a string
        with the value for the 'mode' parameter, see below.)
        
        Parameters
        ----------
        mode : str
            Rebalancing method, among: 'undersampling' (drop samples above a
            certain limit), 'bootstrapping', 'both' (mix of the
            two previous methods).
        """
        if mode == 'undersampling':
            # reassemble data
            d = np.hstack((
                self.init_data,
                self.init_labels.values.reshape(len(self.init_labels), 1)
            ))
            df = pd.DataFrame(d, columns=self.feature_names + ['Class'])
            # group to balance
            g           = df.groupby('Class', as_index=False)
            c           = g.size().min() * 2
            df_balanced = pd.DataFrame(g.apply(
                lambda x: x.sample(c) if x.shape[0] > c else x
            ))
            if self.debug:
                old_n = df.shape[0]
                new_n = df_balanced.shape[0]
                print('[{}] Rebalanced (by "{}") - Dropped {} ({}%) samples.'.format
                      (self.name, mode, old_n - new_n, int(100.*((old_n - new_n)/old_n))))
            # remember new data and labels
            self.data = df_balanced[self.feature_names].reset_index(drop=True)
            self.labels = df_balanced['Class'].reset_index(drop=True)
        elif mode == 'oversampling':
            data, labels = SMOTE().fit_resample(self.init_data, self.init_labels)
            self.data = pd.DataFrame(data, columns=self.feature_names)
            self.labels = pd.DataFrame(labels, columns=['Class'])
            if self.debug:
                old_n = self.init_data.shape[0]
                new_n = data.shape[0]
                print('[{}] Rebalanced (by "{}") - Added {} ({}%) samples.'.format
                      (self.name, mode, new_n - old_n, int(100.*((new_n - old_n)/new_n))))
        elif mode == 'both':
            # reassemble data
            d = np.hstack((
                self.init_data,
                self.init_labels.values.reshape(len(self.init_labels), 1)
            ))
            df = pd.DataFrame(d, columns=self.feature_names + ['Class'])
            # group to balance
            g           = df.groupby('Class', as_index=False)
            c           = int(g.size().max() * 0.3)
            df_balanced = pd.DataFrame(g.apply(
                lambda x: x.sample(c) if x.shape[0] > c else x
            ))
            if self.debug:
                old_n = df.shape[0]
                new_n = df_balanced.shape[0]
                print('[{}] Rebalanced (by "{}")'.format(self.name, mode))
                print('Step 1: Dropped {} ({}%) samples.'.format
                      (old_n - new_n, int(100.*((old_n - new_n)/old_n))))
            data, labels = SMOTE().fit_resample(
                df_balanced[self.feature_names].reset_index(drop=True).values,
                df_balanced['Class'].reset_index(drop=True).values
            )
            self.data = pd.DataFrame(data, columns=self.feature_names)
            self.labels = pd.DataFrame(labels, columns=['Class'])
            if self.debug:
                old_n = new_n
                new_n = data.shape[0]
                print('Step 2: Added {} ({}%) samples.'.format
                      (new_n - old_n, int(100.*((new_n - old_n)/new_n))))

                old_n = self.init_data.shape[0]
                new_n = data.shape[0]
                if new_n < old_n:
                    print('End Result: Dropped {} ({}%) samples.\n'.format
                          (old_n - new_n, int(100.*((old_n - new_n)/old_n))))
                else:
                    print('End Result: Added {} ({}%) samples.\n'.format
                          (new_n - old_n, int(100.*((new_n - old_n)/new_n))))
        else:
            if self.debug: print('[{}] Error: unknown rebalancing method: "{}"'
                                 .format(self.name, mode))

    def _read_data(self, filename):
        """Imports data from a CSV file (one record per line, delimiter is ',').
        Each record has 54 feature columns and a label column.
        e.g.:
        2596,                                                                               (Elevation)
        51,                                                                                 (Aspect)
        3,                                                                                  (Slope)
        258,0,                                                                              (Horizontal/Vertical_Distance_To_Hydrology)
        510,                                                                                (Horizontal_Distance_To_Roadways)
        221,232,148,                                                                        (Hillshade_9am/Noon/3pm)
        6279,                                                                               (Horizontal_Distance_To_Fire_Points)
        1,0,0,0,                                                                            (Wilderness_Area: 4 binary columns)
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,    (Soil_Type: 40 binary columns)
        5                                                                                   (CoverType)

        Parameters
        ----------
        filename : string
            Path of the file to read data from.
        """
        loaded   = pd.read_csv(filename, header=None)
        labels   = loaded[loaded.columns[-1]]
        features = self.extractor(loaded, self.features, **self.extractor_kwargs)
        self.feature_names = list(features.columns)
        return features, labels
        
    def train(self, split=False):
        """Gets the train set from this complete dataset.
        
        Parameters
        ----------
        split : bool
            If true, returns the features and the labels in two different arrays.
            Else, the two are stacked together.
        """
        if split: return [self.train_data, self.train_labels]
        return pd.DataFrame(
            np.hstack((self.train_data, self.train_labels.values.reshape(len(self.train_labels), 1))),
            columns=self.feature_names + ['Class']
        )
    def test(self, split=False):
        """Gets the test set from this complete dataset.
        
        Parameters
        ----------
        split : bool
            If true, returns the features and the labels in two different arrays.
            Else, the two are stacked together.
        """
        if split: return [self.test_data, self.test_labels]
        return pd.DataFrame(
            np.hstack((self.test_data, self.test_labels.values.reshape(len(self.test_labels), 1))),
            columns=self.feature_names + ['Class']
        )
    def dataframe(self, split=False):
        """Gets the complete dataset.
        
        Parameters
        ----------
        split : bool
            If true, returns the features and the labels in two different arrays.
            Else, the two are stacked together.
        """
        if split: return [self.data, self.labels]
        return pd.DataFrame(
            np.hstack((self.data, self.labels.values.reshape(len(self.labels), 1))),
            columns=self.feature_names + ['Class']
        )
        
    def split_samples(self, shuffle=True):
        """
        Parameters
        ----------
        shuffle: bool, optional
            If true, data is shuffled before splitting in test/train sets. Otherwise,
            the beginning is used as train set and the rest as test set (according to
            the split ratio).
        """
        if self.debug:
            print('[{}] Splitting dataset into train/test subsets (with a '
                  '{}%/{}% ratio).'.format(self.name, int(100.*(1.-self.split_ratio)),
                  int(100.*self.split_ratio)))
        self.N           = self.data.shape[0]
        self.N_test      = int(self.split_ratio * self.N)
        self.N_train     = self.N - self.N_test
        
        # idx  = np.arange(self.N)
        idx  = np.arange(self.N)
        if shuffle: np.random.shuffle(idx)
        train_idx = idx[:self.N_train]
        test_idx  = idx[self.N_train:]
        # make train, test datasets
        self.train_data   = self.data.loc[train_idx]
        self.train_labels = self.labels.loc[train_idx]
        self.test_data    = self.data.loc[test_idx]
        self.test_labels  = self.labels.loc[test_idx]
        self.data         = pd.concat([self.train_data, self.test_data])
        self.labels       = pd.concat([self.train_labels, self.test_labels])
        
    def kfold(self, n_splits):
        """
        Performs a K-fold sampling on the dataset and returns each train/test
        subsets as an iterator.
        
        Parameters
        ----------
        n_splits : int
            Number of splits in K-fold sampling.
        """
        kf = KFold(n_splits=n_splits)
        df = self.dataframe()
        for train_idx, test_idx in kf.split(self.dataframe()):
            yield df.loc[train_idx], df.loc[test_idx]
        
    def export(self, filename, drop_header=True):
        """
        Parameters
        ----------
        filename : string
            Base path of the file to export data to. Will be completed with
            '_train' and '_test' suffixes.
        drop_header : bool
            If true, header is not exported in CSV file.
        """
        if '.' in filename: rawname, ext = filename.split('.')
        else: rawname, ext = filename, 'csv'
        # export to specific files
        self.train.to_csv(rawname + '_train.' + ext, sep=',', index=False, header=(not drop_header))
        self.test.to_csv (rawname + '_test.' + ext,  sep=',', index=False, header=(not drop_header))
