# Copyright 2018 - M. Pecheux
# [Forest Cover Type Classification] MAIN5 Machine Learning Project
# ------------------------------------------------------------------------------
# utils.py - Util class to display data (with a boxplot, a piechart...)
# ==============================================================================
import sys
import csv
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Displayer(object):

    def _make_corrmatrix(disp):
        """Creates a correlation matrix plot from the dataset in the given
        Displayer instance.
        
        Parameters
        ----------
        disp : Displayer
            Displayer to get data from.
        """
        data = disp.dataframe[disp.features]
        print(data.corr())
        corr = np.abs(data.corr())
        cols = corr.columns.values
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.heatmap(corr, xticklabels=cols, yticklabels=cols, cmap='Oranges',
                        annot=True, fmt='.1f', mask=mask, square=True)
        plt.tight_layout()
        
    def _make_piechart(disp):
        """Creates a piechart plot from the dataset in the given Displayer
        instance.
        
        Parameters
        ----------
        disp : Displayer
            Displayer to get data from.
        """
        colors       = sns.color_palette('Set1', disp.NB_CLASSES)
        class_counts = disp.dataframe['Class'].value_counts().sort_index()
        class_counts.plot.pie(figsize=(6,6), colors=colors, autopct='%.2f')
        plt.title('Repartition by Class')
        
    def _set_box_color(bp, colors):
        """(Util subfunction) Sets the colors of the different parts of a
        matplotlib boxplot."""
        for i, color in enumerate(colors):
            plt.setp(bp['boxes'][i],        color=color)
            plt.setp(bp['whiskers'][2*i],   color=color)
            plt.setp(bp['whiskers'][2*i+1], color=color)
            plt.setp(bp['caps'][2*i],       color=color)
            plt.setp(bp['caps'][2*i+1],     color=color)
            plt.setp(bp['medians'][i],      color=color)
    
    def _make_boxplot(disp):
        """Creates a boxplot from the dataset in the given Displayer instance.
        
        Parameters
        ----------
        disp : Displayer
            Displayer to get data from.
        """
        colors  = sns.color_palette('Set1', disp.NB_CLASSES)
        nb_rows = max(1, disp.NB_FEATS // 4)
        nb_cols = ceil(disp.NB_FEATS / nb_rows)
        for i, fname in enumerate(disp.features):
            axes = plt.subplot(nb_rows, nb_cols, i+1)
            b = disp.dataframe.boxplot(fname, by='Class', ax=axes,
                                       return_type='both')[fname]
            Displayer._set_box_color(b.lines, colors)
            b.ax.get_xaxis().set_visible(False)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.065,
                            wspace=0.3, hspace=0.35)
        plt.figlegend(labels=[str(c+1) for c in range(disp.NB_CLASSES)],
                      handles=b.lines['boxes'], loc='lower right',
                      ncol=disp.NB_CLASSES)
    
    def _make_hist(disp):
        """Creates a histogram from the dataset in the given Displayer instance.
        
        Parameters
        ----------
        disp : Displayer
            Displayer to get data from.
        """
        colors    = sns.color_palette('Set1', disp.NB_CLASSES)
        nb_rows   = max(1, disp.NB_FEATS // 4)
        nb_cols   = ceil(disp.NB_FEATS / nb_rows)
        # make temporary copy with cast to numerical values
        df = disp.dataframe.copy()
        non_num = df.dtypes[df.dtypes != 'int64'][df.dtypes != 'float64']
        for i in non_num.index: df[i] = pd.to_numeric(df[i])
        # get groups and make histogram
        df_groups = df.groupby('Class').groups
        for i, fname in enumerate(disp.features):
            axes = plt.subplot(nb_rows, nb_cols, i+1)
            m, M = int(df[fname].min()), int(df[fname].max())
            bins = np.linspace(m, M, 10)
            vals = [df.loc[df_groups[g]][fname] for g in df_groups.keys()]
            plt.hist(vals, color=colors)
            axes.set_xlim([m, M])
            axes.set_title(fname)
        # additional plot settings
        plt.figlegend([str(c+1) for c in range(disp.NB_CLASSES)],
                      ncol=disp.NB_CLASSES, loc='lower right')
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3, hspace=0.5)
        plt.suptitle('Histogram grouped by Class')

    PLOT_FUNCS = {
        'corrmatrix': _make_corrmatrix, 'piechart': _make_piechart,
        'boxplot': _make_boxplot, 'hist': _make_hist,
    }
    
    def __init__(self, dataframe, name=None):
        self.name       = 'Displayer' if name is None else name
        
        if isinstance(dataframe, list):
            if len(dataframe) != 2:
                print('[Error] Could not create Displayer: provide a Pandas '
                      'DataFrame or two to concatenate!')
                sys.exit()
            d = np.hstack((
                dataframe[0], dataframe[1].values.reshape(len(dataframe[1]), 1)
            ))
            cols = list(dataframe[0].columns) + ['Class']
            dataframe = pd.DataFrame(d, columns=cols)
        
        self.dataframe  = dataframe
        self.features   = dataframe.loc[:, dataframe.columns != 'Class'].columns
        self.classes    = dataframe['Class'].unique()
        self.NB_FEATS   = len(self.features)
        self.NB_CLASSES = len(self.classes)
        
    @property
    def desc(self):
        """Provides a (short and simple) description of the given dataset."""
        s  = '[{}] Dataset description:\n'.format(self.name) + '=' * (23 + len(self.name)) + '\n'
        s += '- {} samples, {} features\n'.format(self.dataframe.shape[0], self.NB_FEATS)
        s += '- Samples per class:\n'
        s += self.counts + '\n'
        return s
        
    @property
    def desc_long(self):
        """Provides a (long and detailed) description of the given dataset."""
        s  = '[{}] Dataset description:\n'.format(self.name) + '=' * (23 + len(self.name)) + '\n'
        s += '- {} samples, {} features\n'.format(self.dataframe.shape[0], self.NB_FEATS)
        s += '- Feature names:\n{}\n'.format(', '.join(self.features))
        s += '- Samples per class:\n'
        s += self.counts
        s += '\n\n- Column types:\n'
        s += str(self.dataframe.dtypes) + '\n'
        s += '\n\n- Base information:\n'
        s += str(self.dataframe.describe()) + '\n'
        s += '\n- Head of dataset:\n'
        s += str(self.dataframe.head()) + '\n'
        return s
        
    @property
    def counts(self):
        """Returns a string output of a table represntation of the counts of the
        dataset. The first column contains the classes in the dataset, the second
        column contains the integer counts and the third column contains the
        renormalized (percentage) counts."""
        class_counts      = self.dataframe['Class'].value_counts().sort_index()
        class_counts_norm = self.dataframe['Class'].value_counts(normalize=True).sort_index()
        return str(pd.DataFrame(
            [*zip(class_counts.values, class_counts_norm.values)],
            index=class_counts.index, columns=['Counts', 'Counts (Renormalized)']
        ))
        
    def plot(self, plot_names):
        """Plots the dataset in a specific type of graph, among: "piechart",
        "hist" (for "histogram"), "boxplot" or "corrmatrix" (for "correlation
        matrix")."""
        if isinstance(plot_names, str): plot_names = [plot_names]
        elif not isinstance(plot_names, list):
            print('[{}] Error: plot() expects a string or a list of strings to '
                  'specify the plots to show!'.format(self.name))
            sys.exit()
            
        for i, pname in enumerate(plot_names):
            Displayer.PLOT_FUNCS[pname](self)
            if i < len(plot_names) - 1: plt.figure()
        plt.show()
        