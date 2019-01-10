# (Polytech) Machine Learning project: Forest Type classification
This repository contains my work for the Machine Learning class I took at the French school of Engineering Polytech Sorbonne, taught by Patrick Gallinari, Olivier Schwander, Arthur Pajot and Eloi Zablocki.

This project is inspired by the 2014 Kaggle Challenge: "Forest Cover Type Prediction" (check out the dedicated page: https://www.kaggle.com/c/forest-cover-type-prediction). The goal is to train a model to predict as accurately as possible the type of forest cover from a set of various features (elevation, distance to a road, slope...); we are using supervised classification and we have a complete dataset that we can split in training and testing subsets.

After analyzing the dataset, I decided to focus on Decision Trees and Random Forests because I had to work on very unbalanced data with both numerical and categorical features, and also because these types of classifiers are known to be quite good in Kaggle competitions!

## Some info about the repository
This repository contains the report I wrote (in French) for this projet and various scripts:
- the `scripts/` folder provides useful classes and methods to load, shuffle, balance, analyze and display data
- the `outputs/` folder contains some plots that show relevant information about the project's dataset; they were generated thanks to the `Displayer` class I wrote (the `data_analyzer.py` below shows how to use it to generate these plots)
- the other files are small examples of basic usage of scikit-learn's algorithms and of these custom objects, and their application to the project's dataset:
  - `data_analyzer.py` loads the dataset and provides some basic info about it, plus it plots common graphs that are useful for data analysis: the correlation matrix, the piechart of the class repartition, boxplots, histograms... The user can also run a PCA analysis on the dataset if he/she wishes so.
  - `classifiers_comparison.py` is an early trial at classifying this dataset; in this file, I compared the results of several common classifiers of the scikit-learn library to get a first feeling of the next steps
  - `features_engineering.py` allows you to easily check the influence of some feature engineering ideas on the dataset (adding new features, modifying or combining columns...)
  - `dt_or_rf.py` compares the results of a Decision Tree and a Random Forest on the data to select the most promising model
  - `rf_gridsearch.py` tunes the hyperparameters of the Random Forest classifier I focused on

## Tools & Other references
This project is written in Python and mostly relies on 4 libraries:
- NumPy (for data storing and handling): http://www.numpy.org/
- Pandas (idem): https://pandas.pydata.org/pandas-docs/stable/
- scikit-learn (for AI algorithms): https://scikit-learn.org/stable/
- matplotlib (for visualization): https://matplotlib.org/contents.html
