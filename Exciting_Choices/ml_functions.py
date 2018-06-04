from __future__ import division

import data_functions as da
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.naive_bayes
import sklearn.neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

all_models = models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']

# from simpleloop
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
import random
from scipy import optimize
import time
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import itertools
from datetime import datetime
# generate features / predictors

def add_dummy_variable(df, var, dummy_var, lambda_eq):
    '''
    Adds a column with a dummy variable based on a column currently in the data frame. 
    
    Inputs:
        df (pandas data frame): the pandas data frame that contains the current data and variable and that the 
        dummy variable will be added to
        var (str): column name of origin variable
        dummy_var (str): column name for dummy variable
        lambda_eq (lambda equation): equation in the form 'lambda x: x < 23550' to turn the 'var' into the 'dummy_var'
    '''
    
    df[dummy_var] = df[var].apply(lambda_eq)
    
def add_discrete_variable(df, var, discrete_var, num_categories):
    '''
    Adds a column with a discretized variable based on a column currently in the data frame. 
    
    Inputs:
        df (pandas data frame): the pandas data frame that contains the current data and variable and that the 
        dummy variable will be added to
        var (str): column name of origin variable
        discrete_var (str): column name for discretized variable
        num_categories (int): the number of categories to divide the 'var' into. 
    '''
    
    df[discrete_var] = pd.cut(df[var], num_categories)
    
    # build classifier
    
def divide_df(df, var, test_size = 0.3):
    '''
    Divides a data frame into testing and training data frames
    
    Inputs:
        df (pandas data frame): original data frame
        var (string): column name for dependent variable
        test_size (int): weighting for test vs. train division
        
    Outputs:
        x_train, x_test, y_train, y_test = divided data frames to use for classification
    '''
    
    x = df
    y = df[var]
    test_size = 0.3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    
    return x_train, x_test, y_train, y_test

def find_best_n(max_n, max_params, x_train, y_train, x_test, y_test, metric='minkowski'):
    '''
    Finds the optimal number of neighbors and parameters to look at when conducting a 
    K-nearest neighbors classification.
    
    Inputs:
        max_n (int): maximum number of k-neighbors
        max_params (int): maximum number of metric parameters
        metric (str): distance metric
        x_train, y_train, x_test, y_test (pandas data frames): divided data frames to conduct
            training and testing
            
    Output:
        KNeighborsClassifier function with the optimal parameters for the given data. 
    '''
    
    best_score, best_n, best_mp = 0, 0, 0
    for n in range(1, max_n+1):
        for mp in range(1, max_params+1):
            knn = KNeighborsClassifier(n_neighbors=n, metric=metric, metric_params={'p': mp})
            knn.fit(x_train, y_train)
            s = knn.score(x_test, y_test)
            if s > best_score:
                best_n = n
                best_mp = mp
            
    print("The optimal value for k is " + str(best_n) + " with " + str(best_mp) + " metric parameters. ")
    return KNeighborsClassifier(n_neighbors=best_n, metric=metric, metric_params={'p': best_mp})

# evaluate classifier

def evaluate_classifier(classifier, x_train, y_train, x_test, y_test):
    '''
    Evaluates a classifier using sklearn's .score function
    
    Inputs:
        classifier: the classifier that has been set up for analysis
        x_train, y_train, x_test, y_test (pandas data frames): divided data frames to conduct
            training and testing
    '''
    
    print('Training: ')
    print(classifier.score(x_train, y_train))
    print('Testing: ')
    print(classifier.score(x_test, y_test))

### From Rayid's simple loop or magic loop
NOTEBOOK = 0
def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()
    



# Evaluation functions
# calculate precision, recall and auc metrics

def plot_roc(name, probs, true, output_type):
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary

def precision_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()


# Other helper functions

def get_subsets(l):
    subsets = []
    for i in range(1, len(l) + 1):
        for combo in itertools.combinations(l, i):
            subsets.append(list(combo))
    return subsets

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def my_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','baseline','p_at_1', 'p_at_2','p_at_5', 'p_at_10', 'p_at_20', 'p_at_30','p_at_50', 
                                        'r_at_1',
                                        'r_at_2', 'r_at_5',
                                        'r_at_10', 'r_at_20','r_at_30',
                                        'r_at_50'))
    for n in range(1, 2):
        # create training and valdation sets
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 100),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                      recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0)]
                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df

def train_test_over_time(df, features, target='fully_funded', start='Jan 2011'):
    dates = {}
    dates['2011-01-01'] = ['2011-12-31', '2012-06-31']
    dates['2011-07-01'] = ['2012-07-31', '2012-12-31']
    dates['2012-01-01'] = ['2012-12-31', '2013-06-31']

    if start == 'Jan 2011':
        start = '2011-01-01'
        end_train = dates[start][0]
        end_test = dates[start][1]
    elif start == 'Jul 2011':
        start = '2011-07-01'
        end_train = dates[start][0]
        end_test = dates[start][1]
    elif start == 'Jan 2012':
        start = '2012-01-01'
        end_train = dates[start][0]
        end_test = dates[start][1]
    
        
    x_test = da.specify_range(df, 'date_posted', start, end_train)
    x_train = da.specify_range(df, 'date_posted', end_train, end_test)
    x_test, x_train = x_test[features], x_train[features]

    y_test = da.specify_range(df[['date_posted', target]], 'date_posted', start, end_train)
    y_train = da.specify_range(df[['date_posted', target]], 'date_posted', end_train, end_test)
    y_test, y_train = y_test[target], y_train[target]
                         
    return x_test, x_train, y_test, y_train

def find_binary_cols(df):
    binary = []
    for c in df.columns:
        if len(df[c].unique()) <= 2:
            binary.append(c)
    return binary

def turn_to_1_0(df, b):
    for col in b:
        df[col] = df[col].apply(lambda x: 1 if x=='t' else 0)

def category_cols(df, cats):
    for c in cats:
        temp = pd.get_dummies(df[c])
        df = df.join(temp)
        df = df.drop(c, axis=1)
    return df

def classifier_loop(df, features, start, grid_size='test', models_to_run = models_to_run):
    
    # define grid to use: test, small, large
    clfs, grid = define_clfs_params(grid_size)
    
    X_test, X_train, y_test, y_train = train_test_over_time(df, features, start=start)
    # call clf_loop and store results in results_df
    results_df = clf_loop(models_to_run, clfs,grid, X_test, X_train, y_test, y_train)
    
    return results_df


    
