import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.naive_bayes
import sklearn.neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import seaborn as sns

# read data

def read_csv(filepath):
    '''
    Takes a csv and returns a pandas data frame
    
    Input:
        filepath (directory path): file location
        
    Output:
        pandas data frame
    '''
    
    return pd.read_csv(filepath)
    
# pre process data 
    
def replace_na_with_mean(df, var):
    '''
    Replaces all the NA's in a given column with the mean of that column.
    
    Input:
        df (pandas dataframe): original dataframe
        var (string): column containing NA values
    '''
    
    mean_var = main[var].mean()
    main[var].fillna(mean_var, inplace=True)
    
 def fill_missing_with_mean(df):
    '''
    Takes a dataframe and replaces all NA's in all columns with the mean.
    
    Input:
        df (pandas dataframe): original dataframe
    '''
    
    na_cols = list(df.loc[:, df.isna().any()].columns)
    for col in na_cols:
        replace_na_with_mean(df, col) 
        
    return df
    
def na_cols(df):
    '''
    Produces a list of columns in the data frame that have N/A values.
    
    Input:
        df (data frame): data frame of interest
        
    Output:
        list of columns that contain N/A's
    '''
    
    return list(main.loc[:, main.isna().any()].columns)
    
# explore data

def plot_line_graph(df, var, title, xlabel, ylabel):
    '''
    Plots a line graph based on columns in a pandas data frame
    
    Inputs:
        df (pandas data frame): data frame containing values to be graphed
        var (str): column name of dependent variable
        title (str): title for line graph
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
    '''
    
    x = list(df.index.values)
    y = list(df[var].values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.show()
    
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

def find_best_n(max_n, max_params, metric='minkowski', x_train, y_train, x_test, y_test):
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
    

