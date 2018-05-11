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
    
def replace_na(df, var, method='mean'):
    '''
    Replaces all the NA's in a given column with chosen statistical value.
    
    Input:
        df (pandas dataframe): original dataframe
        var (string): column containing NA values
        method (string): the statistical measure to use to find
            replacement values, default set to 'mean'
    '''
    
    val = getattr(df[var], method)()
    df[var].fillna(replace_var, inplace=True)

def fill_missing(df):
    '''
    Takes a dataframe and replaces all NA's in all columns with the mean.
    
    Input:
        df (pandas dataframe): original dataframe
    '''
    
    na_cols = list(df.loc[:, df.isna().any()].columns)
    for col in na_cols:
        replace_na(df, col) 
        
    return df
    
def na_cols(df):
    '''
    Produces a list of columns in the data frame that have N/A values.
    
    Input:
        df (data frame): data frame of interest
        
    Output:
        list of columns that contain N/A's
    '''
    
    return list(df.loc[:, df.isna().any()].columns)
    
# explore data

def find_top(df, col_of_interest, sort_by='projectid', ascending=False):
    '''
    Find the most common (or least) values in a given column
    
    Inputs:
        df (pandas dataframe): dataframe of interest
        col (str): name of column of interest
        sort_by (pandas series): column to sort dataframe by
        ascending: default to False, will show highest values 
        
    Output:
        output (pandas dataframe): Sorted pandas dataframe
    '''
    grouped = df.groupby(col_of_interest, sort=False).count()
    
    return grouped.sort_values(by=sort_by, ascending=False)

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

def plot_all_line_graphs(df, x_label='', y_label='', x_vals='df.index.values'):
    '''
    Plots all the line graphs for all the dependent variables off a given independent variable
        
        df (pandas data frame): data frame containing values to be graphed
        xlabel (str): label for x-axis, default to empty label
        ylabel (str): label for y-axis, default to empty label
        x_vals (data frame column): column to be used for x values, default to index
    
    '''
    for col in df.columns:
        plot_line_graph(df, col, ylabel=col)

def plot_pie_chart(vals, labels, colors = ["red", "blue", "green", "violet", "orange"]):
    '''
    Takes in values and labels and plots a pie chart
    
    Inputs:
        vals (list of ints): values to be charted
        labels (list of strings): labels for each section
        colors (list of strings): Colors for each section of graph. Must be the same length as the number of vals. 
    '''
    
    if len(colors) < len(vals):
        return "Please insert more colors"
    
    plt.pie(
    vals,
    labels=labels,
    # with no shadows
    shadow=False,
    # with colors
    colors=colors,
    # with one slide exploded out
    #explode=(0, 0, 0, 0, 0.15),
    # with the start angle at 90%
    startangle=90,
    # with the percent listed as a fraction
    autopct='%1.1f%%',
    )

    plt.axis('equal')

    # View the plot
    plt.tight_layout()
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
    
