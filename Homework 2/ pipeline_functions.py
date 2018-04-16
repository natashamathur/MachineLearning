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
        
    Modules Required:
        pandas as pd
    '''
    return pd.read_csv(filepath)
    
# pre process data 
    
def replace_na_with_mean(df, var):
    '''
    Replaces all the NA's in a given column with the mean of that column.
    
    Input:
        df (pandas dataframe): original dataframe
        var (string): column containing NA values
        
    Modules Required:
        pandas as pd
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
    
    Modules Required:
        pandas as pd
        matplotlib.pyplot as plt
    '''

    x = list(df.index.values)
    y = list(df[var].values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.show()
    

