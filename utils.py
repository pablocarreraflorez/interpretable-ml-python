import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def create_logger(stream = True, file = True, file_name = 'logging.log'):
    '''
    Create a logger object for logging.

    Args: 
        stream: bool, flag indicating if we want a stream logger
        file: bool, flag indicating if we want to save the logs in a file
        file_name: string, name of the file in which we are saving the logs
    '''
    # Create the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if stream:
        # Set the format
        formatter_stream = logging.Formatter('%(asctime)s : %(levelname)s %(message)s')

        # Create a handler for showing the logs
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter_stream)

        # Add the handlers to the logger
        logger.addHandler(stream_handler)
    
    if file:
        # Set the format
        formatter_file = logging.Formatter('%(asctime)s %(name)s %(lineno)d:%(levelname)s %(message)s')

        # Create a handler for saving the logs
        file_handler = logging.FileHandler(file_name, mode = 'w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter_file)

        # Add the handlers to the logger
        logger.addHandler(file_handler)

    return logger

def plot_distributions(data, n_cols = 5, show = True):
    '''
    Plot the distribution of features in a dataset.
    
    Args:
        data: pd.DataFrame, dataframe containting the data
        dict_labels: dict, dictionary with the labels for categorical variables
        n_cols: integer, number of cols of the plot
        show: bool, whether to show the plot or not
    
    Returns:
        fig: plt.figure, figure object containing the plots
    '''
    # Define the layout of the plot
    n_rows = len(data.columns.tolist()) // n_cols + 1

    # Instantiate the figure
    fig = plt.figure(figsize = (20, 5*n_rows))

    # Recursively add a subplot for each variable
    for i,var in enumerate(data.columns):

        # Add the subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot feature distribution if numeric
        if data[var].dtype in ['int64','float64']:
            sns.distplot(data[var], bins = 20, kde = False, ax = ax)
            ax.set_title(var + ' distribution', weight = 'bold')
            ax.set_xlabel(var)
            ax.set_ylabel('Frecuency')
            
        # Plot feature counts if categorical
        else:
            sns.countplot(data[var], ax = ax)
            ax.set_title(var + ' distribution', weight = 'bold')
            ax.set_xlabel('')
            ax.set_ylabel('Frecuency')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

    # Formatting    
    fig.tight_layout()

    # Show if necessary
    if show:
        plt.show()

    return fig

def plot_vars_vs_target(data, var_target = 'Target', n_cols = 5, show = True):
    '''
    Plot the scatterplot of features vs. target in a dataset.
    
    Args:
        data: pd.DataFrame, dataframe containting the data
        var_target: string, name of the target variable
        dict_labels: dict, dictionary with the labels for categorical variables
        n_cols: integer, number of cols of the plot
        show: bool, whether to show the plot or not
        
    Returns:
        fig: plt.figure, figure object containing the plots
    '''
    # Define the layout of the plot
    n_rows = len(data.columns.tolist()) // n_cols + 1

    # Instantiate the figure
    fig = plt.figure(figsize = (20, 5*n_rows))

    # Recursively add a subplot for each variable
    for i,var in enumerate(data.columns[:-1]):
        # Add the subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot feature distribution if numeric
        if data[var].dtype in ['int64','float64']:
            sns.scatterplot(x = var, y = var_target, data = data, alpha = 0.3, ax = ax)
            ax.set_title('{} vs. {}'.format(var_target, var), weight = 'bold')
            ax.set_xlabel(var)
            ax.set_ylabel(var_target)
            
        # Plot feature distribution if cateogorical
        elif data[var].dtype == 'category':
            sns.violinplot(x = var, y = var_target, data = data, ax = ax)
            ax.set_title('{} vs. {}'.format(var_target, var), weight = 'bold')
            ax.set_xlabel('')
            ax.set_ylabel(var_target)
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
            
        # Skip feature otherwise
        else:
            pass
        
    # Formatting    
    fig.tight_layout()

    # Show if necessary
    if show:
        plt.show()
    
    return fig