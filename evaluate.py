import pandas as pd
import numpy as np
import seaborn as sns
import math
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from math import sqrt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plot_residuals(actual, predicted):
    '''
    Takes in actual and predicted arguments to return a plot of residuals
    '''
    residuals = actual - predicted
    plt.figure(figsize=(10,10))
    plt.axhline(color = 'green')
    plt.scatter(actual, residuals)
    plt.xlabel('actual ($y$)')
    plt.ylabel(r'residual ($\hat{y}-y$)')
    plt.title('Model Residuals')
    return plt.show()

#### Regression errors functions

def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    return ess(actual, predicted) / tss(actual)

def regression_errors(actual, predicted):
    '''
    Takes actual and predicted as arguments and returns the Sum of Squared Errors, 
    Explained Sum of Squares, Total Sum of Squares, Mean Squared Error
    and Root Mean Squared Error
    '''
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    })


def baseline_mean_errors(actual):
    '''
    Takes actual as an argument and returns the baseline Sum of Squared Errors,
    Total Sum of Squares, Mean Squared Error, and Root Mean Squared Error.
    '''
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    '''
    Takes actual and predicted as arguments, returning True if the model performs better 
    than the baseline. Evaluates the model's RMSE.
    '''
    rmse_baseline = rmse(actual, actual.mean())
    rmse_model = rmse(actual, predicted)
    return rmse_model < rmse_baseline